import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.data.get_train_and_val_dataloader_celebA import get_training_data_loader_celebA
from torchdiffeq import odeint_adjoint as odeint

from .base_FM import BaseTrainerFM
from src.utils.__init__ import create_transport
from src.utils.transport import Sampler
from torchmetrics.image.fid import FrechetInceptionDistance

def out2img(samples):
    return torch.clamp(255*samples, 0, 255).to(dtype=torch.uint8, device='cuda')

class DDPMTrainer_ODE_SDE(BaseTrainerFM):
    def __init__(self, args):
        super().__init__(args)
        ## data config
        self.num_epochs = args.train.n_epochs
        self.sigma_min = args.model.sigma_min
        if args.data.celebA:
            self.train_loader, self.val_loader = get_training_data_loader_celebA(
                batch_size=args.train.batch_size,
                root_dir=args.data.datadir_celebA
            )
        else:
            self.train_loader, self.val_loader = get_training_data_loader(
                batch_size=args.train.batch_size,
                training_ids=args.data.training_ids,
                validation_ids=args.data.validation_ids,
                is_grayscale=bool(args.data.is_grayscale),
                spatial_dimension=args.data.spatial_dimension
            )
        # use accelerater for ddp
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.val_loader)
        self.step_size = args.model.step_size
        self.checkpoint_every = args.train.checkpoint_every
        self.eval_freq = args.train.eval_freq
        
        # create transport type
        self.transport = create_transport(
            path_type=args.train.path_type,
            prediction=args.train.prediction,
            loss_weight=args.train.loss_weight,
            train_eps=args.train.train_eps,
            sample_eps=args.train.sample_eps)
        #create sampler
        self.transport_sampler = Sampler(self.transport)
        self.ode_x2z = self.transport_sampler.sample_ode(num_steps=50, reverse=True)
        self.sde_z2x = self.transport_sampler.sample_sde(num_steps=50, sampling_method='Euler')
        #fid
        self.fid = FrechetInceptionDistance(feature=2048,
                                            reset_real_features=True,
                                            normalize=False,
                                            sync_on_compute=True
                                           ).to(self.device)
    def train(self, args):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            if epoch % 100 == 0:
                if self.accelerator.is_main_process:
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "opt": self.optimizer.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "best_loss": self.best_loss,
                        "global_step": self.global_step
                    }
                    torch.save(checkpoint, f"{self.save_dir}/checkpoint_{epoch}.pt")
                    print("checkpoint is saved at epoch {epoch}")
                self.accelerator.wait_for_everyone()
            if (epoch + 1) % self.eval_freq  == 0:
                self.model.eval()
                self.val_epoch(epoch)
        print("Training completed.")
        self.accelerator.end_training()

    def train_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            ncols=70,
            position=0,
            leave=True,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        epoch_loss = 0
        epoch_step = 0
        self.model.train()
        for step, batch in progress_bar:
            images = batch['image'].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            model_kwargs = dict()
            loss_dict = self.transport.training_losses(self.model, images, model_kwargs)
            loss = loss_dict['loss'].mean()

            self.accelerator.backward(loss)
            self.optimizer.step()
            epoch_loss += loss.item()
            self.global_step += images.shape[0]
            epoch_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / epoch_step,
                }
            )
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            avg_loss = loss.item() / 4
            if self.accelerator.is_main_process:
                self.accelerator.log({"training_loss_step": avg_loss})
                # compute FID
                if epoch % 50 == 0 and epoch >0:
                self.fid.update(out2img(images), real=True)
                self.fid.update(out2img(self.sample(images.shape[0])), real=False)
        if self.accelerator.is_main_process:
            self.accelerator.log({"FID": self.fid.compute().item()})
        epoch_loss = epoch_loss / epoch_step
        return epoch_loss

    @torch.no_grad()
    def val_epoch(self, epoch):
        progress_bar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            ncols=70,
            position=0,
            leave=True,
            desc="Validation",
        )
        epoch_loss = 0
        global_val_step = self.global_step
        val_steps = 0
        count_fid_sample = 0
        if_fid = False
        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            ## pure noise
            # x = torch.randn_like(images)
            model_kwargs = dict()
            loss_dict = self.transport.training_losses(self.model, images, model_kwargs)
            loss = loss_dict['loss'].mean()

            # sampling
            _z = self.ode_x2z(images, self.model, **model_kwargs)[-1]
            samples = self.sde_z2x(_z, self.model, **model_kwargs)[-1]

            epoch_loss += loss.item()
            val_steps += images.shape[0]
            global_val_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / val_steps,
                }
            )
            
        # dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        avg_loss = epoch_loss / 4
        if self.accelerator.is_main_process:
            self.accelerator.log({"val_loss_step": avg_loss})
        
        ## sample every n epochs:
        # get some samples
        image_size = images.shape[2]
        if self.spatial_dimension == 2:
            if image_size >= 128:
                num_samples = 8
            else:
                num_samples = 16
        elif self.spatial_dimension == 3:
            num_samples = 2
        
        examples = []
        samples = samples[:num_samples]
        for i in range(len(samples)):
            img = np.transpose(samples[i,...].detach().cpu().numpy(), (1, 2, 0))
            recon = wandb.Image(img)
            examples.append(recon)
        self.accelerator.log({"reconstruction": examples})


    def decode(self, z: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor: 
        func = lambda t, x: self.model(x=x, timesteps=torch.tensor([t]*len(x)).to(self.device), **kwargs)
        _RTOL = 1e-5
        _ATOL = 1e-5
        ode_kwargs = dict(
            method="euler",
            rtol=_RTOL,
            atol=_ATOL,
            adjoint_params=(),
            options=dict(step_size=self.step_size),
        )
        return odeint(
            func,
            z,
            # 0.0,
            torch.tensor([0.0, 1.0], device=z.device, dtype=z.dtype),
            # phi=self.parameters(),
            **ode_kwargs,
        )[-1]


    def sample(self, num_samples):
        sampler = Sampler(self.transport)
        sample_fn = sampler.sample_ode()
        z = torch.randn(num_samples, 3, 32, self.image_size).to(self.device)
        model_kwargs = dict()
        samples = sample_fn(z, self.model, **model_kwargs)[-1]
        return samples
