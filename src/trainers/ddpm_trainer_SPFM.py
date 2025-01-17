import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
import wandb

from src.data.get_train_and_val_dataloader import get_training_data_loader
from torchdiffeq import odeint_adjoint as odeint

from .base_FM import BaseTrainerFM
from src.utils.__init__ import create_transport
from src.utils.transport import Sampler
from torchmetrics.image.fid import FrechetInceptionDistance

def out2img(samples):
    return torch.clamp(255*samples, 0, 255).to(dtype=torch.uint8, device='cuda')

class DDPMTrainer_SPFM(BaseTrainerFM):
    def __init__(self, args):
        super().__init__(args)
        ## data config
        self.image_size = args.data.image_size
        self.num_epochs = args.train.n_epochs
        self.sigma_min = args.model.sigma_min
        self.train_loader, self.val_loader = get_training_data_loader(
            batch_size=args.train.batch_size,
            training_ids=args.data.training_ids,
            validation_ids=args.data.validation_ids,
            is_grayscale=bool(args.data.is_grayscale),
            spatial_dimension=args.data.spatial_dimension,
            image_size=self.image_size,
        )
        self.step_size = args.model.step_size
        self.checkpoint_every = args.train.checkpoint_every
        self.eval_freq = args.train.eval_freq
        wandb.login(key=args.wandb.key)
        self.run = wandb.init(entity=args.wandb.entity, project=args.wandb.project)
        wandb.watch(self.model, log='all', log_freq=10)
        
        # create transport type
        self.transport = create_transport(
            path_type=args.train.path_type,
            prediction=args.train.prediction,
            loss_weight=args.train.loss_weight,
            train_eps=args.train.train_eps,
            sample_eps=args.train.sample_eps)
        #fid
        self.fid = FrechetInceptionDistance(feature=64,
                                            reset_real_features=True,
                                            normalize=False,
                                            sync_on_compute=True
                                           ).to(self.device)
    def train(self, args):

        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            self.run.log({"epoch":epoch, "train_loss_epoch": epoch_loss})
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                self.save_checkpoint(
                    self.run_dir / "checkpoint.pth",
                    epoch,
                    save_message=f"Saving checkpoint for model with loss {self.best_loss}",
                )

            if self.checkpoint_every != 0 and (epoch + 1) % self.checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir / f"checkpoint_{epoch+1}.pth",
                    epoch,
                    save_message=f"Saving checkpoint at epoch {epoch+1}",
                )

            if (epoch + 1) % self.eval_freq  == 0:
                self.model.eval()
                self.val_epoch(epoch)
        print("Training completed.")
        if self.ddp:
            dist.destroy_process_group()

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
            with autocast(enabled=True):
                ## pure noise
                model_kwargs = dict()
                loss_dict = self.transport.training_losses(self.model, images, model_kwargs)
                loss = loss_dict['loss'].mean()
                self.run.log({"training_loss_step":loss.item()})
                
            self.scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()
            self.global_step += images.shape[0]
            epoch_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / epoch_step,
                }
            )
            
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
        fid_score = 0
        fid_steps = 0
        for step, batch in progress_bar:
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                ## pure noise
                # x = torch.randn_like(images)
                model_kwargs = dict()
                loss_dict = self.transport.training_losses(self.model, images, model_kwargs)
                loss = loss_dict['loss'].mean()
                samples = self.sample(len(images))
                torch.clamp_(samples, 0., 1.)
                self.fid.update(out2img(images), real=True)
                self.fid.update(out2img(samples), real=False)

            epoch_loss += loss.item()
            val_steps += images.shape[0]
            global_val_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / val_steps,
                }
            )
        self.run.log({"val_loss": epoch_loss / val_steps})
        self.run.log({"fid_val": self.fid.compute().item()})

        ## sample every n epochs:
        # get some samples
        image_size = images.shape[2]
        if self.spatial_dimension == 2:
            if image_size >= 128:
                num_samples = 4
            else:
                num_samples = 8
        elif self.spatial_dimension == 3:
            num_samples = 2
        # noise = torch.randn((num_samples, *tuple(images.shape[1:]))).to(self.device)
        # recons = self.decode(z=noise, y=None)
        # recons.clamp_(0, 1)
        # examples = []
        # for i in range(num_samples):
        #     img = np.transpose(recons[i,...].cpu().numpy(), (1, 2, 0))
        #     recon = wandb.Image(img)
        #     examples.append(recon)
        # self.run.log({"reconstruction": examples})
        
        examples = []
        samples = samples[:num_samples]
        for i in range(len(samples)):
            img = np.transpose(samples[i,...].detach().cpu().numpy(), (1, 2, 0))
            recon = wandb.Image(img)
            examples.append(recon)
        self.run.log({"reconstruction": examples})


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
        z = torch.randn(num_samples, 3, self.image_size, self.image_size).to(self.device)
        model_kwargs = dict()
        samples = sample_fn(z, self.model, **model_kwargs)[-1]
        return samples