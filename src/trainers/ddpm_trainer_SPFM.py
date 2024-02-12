import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.get_train_and_val_dataloader import get_training_data_loader
from src.utils.simplex_noise import generate_simplex_noise
from torchdiffeq import odeint_adjoint as odeint

from .base_FM import BaseTrainer_FM
from src.utils.__init__ import create_transport


class DDPMTrainer_FM(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.image_size = args.data.image_size
        self.num_epochs = args.n_epochs
        self.sigma_min = args.sigma_min
        self.step_size = args.model_step_size
        self.train_loader, self.val_loader = get_training_data_loader(
            batch_size=args.batch_size,
            training_ids=args.training_ids,
            validation_ids=args.validation_ids,
            augmentation=bool(args.augmentation),
            num_workers=args.num_workers,
            cache_data=bool(args.cache_data),
            is_grayscale=bool(args.is_grayscale),
            spatial_dimension=args.spatial_dimension,
            image_size=self.image_size,
            image_roi=args.image_roi,
        )
        self.transport = create_transport(
            path_type = args.train_path_type,
            prediction = args.train.prediction,
            loss_weight = args.train.loss_weight,
            train_eps = args.train.train_eps,
            sample_eps = args.train.sample_eps
            )
        wandb.login(key=args.wandb.key)
        self.run = wandb.init(entity=args.wandb.entity, project=args.wandb.project)

    def train(self, args):
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.data_parallel:
                self.model = torch.nn.DataParallel(self.model)
                self.model = self.model.to(0)
            self.model.train()
            epoch_loss = self.train_epoch(epoch)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss

                self.save_checkpoint(
                    self.run_dir / "checkpoint.pth",
                    epoch,
                    save_message=f"Saving checkpoint for model with loss {self.best_loss}",
                )

            if args.checkpoint_every != 0 and (epoch + 1) % args.checkpoint_every == 0:
                self.save_checkpoint(
                    self.run_dir / f"checkpoint_{epoch+1}.pth",
                    epoch,
                    save_message=f"Saving checkpoint at epoch {epoch+1}",
                )

            if (epoch + 1) % args.eval_freq == 0:
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
            images = batch["image"].to(0)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):

                # pure noise
                noise = torch.randn_like(images)
                model_kwargs = dict()
                loss_dict = self.transport.training_losses(self.model, x, model_kwargs)
                loss = loss_dict['loss'].mean()

 
            
            self.scaler.scale(loss).backward()
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
        for step, batch in progress_bar:
            images = batch["image"].to(0)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                noise = torch.randn_like(images)

                # pure noise
                noise = torch.randn_like(images)
                model_kwargs = dict()
                loss_dict = self.transport.training_losses(self.model, x, model_kwargs)
                loss = loss_dict['loss'].mean()

            self.logger_val.add_scalar(
                tag="loss", scalar_value=loss.item(), global_step=global_val_step
            )
            epoch_loss += loss.item()
            val_steps += images.shape[0]
            global_val_step += images.shape[0]
            progress_bar.set_postfix(
                {
                    "loss": epoch_loss / val_steps,
                }
            )

        # get some samples
        image_size = images.shape[2]
        if self.spatial_dimension == 2:
            if image_size >= 128:
                num_samples = 4
            else:
                num_samples = 8
        elif self.spatial_dimension == 3:
            num_samples = 2
        noise = torch.randn((num_samples, *tuple(images.shape[1:]))).to(self.device)
        recons = self.decode(z=noise, y=None)
        recons.clamp_(0,1)
        examples = []
        for i in range(num_samples):
            img = np.transpose(recons[i,...].cpu().numpy(), (1,2,0))
            recon = wandb.Image(img)
            examples.append(img)
        self.run.log({'reconstructions': examples})


    def decode(self, z: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor: 
        func = lambda t, x: self.model(x=x, timesteps=torch.tensor([t]*len(x)).to(0), **kwargs)
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
