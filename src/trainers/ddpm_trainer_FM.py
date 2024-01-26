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

from .base import BaseTrainer


class DDPMTrainer_FM(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

        if args.quick_test:
            print("Quick test enabled, only running on a single train and eval batch.")
        self.quick_test = args.quick_test
        self.logger_train = SummaryWriter(log_dir=str(self.run_dir / "train"))
        self.logger_val = SummaryWriter(log_dir=str(self.run_dir / "val"))
        self.num_epochs = args.n_epochs
        self.sigma_min = args.sigma_min
        self.is_rectifiedflow = args.is_rectifiedflow
        self.data_parallel = args.data_parallel
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
            images = self.vqvae_model.encode_stage_2_inputs(batch["image"].to(0))
            
            if self.do_latent_pad:
                with torch.no_grad():
                    images = F.pad(input=images, pad=self.latent_pad, mode="constant", value=0)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                if self.is_rectifiedflow:
                    noise = self.encode(x=images, y=y)
                else:
                    noise = torch.randn_like(images)

                # noise images
                t = torch.rand(len(images), device=images.device, dtype=images.dtype)
                t_ = t[:, None, None, None]  # [B, 1, 1, 1]
                noisy_image = t_ * images + (1 - (1 - self.sigma_min) * t_) * noise
                u = images - (1 - self.sigma_min) * noise
                
                # noisy_image = noisy_image.to(0)
                # t = t.to(0)

                prediction = self.model(x=noisy_image, timesteps=t)
                loss = (prediction-u).square().mean(dim=(1,2,3)).sum()
            
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

            self.logger_train.add_scalar(
                tag="loss", scalar_value=loss.item(), global_step=self.global_step
            )
            if self.quick_test:
                break
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
            images = self.vqvae_model.encode_stage_2_inputs(batch["image"].to(0))
            if self.do_latent_pad:
                images = F.pad(input=images, pad=self.latent_pad, mode="constant", value=0)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True):

                if self.is_rectifiedflow:
                    noise = self.encode(x=images, y=y)
                else:
                    noise = torch.randn_like(images)

                # noise images
                t = torch.rand(len(images), device=images.device, dtype=images.dtype)
                t_ = t[:, None, None, None]  # [B, 1, 1, 1]
                noisy_image = t_ * images + (1 - (1 - self.sigma_min) * t_) * noise
                u = images - (1 - self.sigma_min) * noise

                prediction = self.model(x=noisy_image, timesteps=t)
                loss = (prediction-u).square().mean(dim=(1,2,3)).sum()

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
                fig, ax = plt.subplots(2, 2)
            else:
                num_samples = 8
                fig, ax = plt.subplots(2, 4)
        elif self.spatial_dimension == 3:
            num_samples = 2
            fig, ax = plt.subplots(2, 3)
        noise = torch.randn((num_samples, *tuple(images.shape[1:]))).to(self.device)
        latent_samples = self.inferer.sample(
            input_noise=noise,
            diffusion_model=self.model,
            scheduler=self.scheduler,
            verbose=True,
        )
        if self.do_latent_pad:
            latent_samples = F.pad(
                input=latent_samples, pad=self.inverse_latent_pad, mode="constant", value=0
            )
        samples = self.vqvae_model.decode_stage_2_outputs(latent_samples)
        if self.spatial_dimension == 2:
            for i in range(len(ax.flat)):
                ax.flat[i].imshow(
                    np.transpose(samples[i, ...].cpu().numpy(), (1, 2, 0)), cmap="gray"
                )
                plt.axis("off")
        elif self.spatial_dimension == 3:
            slice_ratios = [0.25, 0.5, 0.75]
            slices = [int(ratio * samples.shape[4]) for ratio in slice_ratios]
            for i in range(num_samples):
                for j in range(len(slices)):
                    ax[i][j].imshow(
                        np.transpose(samples[i, :, :, :, slices[j]].cpu().numpy(), (1, 2, 0)),
                        cmap="gray",
                    )
        self.logger_val.add_figure(tag="samples", figure=fig, global_step=self.global_step)


    # def encode(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
    #     # if y is not None:
    #     func = lambda t, x: self.model(x=x, timesteps=t, **kwargs)

    #     _RTOL = 1e-5
    #     _ATOL = 1e-5
    #     ode_kwargs = dict(
    #         method="dopri5",
    #         rtol=_RTOL,
    #         atol=_ATOL,
    #         adjoint_params=(),
    #     )
    #     return odeint(
    #         func,
    #         x,
    #         # 0.0,
    #         torch.tensor([1.0, 0.0], device=x.device, dtype=x.dtype),
    #         # phi=self.parameters(),
    #         **ode_kwargs,
    #     )[-1]

    def decode(self, z: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor: 
        func = lambda t, x: self.model(x=x, timesteps=t, **kwargs)
        _RTOL = 1e-5
        _ATOL = 1e-5
        ode_kwargs = dict(
            method="euler",
            rtol=_RTOL,
            atol=_ATOL,
            adjoint_params=(),
            options=dict(step_size=0.01),
        )
        return odeint(
            func,
            z,
            # 0.0,
            torch.tensor([0.0, 1.0], device=z.device, dtype=z.dtype),
            # phi=self.parameters(),
            **ode_kwargs,
        )[-1]
