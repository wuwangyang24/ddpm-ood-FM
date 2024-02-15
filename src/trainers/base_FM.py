import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from generative.inferers import DiffusionInferer
from generative.networks.nets import VQVAE, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel

from src.networks import PassthroughVQVAE
from src.utils.simplex_noise import Simplex_CLASS

import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs


class BaseTrainerFM:
    def __init__(self, args):

        # initialize accelarator for DDP
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], 
                                       log_with="wandb", 
                                       project_dir="model/SPFM")
        self.device = self.accelerator.device
        hps = {"Mode": 'ODE_SDE', "num_epochs": args.train.n_epochs, "batch_size": args.train.batch_size}
        self.accelerator.init_trackers(
            args.wandb.project,
            config=hps,
            init_kwargs={
                "wandb": {
                    "entity": args.wandb.entity,
                }
            },
        )

        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        ddpm_channels = 1 if args.data.is_grayscale else 3
        # set up model
        if args.model_type == "small":
            self.model = DiffusionModelUNet(
                spatial_dims=args.data.spatial_dimension,
                in_channels=ddpm_channels,
                out_channels=ddpm_channels,
                num_channels=(128, 256, 256),
                attention_levels=(False, False, True),
                num_res_blocks=1,
                num_head_channels=256,
                with_conditioning=False,
            ).to(self.device)
        elif args.model_type == "big":
            self.model = DiffusionModelUNet(
                spatial_dims=args.data.spatial_dimension,
                in_channels=ddpm_channels,
                out_channels=ddpm_channels,
                num_channels=(256, 512, 768),
                attention_levels=(True, True, True),
                num_res_blocks=2,
                num_head_channels=256,
                with_conditioning=False,
            ).to(self.device)
        else:
            raise ValueError(f"Do not recognise model type {args.model_type}")

        print(f"{sum(p.numel() for p in self.model.parameters()):,} model parameters")

        self.spatial_dimension = args.data.spatial_dimension
        self.image_size = int(args.data.image_size) if args.data.image_size else args.data.image_size
        
        # # set up optimizer, loss, checkpoints
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.train.lr)
        self.save_dir = f"{args.output_dir}/{args.model_name}"
        self.run_dir = Path(args.output_dir) / args.model_name
        
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir, exist_ok=True)

        checkpoint_path = self.run_dir / 'checkpoint.ckpt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.found_checkpoint = True
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            model_state_dict = {k[7:]:v for k, v in checkpoint["model_state_dict"].items()}
            self.model.load_state_dict(model_state_dict)
            self.best_loss = checkpoint["best_loss"]
            # self.accelerator.load_state(self.save_dir)
            print(
                f"---------------Resuming training using checkpoint------------------"
            )
        else:
            self.start_epoch = 0
            self.best_loss = 1000
            self.global_step = 0
            self.found_checkpoint = False

    def save_checkpoint(self, path, epoch, save_message=None):
        if self.accelerator.is_main_process:
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        self.accelerator.wait_for_everyone()
