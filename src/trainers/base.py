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


class BaseTrainer:
    def __init__(self, args):

        # initialise DDP if run was launched with torchrun
        if "LOCAL_RANK" in os.environ:
            print("Setting up DDP.")
            self.ddp = True
            # disable logging for processes except 0 on every node
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0:
                f = open(os.devnull, "w")
                sys.stdout = sys.stderr = f

            # initialize the distributed training process, every GPU runs in a process
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.ddp = False
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        print(f"Arguments: {str(args)}")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # set up model
        if args.model_type == "small":
            self.model = DiffusionModelUNet(
                spatial_dims=args.spatial_dimension,
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
                spatial_dims=args.spatial_dimension,
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

        self.scaler = GradScaler()
        self.spatial_dimension = args.spatial_dimension
        self.image_size = int(args.image_size) if args.image_size else args.image_size

        # set up optimizer, loss, checkpoints
        self.run_dir = Path(args.output_dir) / args.model_name
        # can choose to resume/reconstruct from a specific checkpoint
        if args.ddpm_checkpoint_epoch:
            checkpoint_path = self.run_dir / f"checkpoint_{int(args.ddpm_checkpoint_epoch)}.pth"
        # otherwise select the best checkpoint
        else:
            checkpoint_path = self.run_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.found_checkpoint = True
            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            print(
                f"Resuming training using checkpoint {checkpoint_path} at epoch {self.start_epoch}"
            )
        else:
            self.start_epoch = 0
            self.best_loss = 1000
            self.global_step = 0
            self.found_checkpoint = False

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
        if checkpoint_path.exists():
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # wrap the model with DistributedDataParallel module
        if self.ddp:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.device], find_unused_parameters=True
            )

    def save_checkpoint(self, path, epoch, save_message=None):
        if self.ddp and dist.get_rank() == 0:
            # if DDP save a state dict that can be loaded by non-parallel models
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
        if not self.ddp:
            checkpoint = {
                "epoch": epoch + 1,  # save epoch+1, so we resume on the next epoch
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            }
            print(save_message)
            torch.save(checkpoint, path)
