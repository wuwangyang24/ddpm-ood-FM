import argparse
import ast
from omegeconf import OmegaConf
from scr.trainers.DDPMTrainer_SPFM import DDPMTrainer_SPFM
import os


def main():
    args = OmegaConf.load('config.yaml')
    trainer = DDPMTrainer_SPFM(args)
    trainer.train(args)
    return args


# to run using DDP, run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0  train_ddpm.py --args
if __name__ == "__main__":
    main()
