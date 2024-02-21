import pandas as pd
import torch.distributed as dist
from torchvision import transforms
from src.data.get_dataset_celebA import CelebADataset
import torch


def get_training_data_loader_celebA(
    batch_size: int,
    root_dir: str,
    image_size=256,
    num_workers: int = 1,
):

    transform = transforms.Compose([transforms.Resize((image_size, image_size), transforms.ToTensor()])
    ds_train = CelebADataset(root_dir, transform, True)
    ds_val = CelebADataset(root_dir, transform, False)

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
