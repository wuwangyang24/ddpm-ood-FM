import os
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dor = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)
        self.train = train
        self.val_split = val_split
        if train:
            self.train_images = self.images[:182637]
        else:
            self.val_images = self.images[182638:202599]

    def __len__(self):
        if self.if_train:
            return len(self.train_images)
        else:
            return len(self.val_images)

    def __getitem__(self, idx):
        if train:
            img_name = os.path.join(self.root_dir, self.train_images[idx])
        else:
            img_name = os.path.join(self.root_dir, self.val_images[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image