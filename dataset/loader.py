import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from .augmentation import SegmentationAugmentation

class TumorDataset(Dataset):
    def __init__(self, root_dir, img_size , augment =None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment


        self.images = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".png") and not f.endswith("_mask.png")
        ])

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = img_name.replace(".png", "")
        mask_name = f"{base}_mask.png"

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.augment is not None:
            image, mask = self.augment(image, mask)

        return image, mask


class DatasetLoader:
    def __init__(self, root_dir, img_size=128, batch_size=16, shuffle=True, num_workers=4, pin_memory=True , augment = False):
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.augment = SegmentationAugmentation(img_size) if augment else None

        self.dataset = TumorDataset(
            self.root_dir, 
            self.img_size,
            augment= self.augment
        )

        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return iter(self.loader)

    def get_loader(self):
        return self.loader
