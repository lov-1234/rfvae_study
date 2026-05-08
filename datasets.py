import numpy as np
import torch
from torch.utils.data import Dataset


class DSpritesDataset(Dataset):
    def __init__(self, data_path, device, transform=None):
        self.data = self.load_data(data_path)
        self.device = device
        self.transform = transform

    @staticmethod
    def load_data(data_path):
        print("Loading DSprites Dataset...")
        data = np.load(data_path)
        # Access the 'imgs' key to get the actual image array
        imgs = data['imgs']
        print("Loaded DSprites Dataset")
        return imgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(image).float()
        # Add a channel dimension (C, H, W) for grayscale images
        image = image.unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image
