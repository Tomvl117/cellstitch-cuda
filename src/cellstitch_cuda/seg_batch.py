from torch.utils.data import Dataset
import torch
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint16:
                image = image.astype(np.int32)
            image = torch.from_numpy(image).float()

        image = image.squeeze()

        image = torch.atleast_3d(image)
        return image
