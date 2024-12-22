from torch.utils.data import Dataset
import torch
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        epsilon: float = 1e-3
        percentile = 0.1

        image = self.image_list[idx]
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint16:
                image = image.astype(np.int32)
            image = torch.from_numpy(image).float()

        image = image.squeeze()

        image = torch.atleast_3d(image)

        for c in range(image.shape[0]):
            if image.is_cuda or image.is_mps:
                (p_min, p_max) = torch.quantile(image, torch.tensor([percentile / 100, (100 - percentile) / 100], device = image.device))
            else:
                (p_min, p_max) = np.percentile(image.cpu(), [percentile, 100 - percentile])
            image[c] = (image[c] - p_min) / max(epsilon, p_max - p_min)
        return image
