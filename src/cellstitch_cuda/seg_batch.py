from torch.utils.data import Dataset
import torch
import numpy as np
from instanseg.utils.utils import percentile_normalize


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

        assert (
            3 >= image.dim() >= 2
        ), f"Input image shape {image.shape} is not supported."

        image = torch.atleast_3d(image)

        image = percentile_normalize(image)

        return image
