from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
import os
from glob import glob

class BsplineDataset(Dataset):
    def __init__(self, generator=None, path=None, from_disk=False):
        """
        Args:
            generator: An instance of BsplineMaskGenerator (if from_disk=False)
            path: Path to directory with saved grayscale masks (if from_disk=True)
            from_disk: Whether to load from disk or generate on-the-fly
        """
        self.generator = generator
        self.from_disk = from_disk

        if from_disk:
            self.mask_files = sorted(glob(os.path.join(path, 'mask_*.png')))
        else:
            self.num_samples = 1000  # Number of synthetic samples

    def __len__(self):
        return len(self.mask_files) if self.from_disk else self.num_samples

    def __getitem__(self, idx):
        if self.from_disk:
            # Load grayscale target from disk
            target = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            input_mask = (target > 0).astype(np.float32)  # Binary input
        else:
            # Generate on-the-fly
            target, _ = self.generator.generate()
            target = target.astype(np.float32) / 255.0
            input_mask = (target > 0).astype(np.float32)

        # Convert to PyTorch tensors with shape [1, H, W]
        input_tensor = torch.from_numpy(input_mask).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0)

        return input_tensor, target_tensor
