from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch



class SplinePointDataset(Dataset):
    """
    Custom PyTorch Dataset for loading mask images and their corresponding B-spline control points.
    """
    def __init__(self, root, normalize=True, img_size=(256, 256), num_pts=200):
        """
        Args:
            root (str): Root directory of the dataset, containing 'images' and 'labels' subfolders.
            img_size (tuple): The target size to resize images to.
            num_pts (int): The number of points defining the spline.
        """
        self.root = root
        self.img_size = img_size
        self.num_pts = num_pts
        self.normalize = normalize

        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids = sorted(os.listdir(self.img_dir))
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample (image and spline points).
        """
        id_ = self.ids[idx].split(".")[0]
        mask = Image.open(os.path.join(self.img_dir, f"{id_}.png")).convert("L")
        mask = self.transform(mask)  # Shape: (1, H, W)
        lbl = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = lbl["spline"]  # Shape: (num_pts, 2)
        # Normalize spline points
        if self.normalize:
            spline = self.normalize_spline(torch.from_numpy(spline).float())
        return mask, spline

    def normalize_spline(self, spline: torch.Tensor) -> torch.Tensor:
        """
        Normalize the spline points to the range [0, 1].

        Args:
            spline (torch.Tensor): The spline points tensor of shape (num_pts, 3).

        Returns:
            torch.Tensor: Normalized spline points.
        """
        W, H = self.img_size
        spline[:, 0] /= (W - 1)
        spline[:, 1] /= (H - 1)
        return spline

class SplineEndPointDataset(Dataset):
    """
    Custom PyTorch Dataset for loading mask images and their corresponding B-spline control points.
    """
    def __init__(self, root, normalize=True, img_size=(256, 256), num_pts=200):
        """
        Args:
            root (str): Root directory of the dataset, containing 'images' and 'labels' subfolders.
            img_size (tuple): The target size to resize images to.
            num_pts (int): The number of points defining the spline.
        """
        self.root = root
        self.img_size = img_size
        self.num_pts = num_pts
        self.normalize = normalize

        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids = sorted(os.listdir(self.img_dir))
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample (image and spline points).
        """
        id_ = self.ids[idx].split(".")[0]
        mask = Image.open(os.path.join(self.img_dir, f"{id_}.png")).convert("L")
        # repeat the mask to 3 channels
        mask = self.transform(mask)  # Shape: (1, H, W)
        lbl = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = lbl["spline"]  # Shape: (num_pts, 2)
        # Normalize spline points
        if self.normalize:
            spline = self.normalize_spline(torch.from_numpy(spline).float())
        # get end points
        end_points = spline[[0,-1], :]
        # # plot the mask and end points
        # import matplotlib.pyplot as plt

        # # Convert mask tensor to numpy for plotting
        # mask_np = mask.squeeze().numpy()

        # # Create the plot
        # plt.figure(figsize=(8, 8))
        # plt.imshow(mask_np, cmap='gray')

        # # Plot end points in red
        # plt.scatter(end_x, end_y, color='red', s=50, marker='o')
        # plt.title(f'Mask with End Points - {id_}')
        # plt.axis('off')
        # plt.show()
        return mask, end_points

    def normalize_spline(self, spline: torch.Tensor) -> torch.Tensor:
        """
        Normalize the spline points to the range [0, 1].

        Args:
            spline (torch.Tensor): The spline points tensor of shape (num_pts, 3).

        Returns:
            torch.Tensor: Normalized spline points.
        """
        W, H = self.img_size
        spline[:, 0] /= (W - 1)
        spline[:, 1] /= (H - 1)
        return spline
    
    
    
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SplineEndPointDatasetHM(Dataset):
    """
    Dataset that returns:
      - mask:     FloatTensor shape [1, H, W]
      - heatmaps: FloatTensor shape [2, H, W] (one Gaussian peak per endpoint)
    """
    def __init__(self,
                 root: str,
                 normalize: bool = True,
                 img_size: tuple = (256, 256),
                 num_pts: int = 200,
                 sigma: float = 5.0):
        self.root       = root
        self.normalize  = normalize
        self.img_size   = img_size
        self.num_pts    = num_pts
        self.sigma      = sigma

        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids     = sorted([fn.split(".")[0] for fn in os.listdir(self.img_dir)])
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),        # → [1, H, W] in [0,1]
        ])

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def make_endpoint_heatmaps(ep: np.ndarray, H: int, W: int, sigma: float=5.0):
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        hms = np.zeros((2, H, W), dtype=np.float32)
        for i, (x, y) in enumerate(ep):
            hms[i] = np.exp(-((xs - x)**2 + (ys - y)**2) / (2*sigma*sigma))
        return hms

    def __getitem__(self, idx: int):
        id_ = self.ids[idx]
        # -- load & preprocess mask --
        mask = Image.open(os.path.join(self.img_dir, f"{id_}.png")).convert("L")
        mask = self.transform(mask)  # FloatTensor [1,H,W]

        # -- load spline & pick endpoints --
        data   = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = data["spline"]              # (num_pts, 2) in pixel coords
        ep_pts = spline[[0, -1], :].astype(np.float32)  # [[x1,y1],[x2,y2]]

        # -- normalize for other uses (optional) --
        if self.normalize:
            W, H = self.img_size
            spline_norm = spline.copy()
            spline_norm[:,0] /= (W - 1)
            spline_norm[:,1] /= (H - 1)
            ep_norm = spline_norm[[0, -1], :]
            # but for heatmaps we want pixel coords:
            ep_px = ep_norm * np.array([[W - 1, H - 1]])
        else:
            ep_px = ep_pts

        # -- build heatmaps in pixel space --
        H, W = self.img_size
        hms = self.make_endpoint_heatmaps(ep_px, H, W, self.sigma)
        heatmaps = torch.from_numpy(hms)  # FloatTensor [2,H,W]

        return mask, heatmaps
