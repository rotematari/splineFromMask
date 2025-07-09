from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
from typing import Tuple

class SplinePointDatasetNew(Dataset):
    """
    PyTorch Dataset for binary mask images and their B-spline control points.
    """
    def __init__(self,
                 root: str,
                 normalize: bool = True,
                 resize_to: Tuple[int,int] = None):
        self.root      = root
        self.normalize = normalize

        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids     = sorted(os.listdir(self.img_dir))

        # peek at first sample
        if self.ids:
            first = self.ids[0].split(".")[0]
            lbl   = np.load(os.path.join(self.lbl_dir, f"{first}.npz"))
            spline = lbl["spline"]           # (num_pts, 2)
            self.num_pts = spline.shape[0]

            img = Image.open(os.path.join(self.img_dir, f"{first}.png"))
            W, H = img.size
            img.close()
            # if resize_to is provided, override
            self.img_size = resize_to or (W, H)
        else:
            self.num_pts = 0
            self.img_size = resize_to or (256, 256)

        # build transform
        tfms = []
        tfms.append(transforms.ToTensor())  # ⇒ [0,1]
        self.transform = transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        id_  = self.ids[idx].split(".")[0]
        img  = Image.open(os.path.join(self.img_dir, f"{id_}.png")).convert("L")
        mask = self.transform(img)   # (1,H,W)
        img.close()

        lbl    = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = torch.from_numpy(lbl["spline"]).float()  # (num_pts,2)
        if self.normalize:
            spline = self.normalize_spline(spline)

        return mask, spline

    def normalize_spline(self, spline: torch.Tensor) -> torch.Tensor:
        """
        Normalize spline points (pix coords) to [0,1] by image dims.
        """
        W, H = self.img_size
        spline[:, 0] = spline[:, 0] / (W - 1)
        spline[:, 1] = spline[:, 1] / (H - 1)
        return spline

class SplinePointDataset(Dataset):
    """
    Custom PyTorch Dataset for loading mask images and their corresponding B-spline control points.
    """
    def __init__(self, root, normalize=True, resize_to=None):
        """
        Args:
            root (str): Root directory of the dataset, containing 'images' and 'labels' subfolders.
            resize_to (tuple or None): If provided, resize images to this size. If None, use original image size.
        """
        self.root = root
        self.normalize = normalize
        self.resize_to = resize_to
        
        # Setup directories and get file list
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids = sorted(os.listdir(self.img_dir))
        
        if len(self.ids) > 0:
            # Load first sample to get spline size and image size
            first_id = self.ids[0].split(".")[0]
            first_lbl = np.load(os.path.join(self.lbl_dir, f"{first_id}.npz"))
            spline = first_lbl["spline"]
            self.num_pts = spline.shape[0]
            
            # Get image size from first image
            first_img = Image.open(os.path.join(self.img_dir, f"{first_id}.png"))
            self.img_size = first_img.size  # (width, height)
            first_img.close()
        else:
            self.num_pts = 200  # fallback
            self.img_size = (256, 256)  # fallback
            
        # Set up transforms based on whether we're resizing
        if resize_to is not None:
            self.img_size = resize_to
            self.transform = transforms.Compose([
                transforms.Resize(resize_to),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.50]),  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
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
    def __init__(self, root, normalize=True, resize_to=None):
        """
        Args:
            root (str): Root directory of the dataset, containing 'images' and 'labels' subfolders.
            resize_to (tuple or None): If provided, resize images to this size. If None, use original image size.
        """
        self.root = root
        self.normalize = normalize
        self.resize_to = resize_to
        
        # Setup directories and get file list
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids = sorted(os.listdir(self.img_dir))
        
        if len(self.ids) > 0:
            # Load first sample to get spline size and image size
            first_id = self.ids[0].split(".")[0]
            first_lbl = np.load(os.path.join(self.lbl_dir, f"{first_id}.npz"))
            spline = first_lbl["spline"]
            self.num_pts = spline.shape[0]
            
            # Get image size from first image
            first_img = Image.open(os.path.join(self.img_dir, f"{first_id}.png"))
            self.img_size = first_img.size  # (width, height)
            first_img.close()
        else:
            self.num_pts = 200  # fallback
            self.img_size = (256, 256)  # fallback

        # Set up transforms based on whether we're resizing
        if resize_to is not None:
            self.img_size = resize_to
            self.transform = transforms.Compose([
                transforms.Resize(resize_to),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
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
    

class SplineEndPointDatasetHM(Dataset):
    """
    Dataset that returns:
      - mask:     FloatTensor shape [1, H, W]
      - heatmaps: FloatTensor shape [2, H, W] (one Gaussian peak per endpoint)
    """
    def __init__(self,
                 root: str,
                 normalize: bool = True,
                 resize_to: tuple = None,
                 sigma: float = 5.0):
        
        self.root       = root
        self.normalize  = normalize
        self.resize_to  = resize_to
        self.sigma      = sigma

        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids     = sorted([fn.split(".")[0] for fn in os.listdir(self.img_dir)])
        
        if len(self.ids) > 0:
            # Load first sample to get spline size and image size
            first_lbl = np.load(os.path.join(self.lbl_dir, f"{self.ids[0]}.npz"))
            spline = first_lbl["spline"]
            self.num_pts = spline.shape[0]
            
            # Get image size from first image
            first_img = Image.open(os.path.join(self.img_dir, f"{self.ids[0]}.png"))
            self.img_size = first_img.size  # (width, height)
            first_img.close()
        else:
            self.num_pts = 200  # fallback
            self.img_size = (256, 256)  # fallback
            
        # Set up transforms based on whether we're resizing
        if resize_to is not None:
            self.img_size = resize_to
            self.transform = transforms.Compose([
                transforms.Resize(resize_to),
                transforms.ToTensor(),        # → [1, H, W] in [0,1]
            ])
        else:
            self.transform = transforms.Compose([
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
    @staticmethod
    def make_endpoint_heatmaps_1d(ep: np.ndarray, H: int, W: int, sigma: float=5.0):
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Create single heatmap with both points
        hm = np.zeros((H, W), dtype=np.float32)
        
        for i, (x, y) in enumerate(ep):
            # Add gaussian for each endpoint to the same heatmap
            gaussian = np.exp(-((xs - x)**2 + (ys - y)**2) / (2*sigma*sigma))
            hm = np.maximum(hm, gaussian)  # Take maximum to avoid overlap interference
        
        # Return shape [1, H, W] to maintain consistency with the rest of the code
        return hm[np.newaxis, :]
    
    def __getitem__(self, idx: int):
        id_ = self.ids[idx]
        # -- load & preprocess mask --
        mask = Image.open(os.path.join(self.img_dir, f"{id_}.png")).convert("L")
        original_size = mask.size  # (width, height)
        mask = self.transform(mask)  # FloatTensor [1,H,W]

        # -- load spline & pick endpoints --
        data   = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = data["spline"]              # (num_pts, 2) in pixel coords
        ep_pts = spline[[0, -1], :].astype(np.float32)  # [[x1,y1],[x2,y2]]

        # Scale coordinates if image was resized
        if self.resize_to is not None:
            scale_x = self.img_size[0] / original_size[0]
            scale_y = self.img_size[1] / original_size[1]
            ep_pts[:, 0] *= scale_x  # x coordinates
            ep_pts[:, 1] *= scale_y  # y coordinates

        # -- normalize for other uses (optional) --
        if self.normalize:
            W, H = self.img_size[0], self.img_size[1]  # img_size is (width, height)
            spline_norm = ep_pts.copy()
            spline_norm[:,0] /= (W - 1)
            spline_norm[:,1] /= (H - 1)
            ep_norm = spline_norm
            # but for heatmaps we want pixel coords:
            ep_px = ep_norm * np.array([[W - 1, H - 1]])
        else:
            ep_px = ep_pts

        # -- build heatmaps in pixel space --
        H, W = self.img_size[1], self.img_size[0]  # Convert to (height, width)
        hms = self.make_endpoint_heatmaps_1d(ep_px, H, W, self.sigma)
        heatmaps = torch.from_numpy(hms).float()  # FloatTensor [1,H,W]

        return mask, heatmaps


class SplineHMDataset(Dataset):
    """
    Custom PyTorch Dataset for loading mask images and their corresponding B-spline points as heatmaps .
    """
    def __init__(self, root, normalize=True, resize_to=None, sigma=5.0):
        """
        Args:
            root (str): Root directory of the dataset, containing 'images' and 'labels' subfolders.
            resize_to (tuple or None): If provided, resize images to this size. If None, use original image size.
            sigma (float): Standard deviation for Gaussian heatmaps.
        """
        self.root = root
        self.normalize = normalize
        self.resize_to = resize_to
        self.sigma = sigma
        
        self.img_dir = os.path.join(root, "images")
        self.lbl_dir = os.path.join(root, "labels")
        self.ids = sorted(os.listdir(self.img_dir))
        
        if len(self.ids) > 0:
            # Load first sample to get spline size and image size
            first_id = self.ids[0].split(".")[0]
            first_lbl = np.load(os.path.join(self.lbl_dir, f"{first_id}.npz"))
            spline = first_lbl["spline"]
            self.num_pts = spline.shape[0]
            
            # Get image size from first image
            first_img = Image.open(os.path.join(self.img_dir, f"{first_id}.png"))
            self.img_size = first_img.size  # (width, height)
            first_img.close()
        else:
            self.num_pts = 200  # fallback
            self.img_size = (256, 256)  # fallback
            
        # Set up transforms based on whether we're resizing
        if resize_to is not None:
            self.img_size = resize_to
            self.transform = transforms.Compose([
                transforms.Resize(resize_to),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
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
        original_size = mask.size  # (width, height)
        mask = self.transform(mask)  # Shape: (1, H, W)
        
        lbl = np.load(os.path.join(self.lbl_dir, f"{id_}.npz"))
        spline = lbl["spline"]  # Shape: (num_pts, 2)
        
        # Scale spline coordinates if image was resized
        if self.resize_to is not None:
            scale_x = self.img_size[0] / original_size[0]
            scale_y = self.img_size[1] / original_size[1]
            spline = spline.copy()
            spline[:, 0] *= scale_x  # x coordinates
            spline[:, 1] *= scale_y  # y coordinates
        
        H, W = self.img_size[1], self.img_size[0]  # Note: img_size is (width, height)
        hms = self.spline_to_heatmaps_nd(spline, H, W, self.sigma)
        heatmaps = torch.from_numpy(hms)  # FloatTensor [num_pts,H,W]
        return mask, heatmaps
    
    @staticmethod
    def spline_to_heatmaps_1d(spline: np.ndarray, H: int, W: int, sigma: float=5.0):
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Create single heatmap with both points
        hm = np.zeros((H, W), dtype=np.float32)

        for i, (x, y) in enumerate(spline):
            # Add gaussian for each endpoint to the same heatmap
            gaussian = np.exp(-((xs - x)**2 + (ys - y)**2) / (2*sigma*sigma))
            hm = np.maximum(hm, gaussian)  # Take maximum to avoid overlap interference
        
        # Return shape [1, H, W] to maintain consistency with the rest of the code
        return hm[np.newaxis, :]
    @staticmethod
    def spline_to_heatmaps_nd(spline: np.ndarray, H: int, W: int, sigma: float=5.0):
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Create single heatmap with both points
        hm = np.zeros((spline.shape[0], H, W), dtype=np.float32)

        for i, (x, y) in enumerate(spline):
            # Add gaussian for each endpoint to the same heatmap
            gaussian = np.exp(-((xs - x)**2 + (ys - y)**2) / (2*sigma*sigma))
            hm[i] =  gaussian  # Store each endpoint's heatmap in the first dimension
        
        # Return shape [1, H, W] to maintain consistency with the rest of the code
        return hm


    def find_heatmap_center_weighted(heatmap_coords, heatmap_values):
        """
        Find the weighted centroid of heatmap coordinates.
        
        Args:
            heatmap_coords: np.array of shape (N, 2) with (y, x) coordinates
            heatmap_values: np.array of shape (N,) with corresponding heatmap values
        
        Returns:
            center: np.array of shape (2,) with (y, x) center coordinates
        """
        # Weighted average of coordinates
        total_weight = np.sum(heatmap_values)
        center_y = np.sum(heatmap_coords[:, 0] * heatmap_values) / total_weight
        center_x = np.sum(heatmap_coords[:, 1] * heatmap_values) / total_weight
        
        return np.array([center_y, center_x])
    
    
    def batch_centroids_torch(batch_H):
        """
        batch_H: torch tensor of shape (C, H, W)
        returns a np.array of (x_center, y_center) for each channel
        """
        C, H, W = batch_H.shape
        # make coordinate grids once
        ys = torch.arange(H, device=batch_H.device, dtype=batch_H.dtype).view(1, H, 1).expand(C, H, W)
        xs = torch.arange(W, device=batch_H.device, dtype=batch_H.dtype).view(1, 1, W).expand(C, H, W)
        totals = batch_H.sum(dim=(1,2))               # shape (C,)
        x_centers = (xs * batch_H).sum(dim=(1,2)) / totals
        y_centers = (ys * batch_H).sum(dim=(1,2)) / totals
        return np.array(list(zip(x_centers.tolist(), y_centers.tolist())))
    
if __name__ == "__main__":
    
    
    # plot the labels of SplineHMDataset

    dataset = SplineEndPointDatasetHM(root="src/spline_dataset/ds_256_10_100pts_4-10ctrl_k3_s1_dim2_Nm30",
                              sigma=5.0)

    import matplotlib.pyplot as plt

    # Get one sample
    mask, heatmaps = dataset[15]
    

    # Convert tensors to numpy for plotting
    mask_np = mask.squeeze().numpy()  # Remove channel dimension

    if heatmaps.shape[0] == 1:
        heatmap_np = heatmaps[0].numpy()
    
    else:
        heatmap_np = heatmaps.numpy()  # First endpoint heatmap
    # Create subplot with 1 row, 4 columns
    fig, axes = plt.subplots(1, 3)

    # Plot mask
    axes[0].imshow(mask_np, cmap='gray')
    axes[0].set_title('Mask')
    axes[0].axis('off')

    # if heatmaps.shape[0] == 1:
    # Plot first heatmap
    axes[1].imshow(heatmap_np, cmap='hot')
    axes[1].set_title('Heatmap 1 (First Endpoint)')
    axes[1].axis('off')
    # Plot mask with overlaid heatmaps
    # get heatmap coords


    # If you have multiple heatmaps, shape (C, H, W):


    axes[2].imshow(mask_np, cmap='gray', alpha=0.7)
    axes[2].imshow(heatmap_np, cmap='Reds', alpha=0.5)
    # Plot the center of the heatmap
    # axes[2].scatter(centers[:, 0], centers[:, 1], color='blue', s=100, marker='x', label='Heatmap Center')
    axes[2].legend()
    axes[2].set_title('Mask with Overlaid Heatmaps')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()