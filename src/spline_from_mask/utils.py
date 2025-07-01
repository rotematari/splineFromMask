import os
import cv2
import numpy as np
import wandb
import torch

def batch_centroids_torch(batch_H: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    """
    Compute (x,y) centroid of each channel in a heatmap batch, robustly.
    
    Args:
        batch_H: torch.Tensor of shape (C, H, W), may contain negatives.
        eps: small constant to avoid division by zero.
    
    Returns:
        centroids: np.ndarray of shape (C, 2), each row = (x_center, y_center).
    """
    # 1) clamp negatives (so only "heat" contributes)
    Hpos = batch_H.clamp_min(0.0).to(torch.float32)
    
    # 1.5) zero out values below mean (optional thresholding)
    max_per = Hpos.amax(dim=(1,2), keepdim=True)   # shape (C,1,1)
    thresh  = max_per * 0.9
    Hpos = torch.where(Hpos >= thresh, Hpos, torch.zeros_like(Hpos))

    # 2) shapes and device
    C, H, W = Hpos.shape
    device = Hpos.device
    
    # 3) coordinate grids in float32
    ys = torch.arange(H, device=device, dtype=torch.float32) \
             .view(1, H, 1).expand(C, H, W)
    xs = torch.arange(W, device=device, dtype=torch.float32) \
             .view(1, 1, W).expand(C, H, W)
    
    # 4) total heat per channel (plus eps)
    totals = Hpos.sum(dim=(1, 2)) + eps
    
    # 5) weighted sums → centers
    x_centers = (xs * Hpos).sum(dim=(1, 2)) / totals
    y_centers = (ys * Hpos).sum(dim=(1, 2)) / totals
    
    # 6) stack and convert to NumPy
    centroids = torch.stack((x_centers, y_centers), dim=1)  # shape (C,2)
    return centroids.cpu().numpy()



def compute_difference_images(folder_path):
    files = os.listdir(folder_path)

    # Create mappings for res and _res images
    res_images = {f[4:]: f for f in files if f.startswith("res_") and f.endswith(".png")}
    _res_images = {f[5:]: f for f in files if f.startswith("_res_") and f.endswith(".png")}

    for key in res_images.keys():
        if key in _res_images:
            res_path = os.path.join(folder_path, res_images[key])
            _res_path = os.path.join(folder_path, _res_images[key])
            out_path = os.path.join(folder_path, "__res_" + key)

            img1 = cv2.imread(res_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(_res_path, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                print(f"Skipping unreadable pair: {res_images[key]} and {_res_images[key]}")
                continue
            if img1.shape != img2.shape:
                print(f"Shape mismatch in pair: {res_images[key]} and {_res_images[key]}")
                continue

            diff = cv2.absdiff(img1, img2)
            diff = diff/np.max(np.max(diff)) * 255
            cv2.imwrite(out_path + ".png", diff)
            print(f"Saved difference: {os.path.basename(out_path)}")

if __name__ == "__main__":
    pass
