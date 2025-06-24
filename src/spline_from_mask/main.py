import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhocon.config_parser import glob
from torch.utils.data import DataLoader , random_split
from typing import Dict
# You must define `BsplineMaskGenerator` from earlier
from spline_from_mask.datasets import SplinePointDataset
from spline_from_mask.bspline_dataset import BsplineDataset
from spline_from_mask.bspline_image_label_gen import BsplineMaskGenerator
from spline_from_mask.models import UNetSmall
from spline_from_mask import unets
import numpy as np
from tqdm import tqdm
import wandb as wb

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SplineMaskLoss(nn.Module):
    """
    Loss for mask→spline tasks combining:
      1) MSE on the pixel‐coordinates of the spline points
      2) A mask‐alignment penalty that drives all predicted points onto the binary mask.

    Inputs:
      coords_pred: (B, N, 2), normalized [0,1] predicted spline points
      coords_gt:   (B, N, 2), normalized [0,1] ground‐truth spline points
      mask:        (B, 1, H, W), binary {0,1} mask where spline lives
    Returns:
      scalar loss = mse_loss + λ * mask_alignment_loss
    """
    def __init__(self, lambda_mask: float = 1.0):
        super().__init__()
        self.lambda_mask = lambda_mask
        self.mse = nn.MSELoss()

    def forward(self,
                coords_pred: torch.Tensor,
                coords_gt:   torch.Tensor,
                mask:        torch.Tensor) -> torch.Tensor:
        # shapes
        B, N, _ = coords_pred.shape
        _, _, H, W = mask.shape

        # 1) MSE on pixel‐space coords
        #    map normalized [0,1] → [0, W-1] (or H-1)
        pix_pred = coords_pred * (W - 1)  # (B, N, 2)
        pix_gt   = coords_gt   * (W - 1)  # (B, N, 2)
        loss_mse = self.mse(coords_pred, coords_gt)

        # 2) Mask‐alignment via grid_sample
        #    grid_sample expects coords in [-1,1]
        grid = coords_pred * 2.0 - 1.0         # (B, N, 2)
        grid = grid.view(B, N, 1, 2)           # (B, H_out=N, W_out=1, 2)
        sampled = F.grid_sample(
            mask,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )                                      # (B, 1, N, 1)
        mask_vals = sampled.view(B, N)         # (B, N), values in [0,1]

        # average fraction of points on‐mask → 1 only if all points land on mask
        loss_mask = 1.0 - mask_vals.mean(dim=1)  # (B,)
        loss_mask = loss_mask.mean()             # scalar

        return loss_mse + self.lambda_mask * loss_mask
    
def get_quantized_ids(
    pts: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """
    Convert B×S×2 normalized coords → B×S integer token IDs.

    Args:
        pts        Tensor of shape (B, S, 2), values in [0,1].
        grid_size  G, so V = G*G bins.

    Returns:
        ids        Tensor of shape (B, S), each in [0..V-1].
    """
    B, S, _ = pts.shape
    # 1) Quantize into 0..G-1 per axis
    bins = (pts * grid_size).long().clamp(0, grid_size - 1)  # (B,S,2)
    # 2) Flatten to single token id: x*G + y
    ids = bins[..., 0] * grid_size + bins[..., 1]            # (B,S)
    return ids

def decode_quantized_ids(
    ids: torch.Tensor,
    grid_size: int,
    image_size: int
) -> torch.Tensor:
    """
    Convert quantized token IDs back to 2D coordinates in pixel space.

    Args:
        ids         Tensor of shape (B, S), with values in [0 .. G*G-1]
        grid_size   G, the number of bins per axis
        image_size  N, the side length of the image in pixels (e.g. 256)

    Returns:
        coords      Tensor of shape (B, S, 2), where each (x, y) is in [0 .. N)
    """
    # Compute bin indices along each axis
    x_bins = ids // grid_size      # (B, S)
    y_bins = ids % grid_size       # (B, S)

    # Compute size of each bin in pixels
    bin_size = image_size / grid_size

    # Map bin index to pixel coordinate (using bin center)
    x_coords = x_bins.float() * bin_size + bin_size / 2
    y_coords = y_bins.float() * bin_size + bin_size / 2

    # Stack into (B, S, 2)
    coords = torch.stack([x_coords, y_coords], dim=-1)

    return coords
def train_model(model, train_loader, val_loader,
                epochs=5, lr=1e-4, weight_decay=1e-5,
                wandb_run=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    # criterion = nn.MSELoss()
    # criterion = SplineMaskLoss(lambda_mask=1.0)
    # criterion = F.cross_entropy
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    print("Training on:", next(model.parameters()).device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
            # linearly decay from full teacher forcing to none
        
        for inputs, targets in tqdm(train_loader):
            # print(i)
            inputs, targets = inputs.float(), targets.float()
            inputs, targets = inputs.to(device), targets.to(device)
            quant_gt = get_quantized_ids(targets, model.G)      # (B, S)
            outputs = model(inputs,
                            target_ids=quant_gt)   # (B,S,V)
            quant_gt = get_quantized_ids(targets, model.G)  # (B, S)
            loss = criterion(outputs.permute(0, 2, 1), quant_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.float(), targets.float()
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                quant_gt = get_quantized_ids(targets, model.G)  # (B, S)
                loss = criterion(outputs.permute(0, 2, 1), quant_gt)
                # loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        if wandb_run:
            fig,ax = plt.subplots()
            # outputs_np = outputs.permute(0, 2, 1).cpu().numpy()
            # targets_np = targets.permute(0, 2, 1).cpu().numpy()
            # Decode outputs from quantized IDs to coordinates
            outputs_decoded = decode_quantized_ids(outputs.argmax(dim=-1), model.G, 256)  # (B, S, 2)
            outputs_np = outputs_decoded.permute(0, 2, 1).cpu().numpy()  # (B, S, 2)
            targets_np = targets.permute(0, 2, 1).cpu().numpy()  # (B, S, 2)
            
            ax.plot(outputs_np[0, 0], outputs_np[0, 1], color='red', label='Predicted')
            ax.plot(targets_np[0, 0], targets_np[0, 1], color='green', label='Target')
            ax.legend()
            ax.set_title(f"Epoch {epoch+1} - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
            wb.log({
                "train_loss": avg_train,
                "val_loss": avg_val,
                "pred": wb.Image(fig, caption="Predictions vs Targets"),
            })
            plt.close()
        if epoch > 20 and epoch % 5 != 0:
            checkpoint_path = os.path.join(config['checkpoints_dir'], f'{config["model"]}_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

    # Plot loss curves
    plt.close()
    fig,ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True)
    # ax.tight_layout()
    if wandb_run:
        wb.Image(fig, caption="Training and Validation Loss")
        wb.log({"loss_plot": fig})
    plt.savefig("loss_plot.png")
    plt.close()
    # plt.show()

    return model


def masked_mse_loss(pred, target):
    """
    Computes MSE loss only on pixels where the target > 0.
    Emphasizes learning curve structure, ignores background.
    """
    mask = (target > 0).float()
    squared_error = (pred - target) ** 2
    masked_loss = (squared_error * mask).sum() / (mask.sum() + 1e-8)
    return masked_loss


def train_model_(model, dataloader, epochs=5, lr=1e-4):
    """
    Train the model on a dataset using MSE loss.

    Args:
        model: PyTorch model (e.g. UNetSmall)
        dataloader: DataLoader yielding (input, target) pairs
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        Trained model
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    # criterion = masked_mse_loss

    for epoch in range(epochs):
        print('starting epoch', epoch)
        total_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            print(i)
            inputs, targets = inputs.float(), targets.float()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model




def train_main(wandb_run=None,
               config: Dict = None):

    dataset = SplinePointDataset(root=config['dataset_dir'],
                                    img_size=config['image_size'],
                                    num_pts=config['num_pts'],
                                    normalize=False)
    val_size = int(config['val_size'] * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    # model = UNetSmall()
    # model = unets.UNetSpline(in_channels=1, num_points=config['num_pts'], beta=3.0)
    # model = unets.EncoderRegressionHead(
    #     num_spline_points=config['num_pts'],
    #     in_channels=1,
    #     pretrained=True,
    #     out_dim=2,
    #     mlp_hidden=1048,
    #     dropout=0.2
    # )
    # model = unets.Mask2ControlPoints(
    #     grid_size=64,
    #     hidden_dim=256,
    #     num_steps=config['num_pts'],
    # )
    model = unets.Mask2ControlPointsTransformer(
    grid_size=64,
    num_steps=config['num_pts'],
    d_model=256, nhead=8, num_decoder_layers=4
    ).to(device)





    # model = unets.EncoderTransformerDecoder(
    #     num_spline_points=config['num_pts'],
    #     in_channels=1,
    #     pretrained=True,
    #     out_dim=2,
        
    # )
    model = model.to(device)
    trained_model = train_model(model, train_loader, val_loader,
                                epochs=config["epochs"], lr=config['learning_rate'],
                                weight_decay=config['weight_decay'],
                                wandb_run=wandb_run)

    

    trained_model.eval()

    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = trained_model(x)
        file_name = os.path.join(config['results_dir'], f'res_{i}.png')
        
        # y_pred = y_pred.permute(0, 2, 1).cpu().numpy()  # (B, num_pts, 2) → (B, 2, num_pts)
        y = y.permute(0, 2, 1).cpu().numpy()  # (B, num_pts, 2) → (B, 2, num_pts)
        y_pred = decode_quantized_ids(y_pred.argmax(dim=-1), grid_size=model.G, image_size=config['image_size'][0])  # (B, num_pts, 2)
        y_pred = y_pred.cpu().numpy()  # Convert to numpy for plotting
        # cv2.imwrite(file_name, 255*np.asarray(y_pred[0, 0].cpu()))
        plt.plot(y_pred[0, 0], y_pred[0, 1], color='red', label='Predicted')
        plt.plot(y[0, 0], y[0, 1], color='green', label='Target')
        plt.legend()
        plt.savefig(file_name)
        plt.close()
    checkpoint_path = os.path.join(config['checkpoints_dir'], f'{config["model"]}_{config["epochs"]}.pth')
    torch.save(trained_model.state_dict(), checkpoint_path)


def testing(wandb_run=None,
            config: Dict = None):
    model = unets.UNet(1,1)
    model = model.to(device)
    model.eval()

    all_files = sorted(glob(os.path.join(config['dataset_dir'], "mask_*.png")))
    train_files = all_files[:int(0.8 * len(all_files))]
    val_files = all_files[int(0.2 * len(all_files)):]
    val_set = BsplineDataset(path=config['dataset_dir'], from_disk=True)
    val_set.mask_files = val_files
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model.eval()

    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(x)
        file_name = os.path.join(config['results_dir'], f'res_{i}.png')
        cv2.imwrite(file_name, 255*np.asarray(y_pred[0, 0].cpu()))

import os
import cv2
import numpy as np

def rescale_grayscale_image(img):
    # Only rescale non-zero pixels
    mask = img > 1
    if np.any(mask):
        max_val = img[mask].max()
        if max_val > 1:
            # Rescale non-zero values to range [1, 255]
            img[mask] = (img[mask] / max_val) * 254 + 1  # ensure values stay in [1, 255]
            img = np.clip(img, 1, 255).astype(np.uint8)
    return img

def process_folder(folder_path,config=None,wandb_run=None):
    count = 0
    
    for filename in os.listdir(folder_path):
        if filename.startswith('res_') and filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue
            file_path = os.path.join(folder_path, "_" + filename)
            rescaled_img = rescale_grayscale_image(img)
            cv2.imwrite(file_path, rescaled_img)
            if wandb_run and count < config['wandb_num_images']:
                count += 1
                wb.log({f"rescaled_{filename}": wb.Image(rescaled_img, caption=f"Rescaled {filename}")})
            # print(f"Rescaled and saved: {filename}")



if __name__ == '__main__':
    
    is_wandb =  True
    if is_wandb:
        import wandb as wb
        wb.login()
    else:
        print("WandB is disabled, running without logging.")

    DATASET_DIR_PATH = "src/spline_from_mask/dataset_256_1000_50pts"
    assert os.path.exists(DATASET_DIR_PATH), f"Dataset path {DATASET_DIR_PATH} does not exist."
    RESULTS_DIR_PATH = "src/spline_from_mask/results"
    if not os.path.exists(RESULTS_DIR_PATH):
        os.makedirs(RESULTS_DIR_PATH)
    CHECK_POINTS_DIR_PATH = "src/spline_from_mask/checkpoints"
    if not os.path.exists(CHECK_POINTS_DIR_PATH):
        os.makedirs(CHECK_POINTS_DIR_PATH)
        
    
    config = {
        'dataset_dir': DATASET_DIR_PATH,
        'results_dir': RESULTS_DIR_PATH,
        'checkpoints_dir': CHECK_POINTS_DIR_PATH,
        'model': 'EncoderRegressionHead',  # or 'UNetSpline', 'EncoderRegressionHead', etc.
        'epochs': 100,
        'learning_rate': 1e-5,
        'weight_decay': 1e-6,
        'wandb_num_images': 2,
        'batch_size': 32,  # Batch size for training
        'val_size': 0.1,  # 10% of the dataset for validation
        'image_size': (256, 256),  # Size to resize images to
        'num_pts': 50,  # Number of points defining the spline
    }
    import time
    wandb_run = wb.init(
        entity="dlo",
        project="train",
        config=config,
        name=f"{time.strftime('%Y-%m-%d_%H-%M')}",
        ) if is_wandb else None
    

    train_main(wandb_run=wandb_run, config=config)
    # process_folder(config['results_dir'],wandb_run=wandb_run, config=config)

    if wandb_run:
        wandb_run.finish()
