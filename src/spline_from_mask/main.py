import cv2
import torch
import torch.nn as nn
from pyhocon.config_parser import glob
from torch.utils.data import DataLoader
from typing import Dict
# You must define `BsplineMaskGenerator` from earlier

from spline_from_mask.bspline_dataset import BsplineDataset
from spline_from_mask.bspline_image_label_gen import BsplineMaskGenerator
from spline_from_mask.models import UNetSmall
from spline_from_mask import unets
import numpy as np
from tqdm import tqdm
import wandb as wb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader,
                epochs=5, lr=1e-4,
                wandb_run=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    print("Training on:", next(model.parameters()).device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            # print(i)
            inputs, targets = inputs.float(), targets.float()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        if wandb_run:
            wb.log({
                "train_loss": avg_train,
                "val_loss": avg_val,
            })
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}")

    # Plot loss curves
    import matplotlib.pyplot as plt
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

    all_files = sorted(glob(os.path.join(config['dataset_dir'], "mask_*.png")))
    train_files = all_files[:int(0.8 * len(all_files))]
    val_files = all_files[int(0.2 * len(all_files)):]



    train_set = BsplineDataset(path=config['dataset_dir'], from_disk=True)
    train_set.mask_files = train_files

    val_set = BsplineDataset(path=config['dataset_dir'], from_disk=True)
    val_set.mask_files = val_files

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)

    # model = UNetSmall()
    model = unets.UNet(1,1)
    model = model.to(device)
    trained_model = train_model(model, train_loader, val_loader,
                                epochs=config["epochs"], lr=config['learning_rate'],
                                wandb_run=wandb_run)
    

    import matplotlib.pyplot as plt

    trained_model.eval()

    for i, (x, y) in enumerate(val_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred = trained_model(x)
        file_name = os.path.join(config['results_dir'], f'res_{i}.png')
        cv2.imwrite(file_name, 255*np.asarray(y_pred[0, 0].cpu()))
    checkpoint_path = os.path.join(config['checkpoints_dir'], f'model_epoch_{config["epochs"]}.pth')
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
    
    DATASET_DIR_PATH = "src/spline_from_mask/bspline_dataset"
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
        'epochs': 50,
        'learning_rate': 1e-4,
        'wandb_num_images': 2,
    }
    
    wandb_run = wb.init(
        entity="dlo",  
        project="spline_from_mask",
        config=config,
        name="spline_from_mask_training",
        ) if is_wandb else None
    

    train_main(wandb_run=wandb_run, config=config)
    process_folder(config['results_dir'],wandb_run=wandb_run, config=config)

    if wandb_run:
        wandb_run.finish()
