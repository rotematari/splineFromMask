from typing import List, Dict, Any
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from edges_from_mask import models
import numpy as np
import matplotlib.pyplot as plt
def run_epoch(config: Dict[str, Any], train_loader: DataLoader, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module) -> None:
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        inputs, targets = batch  # Move inputs and targets to the device
        inputs = inputs.to(config['device'])
        targets = targets.to(config['device'])
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def val(config: Dict[str, Any], val_loader: DataLoader, model: nn.Module, criterion: nn.Module) -> None:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(config['device'])
            targets = targets.to(config['device'])
            outputs = model(inputs)
            loss = criterion(outputs, targets) 

            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train(model: nn.Module, config: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader) -> None:
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],weight_decay=config['weight_decay'])
    # criterion = nn.MSELoss()
    criterion = config.get('criterion', nn.MSELoss())  # Allow overriding the criterion
    # pos_weight = torch.tensor([200.0, 200.0], device=config['device'])
    # criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  
    model.train()

    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")
    train_losses =  []
    val_losses = []
    for epoch in range(config['epochs']):
        train_loss = run_epoch(config, train_loader, model, optimizer, criterion)
        val_loss = val(config, val_loader, model, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % 10 == 0:
            test(model, config, val_loader)

        print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    # Optionally save the model
    checkpoint_path = os.path.join(config['checkpoints_dir'], f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    return train_losses, val_losses
    
def test(model: nn.Module,
         config: Dict[str, Any],
         test_loader: DataLoader) -> float:
    
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    os.makedirs(config['results_dir'], exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # Move to device
            inputs  = inputs.to(config['device'])   # [B,1,H,W]
            targets = targets.to(config['device'])  # [B,2,H,W]

            outputs = model(inputs)                 # [B,2,H,W]

            batch_size = inputs.size(0)

            # Accumulate loss weighted by batch size
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            

            # Visualize up to wandb_num_images samples
            n_vis = min(batch_size, config.get('wandb_num_images', 2))
            for i in range(n_vis):
                mask_np = inputs[i, 0].cpu().numpy()    # [H,W]
                pred = outputs[i].cpu().numpy()      # [2,H,W]
                true = targets[i].cpu().numpy()      # [2,H,W]

                # unnormalize predictions to img_size
                pred = pred * config['image_size'][0]  # assuming img_size is (H, W)
                true = true * config['image_size'][0]

                # find endpoints by argmax in each channel
                # y1p, x1p = pred[1], pred[0]

                # y1t, x1t = true[1], true[0]

                y1p, x1p = pred[0,1], pred[0,0]
                y2p, x2p = pred[1,1], pred[1,0]
                y1t, x1t = true[0,1], true[0,0]
                y2t, x2t = true[1,1], true[1,0]
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(mask_np, cmap='gray')
                # ax.scatter([x1p], [y1p], c='red', marker='o',
                #            label='Predicted Tips', s=25)
                # ax.scatter([x1t], [y1t], c='blue', marker='x',
                #            label='Ground Truth Tips', s=25)
                ax.scatter([x1p, x2p], [y1p, y2p], c='red', marker='o',
                           label='Predicted Tips', s=25)
                ax.scatter([x1t, x2t], [y1t, y2t], c='blue', marker='x',
                           label='Ground Truth Tips', s=25)
                ax.axis('off')
                ax.legend(loc='upper right')
                plt.tight_layout()

                save_path = os.path.join(
                    config['results_dir'],
                    f"test_batch{batch_idx}_sample{i}.png"
                )
                plt.savefig(save_path)
                plt.close(fig)

    avg_loss = total_loss / total_samples
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

def test_hm(model: nn.Module,
         config: Dict[str, Any],
         test_loader: DataLoader) -> float:
    model.eval()
    criterion = nn.MSELoss()
    # pos_weight = torch.tensor([200.0, 200.0], device=config['device'])
    # criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total_loss = 0.0
    total_samples = 0

    os.makedirs(config['results_dir'], exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # Move to device
            inputs  = inputs.to(config['device'])   # [B,1,H,W]
            targets = targets.to(config['device'])  # [B,2,H,W]

            outputs = model(inputs)                 # [B,2,H,W]

            batch_size = inputs.size(0)

            # Accumulate loss weighted by batch size
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            outputs = torch.sigmoid(outputs) 

            # Visualize up to wandb_num_images samples
            n_vis = min(batch_size, config.get('wandb_num_images', 2))
            for i in range(n_vis):
                mask_np = inputs[i, 0].cpu().numpy()    # [H,W]
                pred_hm = outputs[i].cpu().numpy()      # [2,H,W]
                true_hm = targets[i].cpu().numpy()      # [2,H,W]

                # find endpoints by argmax in each channel
                y1p, x1p = np.unravel_index(pred_hm[0].argmax(), pred_hm[0].shape)
                y2p, x2p = np.unravel_index(pred_hm[1].argmax(), pred_hm[1].shape)
                y1t, x1t = np.unravel_index(true_hm[0].argmax(), true_hm[0].shape)
                y2t, x2t = np.unravel_index(true_hm[1].argmax(), true_hm[1].shape)

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(mask_np, cmap='gray')
                ax.scatter([x1p, x2p], [y1p, y2p], c='red', marker='o',
                           label='Predicted Tips', s=25)
                ax.scatter([x1t, x2t], [y1t, y2t], c='blue', marker='x',
                           label='Ground Truth Tips', s=25)
                ax.axis('off')
                ax.legend(loc='upper right')
                plt.tight_layout()

                save_path = os.path.join(
                    config['results_dir'],
                    f"test_batch{batch_idx}_sample{i}.png"
                )
                plt.savefig(save_path)
                plt.close(fig)

    avg_loss = total_loss / total_samples
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def main(config: Dict[str, Any]) -> None:

    # setup device 
    set_seed(config['seed'])  # Set random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    # Load dataset
    from spline_from_mask.datasets import SplineEndPointDataset, SplineEndPointDatasetHM

    dataset = SplineEndPointDataset(root=config['dataset_dir'],
                                    img_size=config['image_size'],
                                    num_pts=config['num_pts'],
                                    normalize=True)
    val_size = int(config['val_size'] * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model

    model = models.EncoderRegressionHead_2().to(device)

    train_losses, val_losses = train(model, config, train_loader, val_loader)
    
    test(model, config, val_loader)  # Using validation loader for testing
    # plot training and validation losses
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(config['results_dir'], 'loss_plot.png'))
    plt.show()
    

if __name__ == "__main__":
    # define paths
    DATASET_DIR_PATH = "src/spline_from_mask/dataset_256_32_50pts"
    assert os.path.exists(DATASET_DIR_PATH), f"Dataset path {DATASET_DIR_PATH} does not exist."
    RESULTS_DIR_PATH = "src/edges_from_mask/results"
    if not os.path.exists(RESULTS_DIR_PATH):
        os.makedirs(RESULTS_DIR_PATH)
    CHECK_POINTS_DIR_PATH = "src/spline_from_mask/checkpoints"
    if not os.path.exists(CHECK_POINTS_DIR_PATH):
        os.makedirs(CHECK_POINTS_DIR_PATH)
        
    # setup config
    config = {
        'dataset_dir': DATASET_DIR_PATH,
        'results_dir': RESULTS_DIR_PATH,
        'checkpoints_dir': CHECK_POINTS_DIR_PATH,
        'model': 'EncoderRegressionHead',  # or 'UNetSpline', 'EncoderRegressionHead', etc.
        'epochs': 1000,
        'lr': 1e-6,
        'criterion': nn.L1Loss(),  # or any other loss function
        'weight_decay': 1e-5,
        'wandb_num_images': 1,
        'batch_size': 32,  # Batch size for training
        'val_size': 0.1,  # 10% of the dataset for validation
        'image_size': (256, 256),  # Size to resize images to
        'num_pts': 50,  # Number of points defining the spline
        'seed': 42,  # Random seed for reproducibility
    }
    main(config)