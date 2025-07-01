"""
Main training script for spline heatmap prediction model.
"""
from typing import List, Dict, Any, Tuple, Optional
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from key_point_from_mask import models
import numpy as np
import matplotlib.pyplot as plt

from key_point_from_mask.eval import ModelEvaluator
from key_point_from_mask.train import ModelTrainer


def setup_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories."""
    for dir_key in ['results_dir', 'checkpoints_dir']:
        if dir_key in config:
            os.makedirs(config[dir_key], exist_ok=True)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    from spline_dataset.datasets import SplineEndPointDatasetHM

    dataset = SplineEndPointDatasetHM(root=config['dataset_dir'], normalize=True)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image size: {dataset.img_size}")
    print(f"Number of points per spline: {dataset.num_pts}")
    
    # Update config with dataset properties
    config['num_pts'] = dataset[0][1].shape[0]  # Number of points per spline
    config['img_size'] = dataset.img_size
    # Set num_classes based on number of spline endpoints (typically 2 for start/end points)
    

    print(f"Config updated - num_pts: {config['num_pts']}, img_size: {config['img_size']}, num_classes: {config['num_classes']}")
    
    val_size = int(config['val_size'] * len(dataset))
    train_size = len(dataset) - val_size
    
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config.get('num_workers', 0)
        
    )
    
    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    return train_loader, val_loader


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                        results_dir: str) -> None:
    """Plot and save training curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_keys = ['dataset_dir', 'epochs', 'lr', 'batch_size']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if not os.path.exists(config['dataset_dir']):
        raise FileNotFoundError(f"Dataset directory not found: {config['dataset_dir']}")


def main(config: Dict[str, Any]) -> None:
    """Main training function."""
    # Validate configuration
    validate_config(config)
    
    # Setup
    set_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    setup_directories(config)
    
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Initialize model
    # model = models.EncoderHeatmapHead(config).to(device)
    model = models.EncoderEdgePointsHM(config=config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training
    trainer = ModelTrainer(model, config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, config['results_dir'])
    
    # # Testing
    # evaluator = ModelEvaluator(model, config)
    # test_loss = evaluator.test(val_loader)
    
    print("Training completed successfully!")
    return model, train_losses, val_losses

if __name__ == "__main__":
    # define paths
    DATASET_DIR_PATH = "src/spline_dataset/ds_256_1000_100pts_5-15ctrl_k5_s10_dim2_Nm20"
    assert os.path.exists(DATASET_DIR_PATH), f"Dataset path {DATASET_DIR_PATH} does not exist."
    RESULTS_DIR_PATH = "src/key_point_from_mask/results"
    if not os.path.exists(RESULTS_DIR_PATH):
        os.makedirs(RESULTS_DIR_PATH)
    CHECK_POINTS_DIR_PATH = "src/key_point_from_mask/checkpoints"
    if not os.path.exists(CHECK_POINTS_DIR_PATH):
        os.makedirs(CHECK_POINTS_DIR_PATH)
        
    # setup config
    config = {
        # Dataset and paths
        'dataset_dir': DATASET_DIR_PATH,
        'results_dir': RESULTS_DIR_PATH,
        'checkpoints_dir': CHECK_POINTS_DIR_PATH,
        
        # Model configuration
        'model': 'splineHM',  # or 'UNetSpline', 'EncoderRegressionHead', etc.
        'model_output_type': 'heatmaps',  # 'coordinates' or 'heatmaps'
        'num_classes': 1,  # Will be updated from dataset (typically 2 for start/end points)
        'num_pts': 1,     # Will be updated from dataset
        'img_size': (256, 256),  # Will be updated from dataset
        
        # Training parameters
        'epochs': 10,
        'lr': 1e-4,
        'min_lr': 1e-5,  # Minimum learning rate for the scheduler
        'criterion': nn.MSELoss(),  # or any other loss function
        'weight_decay': 1e-3,
        'batch_size': 8,  # Batch size for training
        'val_size': 0.1,  # 10% of the dataset for validation
        
        # Visualization and logging
        'wandb_num_images': 1,
        'plot_interval': 1,  # Plot heatmaps every N epochs
        'test_vis_batches': 2,  # Number of test batches to visualize
        
        # System configuration
        'seed': 42,  # Random seed for reproducibility
        'num_workers': 0,  # DataLoader workers
    }
    main(config)