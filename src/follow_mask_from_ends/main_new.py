"""
Main training script for spline heatmap/path prediction model.
"""
from typing import List, Dict, Any, Tuple
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import spline_dataset.datasets as ds_module
from follow_mask_from_ends import models
from follow_mask_from_ends.train import ModelTrainer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def validate_config(config: Dict[str, Any]) -> None:
    """Ensure required config keys exist and dataset directory is valid."""
    required_keys = ['dataset_dir', 'results_dir', 'epochs', 'lr', 'batch_size', 'val_size', 'dataset']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    if not os.path.exists(config['dataset_dir']):
        raise FileNotFoundError(f"Dataset directory not found: {config['dataset_dir']}")


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Dynamically select and load dataset, split into train/val, and return loaders."""
    # Dynamically pick dataset class
    DatasetClass = getattr(ds_module, config['dataset'])
    dataset = DatasetClass(root=config['dataset_dir'],
                           normalize=config.get('normalize', True))
    print(f"Loaded dataset ({config['dataset']}) with {len(dataset)} samples")
    print(f"Image size: {dataset.img_size}, points per spline: {dataset.num_pts}")

    # Update config with dataset properties
    config['img_size'] = dataset.img_size[0]
    config['num_pts']  = dataset.num_pts

    # Split
    val_size   = int(config['val_size'] * len(dataset))
    train_size = len(dataset) - val_size
    generator  = torch.Generator().manual_seed(config.get('seed', 42))
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

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


def plot_training_curves(train_losses: List[float],
                         val_losses:   List[float],
                         results_dir:  str) -> None:
    """Plot and save training/validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses,   label='Val Loss',   alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def main(config: Dict[str, Any]) -> Tuple[nn.Module, List[float], List[float]]:
    """Main entry point: validate config, prepare data, init model, and train."""
    validate_config(config)
    set_seed(config.get('seed', 42))

    # Prepare directories
    os.makedirs(config['results_dir'], exist_ok=True)
    config['checkpoints_dir'] = os.path.join(config['results_dir'], 'checkpoints')
    config['plots_dir']       = os.path.join(config['results_dir'], 'plots')
    os.makedirs(config['checkpoints_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = create_data_loaders(config)

    # Model selection from config
    model_cls    = getattr(models, config.get('model', 'VisionTransformer'))
    model        = model_cls(config=config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized {config.get('model')} with {total_params:,} parameters")

    # Trainer
    trainer = ModelTrainer(model, config)
    try:
        train_losses, val_losses = trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        trainer._save_checkpoint(epoch='interrupt')
        raise

    # Plot curves
    plot_training_curves(train_losses, val_losses, config['plots_dir'])
    print("Training completed successfully!")
    return model, train_losses, val_losses


if __name__ == "__main__":
    # Configuration
    config = {
        'dataset_dir':    "src/spline_dataset/ds_256_1024_16pts_5-12ctrl_k5_s10_dim2_Nm1",
        'results_dir':    "src/follow_mask_from_ends/results",
        'dataset':        'SplinePointDatasetNew',       
        'model':          'MaskEncoder_2',  # or 'ViTDETRPathRegressor'
        'epochs':         400,
        'lr':             1e-4,
        'scheduler':      None,  
        'weight_decay':   1e-5,  
        'batch_size':     16,
        'val_size':       0.1,
        'seed':           42,
        'num_workers':    0,
        'criterion':      nn.SmoothL1Loss(beta=1),
        'normalize':      True,
        'teacher_forcing': False,  # Enable teacher forcing
        'initial_tf_rate':        1.0,  # Probability of applying teacher forcing
        'final_tf_rate':          0.0,  # Probability of applying teacher forcing
        'scheduler':      None,  # Optional: can be set to a learning rate scheduler
        'plot_interval': 1,  # Plot every 2 epochs
        'checkpoint_interval': 5,  # Save checkpoint every 5 epochs
        'mask_loss_weight': 1.0,  # Weight for mask loss
        ## Model-specific parameters 
        # For MaskEncoder 
        'mid_channels': 256,  # Mid channels for the conv block
        'encoder_dim': 128,  # Dimension for the encoder
    }
    main(config)
