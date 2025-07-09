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

from edges_from_mask import models
import numpy as np
import matplotlib.pyplot as plt


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config['device']
        
        # Setup optimizer and criterion
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-3)
        )
        self.criterion = config.get('criterion', nn.MSELoss())
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def run_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation and return loss and average inference time."""
        self.model.eval()
        total_loss = 0.0
        total_time = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                start_time = time.time()
                outputs = self.model(inputs)
                total_time += (time.time() - start_time)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_loader)
        avg_time = total_time / len(val_loader)
        
        return avg_loss, avg_time

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[float], List[float]]:
        """Main training loop."""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            train_loss = self.run_epoch(train_loader)
            val_loss, avg_time = self.validate(val_loader)
            
            # Visualize progress
            if epoch % self.config.get('plot_interval', 10) == 0:
                # Check if model outputs coordinates or heatmaps
                model_output_type = self.config.get('model_output_type', 'heatmaps')
                if model_output_type == 'coordinates':
                    self._plot_coordinates(val_loader, epoch)
                else:
                    self._plot_heatmaps(val_loader, epoch)
                
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{self.config['epochs']}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Inference Time: {avg_time:.4f}s")
        
        # Save final model
        self._save_checkpoint(epoch)
        
        return self.train_losses, self.val_losses
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config['checkpoints_dir'], 
            f"model_epoch_{epoch}.pth"
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    def _plot_heatmaps(self, val_loader: DataLoader, epoch: int) -> None:
        """Plot heatmap predictions for visualization."""
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets = next(iter(val_loader))
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            heatmaps = self.model(inputs)[0]
        
        # Convert to numpy
        mask_np = inputs[0, 0].cpu().numpy()
        heatmap_np = heatmaps.cpu().numpy()
        target_np = targets[0].cpu().numpy()
        
        # Create visualization with 2 rows: predicted and true overlaid on mask
        n_channels = heatmap_np.shape[0]  # Use all available channels
        
        # If more than 5 channels, arrange in 4 rows instead of side-by-side comparison
        if n_channels > 5:
            cols = min(n_channels, 6)  # Max 6 columns to keep readable
            rows = 4
            fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 8))
            if cols == 1:
                axes = axes.reshape(4, 1)
            else:
                axes = axes.reshape(4, cols)
            
            # Calculate how many channels to show per row
            channels_per_row = min(cols, n_channels)
            
            for i in range(channels_per_row):
                # Row 0: Ground truth heatmap on mask for channels 0 to channels_per_row-1
                if i < n_channels:
                    axes[0, i].imshow(mask_np, cmap='gray', alpha=0.7)
                    axes[0, i].imshow(target_np[i], cmap='Blues', alpha=0.6)
                    axes[0, i].set_title(f'GT Ch{i+1}', fontsize=9)
                    axes[0, i].axis('off')
                
                # Row 1: Predicted heatmap on mask for channels 0 to channels_per_row-1
                if i < n_channels:
                    axes[1, i].imshow(mask_np, cmap='gray', alpha=0.7)
                    axes[1, i].imshow(heatmap_np[i], cmap='Reds', alpha=0.6)
                    axes[1, i].set_title(f'Pred Ch{i+1}', fontsize=9)
                    axes[1, i].axis('off')
                
                # Row 2: Ground truth heatmap on mask for additional channels (if available)
                next_ch_idx = i + channels_per_row
                if next_ch_idx < n_channels:
                    axes[2, i].imshow(mask_np, cmap='gray', alpha=0.7)
                    axes[2, i].imshow(target_np[next_ch_idx], cmap='Blues', alpha=0.6)
                    axes[2, i].set_title(f'GT Ch{next_ch_idx+1}', fontsize=9)
                    axes[2, i].axis('off')
                else:
                    axes[2, i].axis('off')
                
                # Row 3: Predicted heatmap on mask for additional channels (if available)
                if next_ch_idx < n_channels:
                    axes[3, i].imshow(mask_np, cmap='gray', alpha=0.7)
                    axes[3, i].imshow(heatmap_np[next_ch_idx], cmap='Reds', alpha=0.6)
                    axes[3, i].set_title(f'Pred Ch{next_ch_idx+1}', fontsize=9)
                    axes[3, i].axis('off')
                else:
                    axes[3, i].axis('off')
            
            # Hide unused subplots if channels_per_row < cols
            for i in range(channels_per_row, cols):
                for row in range(4):
                    axes[row, i].axis('off')
        else:
            # Original layout for 5 or fewer channels
            fig, axes = plt.subplots(2, n_channels, figsize=(4 * n_channels, 8))
            if n_channels == 1:
                axes = axes.reshape(2, 1)
            elif n_channels > 1:
                axes = axes.reshape(2, n_channels)
            
            for i in range(n_channels):
                # Row 0: Mask with predicted heatmap overlay
                axes[0, i].imshow(mask_np, cmap='gray', alpha=0.7)
                axes[0, i].imshow(heatmap_np[i], cmap='Reds', alpha=0.6)
                axes[0, i].set_title(f'Predicted on Mask - Channel {i+1}')
                axes[0, i].axis('off')
                
                # Row 1: Mask with true heatmap overlay
                axes[1, i].imshow(mask_np, cmap='gray', alpha=0.7)
                axes[1, i].imshow(target_np[i], cmap='Blues', alpha=0.6)
                axes[1, i].set_title(f'Ground Truth on Mask - Channel {i+1}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Add a final plot showing peak detection comparison
        from spline_from_mask.utils import batch_centroids_torch
        
        # Create a separate figure for peak comparison
        fig2, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(mask_np, cmap='gray')
        
        # Find predicted peaks using batch_centroids_torch
        heatmaps_tensor = torch.from_numpy(heatmap_np).unsqueeze(0)  # Add batch dimension
        pred_centroids = batch_centroids_torch(heatmaps_tensor[0])  # Shape: [num_channels, 2]
        
        pred_points = []
        true_points = []
        
        for ch in range(heatmap_np.shape[0]):
            # Predicted peak from centroid calculation
            x_pred, y_pred = pred_centroids[ch]
            pred_points.append((x_pred, y_pred))
            
            # True peak using argmax (as ground truth should be precise)
            y_true, x_true = np.unravel_index(target_np[ch].argmax(), target_np[ch].shape)
            true_points.append((x_true, y_true))
        
        # Plot points
        if pred_points:
            pred_x, pred_y = zip(*pred_points)
            ax.scatter(pred_x, pred_y, c='red', marker='o', s=50, label='Predicted Centroids')
            ax.plot(pred_x, pred_y, 'r-', alpha=0.7)
        
        if true_points:
            true_x, true_y = zip(*true_points)
            ax.scatter(true_x, true_y, c='blue', marker='x', s=50, label='Ground Truth')
            ax.plot(true_x, true_y, 'b-', alpha=0.7)
        
        ax.set_title(f'Peak Detection Comparison - Epoch {epoch}')
        ax.axis('off')
        ax.legend()
        
        # Save the peak comparison figure
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"peak_comparison_epoch_{epoch}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # Save the original heatmap figure
        plt.figure(fig.number)  # Switch back to original figure
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"heatmaps_epoch_{epoch}.png"))
        plt.close(fig)

    def _plot_coordinates(self, val_loader: DataLoader, epoch: int) -> None:
        """Plot coordinate predictions for visualization (for transformer models)."""
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets = next(iter(val_loader))
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Get model predictions (coordinates)
            pred_coords = self.model(inputs)  # Shape: (B, num_pts, 2)
        
        # Convert to numpy and denormalize coordinates to pixel space
        mask_np = inputs[0, 0].cpu().numpy()  # First image in batch
        pred_coords_np = pred_coords[0].cpu().numpy()  # Shape: (num_pts, 2)
        target_coords_np = targets[0].cpu().numpy()  # Shape: (num_pts, 2)
        
        # Get image dimensions
        H, W = mask_np.shape
        
        # Denormalize coordinates from [0,1] to pixel space
        pred_coords_px = pred_coords_np.copy()
        pred_coords_px[:, 0] *= (W - 1)  # x coordinates
        pred_coords_px[:, 1] *= (H - 1)  # y coordinates
        
        target_coords_px = target_coords_np.copy()
        target_coords_px[:, 0] *= (W - 1)  # x coordinates
        target_coords_px[:, 1] *= (H - 1)  # y coordinates
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Ground Truth
        axes[0].imshow(mask_np, cmap='gray')
        axes[0].scatter(target_coords_px[:, 0], target_coords_px[:, 1], 
                       c='blue', s=50, marker='o', label='GT Points', alpha=0.8)
        axes[0].plot(target_coords_px[:, 0], target_coords_px[:, 1], 
                    'b-', alpha=0.6, linewidth=2, label='GT Spline')
        axes[0].set_title(f'Ground Truth - Epoch {epoch}')
        axes[0].axis('off')
        axes[0].legend()
        
        # Right plot: Predictions
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].scatter(pred_coords_px[:, 0], pred_coords_px[:, 1], 
                       c='red', s=50, marker='o', label='Pred Points', alpha=0.8)
        axes[1].plot(pred_coords_px[:, 0], pred_coords_px[:, 1], 
                    'r-', alpha=0.6, linewidth=2, label='Pred Spline')
        axes[1].set_title(f'Predictions - Epoch {epoch}')
        axes[1].axis('off')
        axes[1].legend()
        
        plt.tight_layout()
        
        # Save the figure
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"coordinates_epoch_{epoch}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Also create a comparison plot with both GT and predictions on the same image
        fig2, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask_np, cmap='gray')
        
        # Plot ground truth
        ax.scatter(target_coords_px[:, 0], target_coords_px[:, 1], 
                  c='blue', s=60, marker='o', label='Ground Truth', alpha=0.8, edgecolors='white')
        ax.plot(target_coords_px[:, 0], target_coords_px[:, 1], 
               'b-', alpha=0.7, linewidth=3, label='GT Spline')
        
        # Plot predictions
        ax.scatter(pred_coords_px[:, 0], pred_coords_px[:, 1], 
                  c='red', s=60, marker='s', label='Predictions', alpha=0.8, edgecolors='white')
        ax.plot(pred_coords_px[:, 0], pred_coords_px[:, 1], 
               'r--', alpha=0.7, linewidth=3, label='Pred Spline')
        
        ax.set_title(f'Coordinate Prediction Comparison - Epoch {epoch}')
        ax.axis('off')
        ax.legend(loc='upper right')
        
        # Save the comparison figure
        plt.savefig(os.path.join(results_dir, f"coordinate_comparison_epoch_{epoch}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
    
class ModelEvaluator:
    """Handles model evaluation and testing."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config['device']
        
    def test(self, test_loader: DataLoader) -> float:
        """Test the model and return average loss."""
        self.model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                batch_size = inputs.size(0)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Visualize some test samples
                if batch_idx < self.config.get('test_vis_batches', 2):
                    # Check if model outputs coordinates or heatmaps
                    model_output_type = self.config.get('model_output_type', 'heatmaps')
                    if model_output_type == 'coordinates':
                        self._visualize_test_batch_coordinates(inputs, outputs, targets, batch_idx)
                    else:
                        self._visualize_test_batch(inputs, outputs, targets, batch_idx)
        
        avg_loss = total_loss / total_samples
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss
    
    def _visualize_test_batch(self, inputs: torch.Tensor, outputs: torch.Tensor, 
                             targets: torch.Tensor, batch_idx: int) -> None:
        """Visualize test batch results."""
        from spline_from_mask.utils import batch_centroids_torch
        
        batch_size = inputs.size(0)
        n_vis = min(batch_size, self.config.get('wandb_num_images', 2))
        
        for i in range(n_vis):
            mask_np = inputs[i, 0].cpu().numpy()
            pred_hm = outputs[i].cpu().numpy()
            true_hm = targets[i].cpu().numpy()
            
            # Find peak points using batch_centroids_torch for predictions
            pred_points = []
            true_points = []
            
            # Use batch_centroids_torch for predicted peaks (more accurate than argmax)
            pred_hm_tensor = outputs[i:i+1]  # Keep batch dimension for single sample
            pred_centroids = batch_centroids_torch(pred_hm_tensor[0])  # Shape: [1, num_channels, 2]
            
            for ch in range(pred_hm.shape[0]):
                # Predicted peak from centroid calculation
                x_pred, y_pred = pred_centroids[ch]
                pred_points.append((x_pred, y_pred))
                
                # True peak using argmax (as ground truth should be precise)
                y_true, x_true = np.unravel_index(true_hm[ch].argmax(), true_hm[ch].shape)
                true_points.append((x_true, y_true))
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(mask_np, cmap='gray')
            
            # Plot points
            if pred_points:
                pred_x, pred_y = zip(*pred_points)
                ax.scatter(pred_x, pred_y, c='red', marker='o', s=50, label='Predicted')
                ax.plot(pred_x, pred_y, 'r-', alpha=0.7)
            
            if true_points:
                true_x, true_y = zip(*true_points)
                ax.scatter(true_x, true_y, c='blue', marker='x', s=50, label='Ground Truth')
                ax.plot(true_x, true_y, 'b-', alpha=0.7)
            
            ax.set_title(f'Test Sample {batch_idx}_{i}')
            ax.axis('off')
            ax.legend()
            
            # Save
            save_path = os.path.join(
                self.config['results_dir'],
                f"test_batch{batch_idx}_sample{i}.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    def _visualize_test_batch_coordinates(self, inputs: torch.Tensor, outputs: torch.Tensor, 
                                         targets: torch.Tensor, batch_idx: int) -> None:
        """Visualize test batch results for coordinate predictions."""
        batch_size = inputs.size(0)
        n_vis = min(batch_size, self.config.get('wandb_num_images', 2))
        
        for i in range(n_vis):
            mask_np = inputs[i, 0].cpu().numpy()
            pred_coords = outputs[i].cpu().numpy()  # Shape: (num_pts, 2)
            true_coords = targets[i].cpu().numpy()  # Shape: (num_pts, 2)
            
            # Get image dimensions
            H, W = mask_np.shape
            
            # Denormalize coordinates from [0,1] to pixel space
            pred_coords_px = pred_coords.copy()
            pred_coords_px[:, 0] *= (W - 1)  # x coordinates
            pred_coords_px[:, 1] *= (H - 1)  # y coordinates
            
            true_coords_px = true_coords.copy()
            true_coords_px[:, 0] *= (W - 1)  # x coordinates
            true_coords_px[:, 1] *= (H - 1)  # y coordinates
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(mask_np, cmap='gray')
            
            # Plot ground truth
            ax.scatter(true_coords_px[:, 0], true_coords_px[:, 1], 
                      c='blue', s=60, marker='o', label='Ground Truth', alpha=0.8, edgecolors='white')
            ax.plot(true_coords_px[:, 0], true_coords_px[:, 1], 
                   'b-', alpha=0.7, linewidth=3, label='GT Spline')
            
            # Plot predictions
            ax.scatter(pred_coords_px[:, 0], pred_coords_px[:, 1], 
                      c='red', s=60, marker='s', label='Predictions', alpha=0.8, edgecolors='white')
            ax.plot(pred_coords_px[:, 0], pred_coords_px[:, 1], 
                   'r--', alpha=0.7, linewidth=3, label='Pred Spline')
            
            ax.set_title(f'Test Sample {batch_idx}_{i} - Coordinate Predictions')
            ax.axis('off')
            ax.legend()
            
            # Save
            save_path = os.path.join(
                self.config['results_dir'],
                f"test_coordinates_batch{batch_idx}_sample{i}.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        

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
    from spline_dataset.datasets import SplineHMDataset,SplinePointDataset
    
    dataset = SplinePointDataset(root=config['dataset_dir'], normalize=True)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image size: {dataset.img_size}")
    print(f"Number of points per spline: {dataset.num_pts}")
    
    # Update config with dataset properties
    config['num_pts'] = dataset.num_pts
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
    model = models.DeepLabTransformer(num_pts=config['num_pts']).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training
    trainer = ModelTrainer(model, config)
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, config['results_dir'])
    
    # Testing
    evaluator = ModelEvaluator(model, config)
    test_loss = evaluator.test(val_loader)
    
    print("Training completed successfully!")
    return model, train_losses, val_losses, test_loss
    

if __name__ == "__main__":
    # define paths
    DATASET_DIR_PATH = "src/spline_dataset/ds_256x256_1000splines_10pts_4-10ctrl_k3_s0p4_dim2"
    assert os.path.exists(DATASET_DIR_PATH), f"Dataset path {DATASET_DIR_PATH} does not exist."
    RESULTS_DIR_PATH = "src/edges_from_mask/results"
    if not os.path.exists(RESULTS_DIR_PATH):
        os.makedirs(RESULTS_DIR_PATH)
    CHECK_POINTS_DIR_PATH = "src/edges_from_mask/checkpoints"
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
        'model_output_type': 'coordinates',  # 'coordinates' or 'heatmaps'
        'num_classes': 2,  # Will be updated from dataset (typically 2 for start/end points)
        'num_pts': 1,     # Will be updated from dataset
        'img_size': (256, 256),  # Will be updated from dataset
        
        # Training parameters
        'epochs': 1000,
        'lr': 1e-4,
        'min_lr': 1e-5,  # Minimum learning rate for the scheduler
        'criterion': nn.MSELoss(),  # or any other loss function
        'weight_decay': 1e-3,
        'batch_size': 8,  # Batch size for training
        'val_size': 0.1,  # 10% of the dataset for validation
        
        # Visualization and logging
        'wandb_num_images': 1,
        'plot_interval': 10,  # Plot heatmaps every N epochs
        'test_vis_batches': 2,  # Number of test batches to visualize
        
        # System configuration
        'seed': 42,  # Random seed for reproducibility
        'num_workers': 0,  # DataLoader workers
    }
    main(config)