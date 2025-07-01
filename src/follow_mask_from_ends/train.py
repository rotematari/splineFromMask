import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
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
            
            heatmaps = self.model(inputs)
        batch_size = inputs.size(0)
        if batch_size > 4:
            batch_size = 4
        # Convert to numpy

        
        n_channels = heatmaps[0].shape[0]
        
        # Create 2 plots side by side
        fig, axes = plt.subplots(batch_size, n_channels+1, figsize=(16, 8))
        for k in range(batch_size):
            mask_np = inputs[k,0].cpu().numpy()
            heatmap_np = heatmaps[k].cpu().numpy()
            target_np = targets[k].cpu().numpy()
            for i in range(n_channels+1):
                if i == 0:
                    # Left plot: Mask with ground truth heatmaps overlaid
                    axes[k,0].imshow(mask_np, cmap='gray', alpha=0.7)
                    # Overlay all ground truth heatmaps
                    for j in range(n_channels):
                        axes[k,0].imshow(target_np[j], cmap='Blues', alpha=0.4)
                    axes[k,0].set_title(f'Ground Truth Heatmaps on Mask - Epoch {epoch}')
                    axes[k,0].axis('off')
                else:
                    # Overlay each channel's heatmap
                    axes[k,i].imshow(mask_np, cmap='gray', alpha=0.7)
                    axes[k,i].imshow(heatmap_np[i-1], cmap='Reds', alpha=0.4)
                    axes[k,i].set_title(f'Channel {i} Heatmap - Epoch {epoch}')
                    axes[k,i].axis('off')


        plt.tight_layout()
        
        # Save the figure
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, f"heatmaps_epoch_{epoch}.png"), dpi=150, bbox_inches='tight')
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
        
    