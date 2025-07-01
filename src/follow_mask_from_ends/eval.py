import torch.nn as nn
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
import numpy as np

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
        
