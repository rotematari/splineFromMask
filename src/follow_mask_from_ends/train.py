import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.optim import lr_scheduler

def mask_adherence_loss(preds, masks):
    """
    preds: Tensor of shape [B, 2], coordinates in normalized [0,1] image space
    masks: Tensor of shape [B, 1, H, W], binary mask (0 background, 1 foreground)
    """
    B, _, H, W = masks.shape
    # reshape preds for grid_sample: [B, 1, 1, 2], coords in [-1,1]
    grid = preds.view(B, 1, 1, 2) * 2 - 1
    # sample mask values at those points: out shape [B, 1, 1, 1]
    sampled = F.grid_sample(masks, grid, mode='bilinear', align_corners=True)
    sampled = sampled.view(B)
    # penalize (1 - mask_value)^2 so perfect adherence (mask=1) gives zero
    return ((1 - sampled) ** 2).mean()


class ModelTrainer:
    """Handles model training, autoregressive evaluation, and visualizing path predictions."""

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model.to(config['device'])
        self.config = config
        self.device = config['device']

        # Setup optimizer, criterion, (optional) scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-3)
        )
        self.coord_criterion = config.get('criterion', nn.MSELoss())
        self.hm_criterion    = nn.MSELoss()
        self.hm_weight       = config.get('heatmap_weight', 1.)
        self.scheduler = config.get('scheduler', None)
        self.mask_loss_weight = config.get('mask_loss_weight', 1.0)
        self.vote_radius = config.get('vote_radius', 50.0)  # radius for voting loss
        # Teacher forcing rate
        self.teacher_forcing   = config.get('teacher_forcing', False)
        self.tf_initial_rate  = config.get('initial_tf_rate', 1.0)
        self.tf_final_rate    = config.get('final_tf_rate',   0.0)
        self.tf_rate          = self.tf_initial_rate  # will be updated each epoch

        # Training history
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []

        # Ensure output dirs exist
        os.makedirs(config['checkpoints_dir'], exist_ok=True)
        os.makedirs(config.get('plots_dir', 'plots'), exist_ok=True)
    
    @staticmethod
    def _make_gaussian_heatmaps(
        coords: Tensor, H: int, W: int, sigma: float = 0.02
    ) -> Tensor:
        """
        coords: [B, n, 2] in [0,1]
        returns: heatmaps [B, n, H, W], each a Gaussian bump around (x,y)
        """
        device = coords.device
        B, n, _ = coords.shape

        # 1) build normalized grids
        xs = torch.linspace(0.0, 1.0, W, device=device)
        ys = torch.linspace(0.0, 1.0, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]
        # shape for broadcasting: [1, 1, H, W]
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)

        # 2) extract target coords → [B, n, 1, 1]
        x_t = coords[..., 0].view(B, n, 1, 1)
        y_t = coords[..., 1].view(B, n, 1, 1)

        # 3) squared distance & Gaussian
        dist2 = (grid_x - x_t)**2 + (grid_y - y_t)**2
        return torch.exp(-dist2 / (2 * sigma**2))  # [B, n, H, W]
    @staticmethod
    def build_gt_H_U(mask: torch.Tensor, 
                 nodes: torch.Tensor, 
                 r: float):
        """
        mask:  [H, W] binary (0/1) tensor
        nodes: [M, 2] float tensor of (x, y) coords for each node
        r:     cutoff radius in pixels for heatmap support

        Returns:
        H_gt: [N, M]   heatmap scores in [0,1]
        U_gt: [N, M, 2] unit-offset vectors
        pix:  [N, 2]  pixel coords (x, y) of the mask==1 points
        """
        # 1) collect pixel coords where mask==1
        coords = torch.where(mask > 0)             # each is [N]
        pix = torch.stack(coords, dim=1).float() # [N, 2]

        # 2) broadcast to compute distances to each node j
        #    pix_exp: [N, 1, 2], nodes_exp: [1, M, 2]
        pix_exp   = pix.unsqueeze(1)               # [N, 1, 2]
        nodes_exp = nodes.unsqueeze(0)             # [1, M, 2]

        # 3) Euclidean distance [N, M]
        dist = torch.norm(pix_exp - nodes_exp, dim=2)

        # 4) build H_gt by linear falloff, zero beyond radius
        H_gt = torch.clamp(1.0 - dist / r, min=0.0) # [N, M]

        # 5) build U_gt = (y_j - p_i) / dist  → [N, M, 2]
        offset = nodes_exp - pix_exp              # [N, M, 2]
        # avoid division by zero:
        dist_eps = dist.unsqueeze(-1).clamp(min=1e-6)  # [N, M, 1]
        U_gt = offset / dist_eps                   # [N, M, 2]

        # 6) zero out offsets where pixel is too far (dist > r)
        outside = (dist > r).unsqueeze(-1)         # [N, M, 1]
        U_gt = U_gt.masked_fill(outside, 0.0)

        return H_gt, U_gt, pix
    def voting_loss(H_pred: torch.Tensor,
                H_gt:   torch.Tensor,
                U_pred: torch.Tensor,
                U_gt:   torch.Tensor) -> torch.Tensor:
        """
        Computes the voting loss:
        L = (1/N) * (sum_{i,j} (H_pred[i,j]-H_gt[i,j])^2
                    + sum_{i,j} ||U_pred[i,j]-U_gt[i,j]||^2 )
        Args:
            H_pred: [N, M] heatmap predictions
            H_gt:   [N, M] heatmap ground-truth
            U_pred: [N, M, 2] offset predictions
            U_gt:   [N, M, 2] offset ground-truth
        Returns:
            Scalar loss tensor.
        """
        N = H_pred.shape[0]

        # Option A: exact formula via sums
        heatmap_se = (H_pred - H_gt).pow(2)             # [N, M]
        offset_se  = (U_pred - U_gt).pow(2).sum(dim=2)  # [N, M]
        loss = (heatmap_se.sum() + offset_se.sum()) / N
        return loss

        # Option B: using MSELoss with sum-reduction
        # loss_h = F.mse_loss(H_pred, H_gt, reduction='sum')
        # loss_u = F.mse_loss(U_pred, U_gt, reduction='sum')
        # return (loss_h + loss_u) / N
    def _calc_loss(self, outputs: Tuple[Tensor,Tensor], targets: Tensor) -> Tensor:
        """
        outputs = (coords_pred, heatmap_pred)
        coords_pred: [B,n,2]
        targets:      [B,n,2]        (normalized x,y in [0,1])
        """
        coords_pred = outputs
        coords_pred = coords_pred.squeeze(1)  # → [B, n, 2]
        coords_true = targets                  # [B, n, 2]

        # ---- 1) coordinate loss (with extra weight on ends) ----
        # here’s your original scheme:
        ends_loss = (
            self.coord_criterion(coords_pred[:, :2],  coords_true[:, :2]) +
            self.coord_criterion(coords_pred[:, -2:], coords_true[:, -2:])
        )
        mid_loss = self.coord_criterion(
            coords_pred[:, 2:-2], coords_true[:, 2:-2]
        )
        coord_loss = mid_loss + 3.0 * ends_loss


        total_loss = coord_loss
        return total_loss
    def _calc_loss2(self,
                outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                targets: torch.Tensor,
                masks:   torch.Tensor) -> torch.Tensor:
        """
        outputs = (coords_reg, coords_vote, hm, offsets)
        coords_reg: [B, n, 2]    (normalized x,y ∈ [0,1])
        coords_vote: [B, n, 2]   (normalized too, from soft-argmax)
        hm:         [B, n, H, W] heatmap logits→softmax
        offsets:    [B, n, H, W, 2] unit-offset vectors
        targets:    [B, n, 2]      normalized ground-truth nodes
        masks:      [B, H, W]      binary DLO masks
        """
        coords_reg, coords_vote, hm_pred, off_pred = outputs
        B, n, H, W = hm_pred.shape

        # 1) coordinate regression loss (with extra weight on ends)
        ends_loss = (
            self.coord_criterion(coords_reg[:, :2],  targets[:, :2]) +
            self.coord_criterion(coords_reg[:, -2:], targets[:, -2:])
        )
        mid_loss = self.coord_criterion(coords_reg[:, 2:-2], targets[:, 2:-2])
        coord_loss = mid_loss + 3.0 * ends_loss

        # 2) voting loss over the batch
        vote_loss = 0.0
        for b in range(B):
            mask_b = masks[b]            # [H, W]
            nodes_norm = targets[b]      # [n, 2] in [0,1]
            # convert to pixel coords
            nodes_pix = nodes_norm * torch.tensor([W, H],
                                device=nodes_norm.device)

            # build ground-truth heatmaps & offsets
            H_gt, U_gt, pix = self.build_gt_H_U(mask_b, nodes_pix, self.vote_radius)
            # H_gt: [N, n], U_gt: [N, n, 2], pix: [N, 2]

            # gather predicted heatmaps at each foreground pixel
            # hm_pred[b]: [n, H, W] → reorder → [H, W, n] → mask→ [N, n]
            H_pred_b = hm_pred[b].permute(1,2,0)[mask_b>0]       # [N, n]

            # gather predicted offsets
            # off_pred[b]: [n, H, W, 2] → [H, W, n, 2] → mask→ [N, n, 2]
            U_pred_b = off_pred[b].permute(2,3,0,1)[mask_b>0]    # [N, n, 2]

            vote_loss += self.voting_loss(H_pred_b, H_gt, U_pred_b, U_gt)

        vote_loss = vote_loss / B

        # 3) total
        total = coord_loss + self.vote_weight * vote_loss
        return total
    def run_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            loss = self._calc_loss2(preds, targets, inputs)
            # compute loss on mask
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()
            print(f"→ Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run autoregressive evaluation and return loss and average inference time."""
        self.model.eval()
        total_loss = 0.0
        total_time = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
    
                # autoregressive inference
                start = time.time()

                preds = self.model(inputs)
                total_time += time.time() - start
                # true deltas
                loss = self._calc_loss(preds, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_time = total_time / len(val_loader)
        return avg_loss, avg_time

    def train(self,
              train_loader: DataLoader,
              val_loader:   DataLoader
             ) -> Tuple[List[float], List[float]]:
        n_epochs = self.config['epochs']
        plot_interval = self.config.get('plot_interval', 10)
        ckpt_interval = self.config.get('checkpoint_interval', 10)

        print(f"Starting training for {n_epochs} epochs…")
        for epoch in range(1, n_epochs + 1):
            # linear decay of teacher-forcing rate
            if self.teacher_forcing and n_epochs > 1:
                frac         = (epoch - 1) / (n_epochs - 1)
                self.tf_rate = (self.tf_initial_rate * (1 - frac)
                                + self.tf_final_rate * frac)
                print(f"Epoch {epoch}/{n_epochs} — TF rate: {self.tf_rate:.3f}")
            
            train_loss = self.run_epoch(train_loader)
            val_loss, avg_time = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch}/{n_epochs} — "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Inference Time: {avg_time:.3f}s")

            # plot predictions periodically
            if epoch % plot_interval == 0:
                self._plot_predictions(val_loader, epoch, n_samples=min(self.config.get('n_samples', 16),self.config['batch_size']))

            # checkpoint periodically
            if epoch % ckpt_interval == 0:
                self._save_checkpoint(epoch)
            
            self._plot_training_curves(self.train_losses, self.val_losses,
                                      self.config['plots_dir'])

        return self.train_losses, self.val_losses

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(
            self.config['checkpoints_dir'],
            f"model_epoch_{epoch}.pth"
        )
        torch.save(self.model.state_dict(), path)
        print(f"→ Saved checkpoint: {path}")

    def _plot_predictions(self, val_loader: DataLoader, epoch: int, n_samples: int = 2) -> None:
        """Draw the binary mask with ground-truth (green) & predicted (red) paths."""
        self.model.eval()

        samples = []
        for inputs, targets in val_loader:
            for inputs, targets in zip(inputs, targets):
                samples.append((inputs, targets))
            if len(samples) >= n_samples:
                break
        
        # Calculate grid dimensions
        n_rows = (n_samples + 3) // 4  # Ceiling division
        n_columns = min(4, n_samples)
        
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(4 * n_columns, 4 * n_rows))
        
        # Handle different subplot arrangements
        if n_rows == 1 and n_columns == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        with torch.no_grad():
            for i, (imgs, trues) in enumerate(samples):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                imgs = imgs.to(self.device)

                path = self.model(imgs.unsqueeze(0))
                path = path.squeeze(0)
                # move to CPU/numpy
                mask = imgs.cpu().numpy()[0] 
                true_pts = (trues * self.config['img_size']).cpu().numpy().astype(int)
                pred_pts = (path * self.config['img_size']).cpu().numpy().astype(int)

                ax.imshow(mask, cmap='gray')
                ax.plot(true_pts[:, 0], true_pts[:, 1], '-o', markersize=3, label='GT')
                ax.plot(pred_pts[:, 0], pred_pts[:, 1], '-x', markersize=3, label='Pred')

                ax.set_title(f"Epoch {epoch} - Sample {i+1}")
                ax.axis('off')
                ax.legend(loc='upper right')
            
            # Hide unused subplots
            for j in range(len(samples), len(axes)):
                axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.config.get('plots_dir', 'plots'),
                                 f"pred_epoch_{epoch}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"→ Saved prediction plot: {save_path}")
    def _plot_training_curves(self, train_losses: List[float],
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