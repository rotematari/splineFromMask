from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
from einops import rearrange
import segmentation_models_pytorch as smp
from typing import Dict


class MaskEncoder(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        cfg = config or {}
        encoder_dim = cfg.get('encoder_dim', 256)
        self.backbone = smp.DeepLabV3Plus(
            encoder_name=cfg.get('encoder_name', 'resnet18'),
            encoder_weights=cfg.get('encoder_weights', None),
            in_channels=1,
            classes=encoder_dim
        )

        self.keypoint_head = KeypointHead(
            in_channels=encoder_dim,
            n_points=cfg.get('num_pts', 10),  # number of keypoints
            H=cfg.get('img_size', 256),  # height of input mask
            W=cfg.get('img_size', 256)   # width of input mask
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        feats = self.backbone(x)         # [B, C, H, W]
        # [B, C, H, W] -> [B, 1, n_points, 2]
        feats = self.keypoint_head(feats)  # [B, 1, n_points, 2]
        # reshape to [B, n_points, 2] for further processing
        coords ,hm = feats
        # coords = self.regressor(feats)  # [B, 2n] - flattened coords
        # coords = coords.view(coords.size(0), -1, 2)  # [
        
        return coords.view(coords.size(0), -1, 2)

class KeypointHead(nn.Module):
    def __init__(self, in_channels, n_points, H, W):
        super().__init__()
        self.H = H
        self.W = W

        # predict one heatmap per point
        self.heatmap = nn.Conv2d(in_channels, n_points, kernel_size=1)
        # precompute coordinate grids
        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        self.register_buffer('grid_x', grid_x[None,None])  # [1,1,H,W]
        self.register_buffer('grid_y', grid_y[None,None])
        
    def forward(self, feat):  # feat: [B, C, H, W]
        # feat = self.block(feat)   # [B, mid, H, W]
        h = self.heatmap(feat)    # [B, n, H, W]
        p = F.softmax(h.view(h.shape[0], h.shape[1], -1), dim=-1)
        # p: [B, n, H*W] - softmax over spatial dimensions
        p = p.view_as(h)          # [B, n, H, W]
        
        # expected x, y
        x = (p * self.grid_x).sum(dim=[2,3])   # [B, n]
        y = (p * self.grid_y).sum(dim=[2,3])   # [B, n]
        coords = torch.stack((x,y), dim=-1)    # [B, n, 2]
        return coords.unsqueeze(1) , p             # [B, 1, n, 2], [B, n, H, W] - heatmaps
    


class MaskEncoder_2(nn.Module):
    def __init__(self, config: Dict = None):
        super().__init__()
        cfg = config or {}
        encoder_dim = cfg.get('encoder_dim', 256)
        self.backbone = smp.DeepLabV3Plus(
            encoder_name=cfg.get('encoder_name', 'resnet18'),
            encoder_weights=cfg.get('encoder_weights', None),
            in_channels=1,
            classes=encoder_dim
        )

        self.keypoint_head = KeypointHead(
            in_channels=encoder_dim,
            n_points=cfg.get('num_pts', 10),  # number of keypoints
            H=cfg.get('img_size', 256),  # height of input mask
            W=cfg.get('img_size', 256)   # width of input mask
        )

        # 3) voting head: heatmaps + offsets
        self.vote_head = VoteHead(in_channels=encoder_dim,
                                  n_points=cfg.get('num_pts', 10),
                                  H=cfg.get('img_size', 256),
                                  W=cfg.get('img_size', 256)
                              ) # [B, n, 2], [B, n, H, W], [B, n, H, W, 2]

    def forward(self, x: torch.Tensor):
        # x: [B,1,H,W]
        feats = self.backbone(x)     # [B, C, H, W]

        # -- global regression branch --
        coords_reg,_ = self.keypoint_head(feats)
        B = coords_reg.size(0)
        coords_reg = coords_reg.view(B, -1, 2)  # [B, n, 2]

        # -- voting branch --
        coords_vote, hm, offsets = self.vote_head(feats)
        # coords_vote: [B, n, 2], hm: [B, n, H, W], offsets: [B, n, H, W, 2]

        return coords_reg, coords_vote, hm, offsets


class VoteHead(nn.Module):
    def __init__(self, in_channels, n_points, H, W):
        super().__init__()
        self.n = n_points
        self.H = H
        self.W = W

        # one heatmap per node
        self.heatmap = nn.Conv2d(in_channels, n_points, kernel_size=1)
        # one 2-vector offset per node
        self.offset  = nn.Conv2d(in_channels, n_points*2,   kernel_size=1)

        # pre-make normalized grids for soft-argmax
        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x[None,None])  # [1,1,H,W]
        self.register_buffer('grid_y', grid_y[None,None])

    def forward(self, feat):
        B = feat.size(0)

        # 1) heatmaps → softmax over H×W
        h = self.heatmap(feat)         # [B, n, H, W]
        p = F.softmax(h.flatten(2), dim=-1).view_as(h)  # [B, n, H, W]

        # 2) soft-argmax for vote coords
        x = (p * self.grid_x).sum(dim=[2,3])  # [B, n]
        y = (p * self.grid_y).sum(dim=[2,3])  # [B, n]
        coords = torch.stack([x,y], dim=-1)   # [B, n, 2]

        # 3) offsets
        off = self.offset(feat)              # [B, 2n, H, W]
        off = off.view(B, self.n, 2, self.H, self.W)       # [B, n, 2, H, W]
        off = off.permute(0,1,3,4,2).contiguous()           # [B, n, H, W, 2]

        return coords, p, off




if __name__ == "__main__":

    
    model = SplineTracer(config={
        'img_size': 256,
        'patch_size': 16,
        'in_chans': 1,
        'num_classes': 50,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'K': 50  # Number of points to predict
    })
    
    input_tensor = torch.randn(2, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image
    path_seq = torch.randn(2, 1, 2)  # Batch size of 1, 10 points, each with (x,y) coords
    
    output = model(input_tensor, path_seq)
    print("Output shape:", output.shape)  # Should be [2, 2*K]

    # print model size
    print("Model size (parameters):", sum(p.numel() for p in model.parameters() if p.requires_grad))

    iters= 10
    import time
    start = time.time()
    for i in range(iters):
        output = model(input_tensor, path_seq)
        new_point = path_seq[:, -1, :] + output  # simulate next point prediction
        path_seq = torch.cat((path_seq, new_point.unsqueeze(1)), dim=1)
    
    print("Output shape after iterations:", path_seq.shape)  # Should be [2, 2*K]
    end = time.time()
    print(f"Average inference time over {iters} iterations: {(end - start) / iters:.6f} seconds")