import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EncoderRegressionHead(nn.Module):
    """
    Input:  B×1×256×256 mask
    Output: B×4  – predicted [x1,y1,x2,y2] in normalized [0,1]
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=1), # 128×128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64×64
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1), # 32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3, stride=2, padding=1), # 16×16
            nn.ReLU(inplace=True),
        )
        self.encoder = models
        self.pool = nn.AdaptiveAvgPool2d((1,1))  # → B×256×1×1
        self.regressor = nn.Sequential(
            nn.Flatten(),               # → B×256
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64,   4),         # → B×4
            nn.Sigmoid()                # ensure output ∈ [0,1]
        )

    def forward(self, x):
        # x: B×1×256×256
        x = self.encoder(x)
        x = self.pool(x)
        x = self.regressor(x)
        # x: B×4

        return x.view(-1, 2, 2)  # Ensure output shape is B×2×2

class EncoderRegressionHead_2(nn.Module):
    """
    Encoder backbone with a regression head for predicting spline control points.

    Args:
        num_spline_points (int): Number of points in the spline.
        in_channels (int): Number of input image channels (e.g., 1 for binary masks).
        pretrained (bool): Whether to load pretrained weights for the backbone.
        out_dim (int): Dimension per spline point (e.g., 2 for 2D, 3 for 3D).
        mlp_hidden (int): Hidden size of the MLP.
        dropout (float): Dropout probability between MLP layers.
    """

    def __init__(
        self,
        pretrained: bool = False,
        out_dim: int = 2,
        mlp_hidden: int = 2048,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.out_dim = out_dim

        # --- Backbone: ResNet34 encoder ---
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # Adapt first conv if input channels != 3
        # for param in resnet.parameters():
        #     param.requires_grad = True
        # for name, p in resnet.named_parameters():
        #     # keep everything frozen except layer4
        #     if "layer4" in name:
        #         p.requires_grad = True
        # if in_channels != 3:
        #     resnet.conv1 = nn.Conv2d(
        #         in_channels,
        #         resnet.conv1.out_channels,
        #         kernel_size=resnet.conv1.kernel_size,
        #         stride=resnet.conv1.stride,
        #         padding=resnet.conv1.padding,
        #         bias=False,
        #     )
        #     nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        #     resnet.conv1.weight.requires_grad = True

        # Remove avgpool and fully-connected layers
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        # freeze the encoder layers

        self.feature_dim = 2048  # final channel count of ResNet18

        # # --- Regression head ---
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        # self.dropout = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(self.feature_dim, mlp_hidden)
        
        # self.fc2 = nn.Linear(mlp_hidden, mlp_hidden//2)
        # self.fc3 = nn.Linear(mlp_hidden//2, 4)

        self.mlp =  nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, 2),
            
        )
        self.mlp2 =  nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, 2),
            
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Regression output of shape (B, num_spline_points, out_dim).
        """
        # x = x.repeat(1, 3, 1, 1)  # Convert single-channel input to 3 channels (e.g., for ResNet) 
        # Extract features
        feat = self.encoder(x)                # (B, feature_dim, H', W')
        pooled = self.avgpool(feat).view(feat.size(0), -1)  # (B, feature_dim)

        # MLP regression
        p1 = self.mlp(pooled)                  # (B, mlp_hidden//2)
        p2 = self.mlp2(pooled)                  # (B, mlp_hidden//2)
        x = torch.cat([p1, p2], dim=1)         # (B, mlp_hidden)
        # x = p1
        # Reshape to (B, num_spline_points, out_dim)
        return x.view(-1, 2, 2)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_channels=32):
        super().__init__()
        b = base_channels
        self.down1 = DownSample(in_channels, b)       # 1→32
        self.down2 = DownSample(b, b*2)               # 32→64
        self.down3 = DownSample(b*2, b*4)             # 64→128
        self.down4 = DownSample(b*4, b*8)             # 128→256

        self.bottleneck = DoubleConv(b*8, b*16)       # 256→512

        self.up1 = UpSample(b*16, b*8)                # 512→256
        self.up2 = UpSample(b*8,  b*4)                # 256→128
        self.up3 = UpSample(b*4,  b*2)                # 128→64
        self.up4 = UpSample(b*2,  b)                  # 64→32

        self.out = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down1(x)
        down_2, p2 = self.down2(p1)
        down_3, p3 = self.down3(p2)
        down_4, p4 = self.down4(p3)

        b = self.bottleneck(p4)

        up_1 = self.up1(b, down_4)
        up_2 = self.up2(up_1, down_3)
        up_3 = self.up3(up_2, down_2)
        up_4 = self.up4(up_3, down_1)

        out = self.out(up_4)
        
        # out = torch.sigmoid(out)  # Apply sigmoid to ensure output is in [0, 1] range
        return out