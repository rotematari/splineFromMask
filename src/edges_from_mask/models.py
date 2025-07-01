from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp

class DeepLabTransformer(nn.Module):
    def __init__(self,
                 num_pts: int,
                 img_size: int = 256,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_enc_layers: int = 4,
                 num_dec_layers: int = 4):
        super().__init__()
        self.img_size = img_size
        
        # 1) Load DeeplabV3+ backbone encoder
        backbone = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1  # dummy
        )
        self.backbone = backbone.encoder
        
        # Get the feature map size after backbone processing
        # For ResNet18 encoder, the output is typically 1/32 of input size
        self.feat_size = img_size // 16  # e.g., 256//32 = 8
        self.feat_seq_len = self.feat_size * self.feat_size  # e.g., 8*8 = 64
        
        # Get backbone output channels (ResNet18 final layer has 512 channels)
        self.backbone_channels = 512
        
        # Project backbone features → d_model
        self.input_proj = nn.Linear(self.backbone_channels, d_model)

        # Positional embeddings for H*W tokens
        self.pos_embed = nn.Parameter(torch.randn(self.feat_seq_len, 1, d_model))

        # 2) Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_enc_layers)

        # 3) Transformer decoder + learnable queries
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_dec_layers)
        self.query_embed = nn.Embedding(num_pts, d_model)

        # 4) MLP to map each query output → (x,y)
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        
        # --- 1) Get CNN features using encoder only ---
        features = self.backbone(x)[-1]

        # Flatten spatial dims → sequence
        B, C, H, W = features.shape
        seq = features.view(B, C, H*W).permute(2, 0, 1)     # (S=H*W, B, C)
        seq = self.input_proj(seq)                      # → (S, B, d_model)
        
        # Handle positional embedding size mismatch
        seq_len = seq.shape[0]
        if seq_len != self.feat_seq_len:
            # Resize positional embeddings if needed
            pos_embed = self.pos_embed[:seq_len] if seq_len < self.feat_seq_len else self.pos_embed
        else:
            pos_embed = self.pos_embed
            
        seq = seq + pos_embed                           # Add positional embeddings

        # --- 2) Encode ---
        memory = self.encoder(seq)                      # (S, B, d_model)

        # --- 3) Decode with point queries ---
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_pts, B, d_model)
        # tgt = torch.zeros_like(query_pos)               # (num_pts, B, d_model)
        tgt = query_pos
        hs = self.decoder(tgt, memory)                  # (num_pts, B, d_model)

        # --- 4) Regress (x,y) per query ---
        coords = self.coord_head(hs)                    # (num_pts, B, 2)
        coords = coords.permute(1, 0, 2)                # (B, num_pts, 2)
        return coords












class EncoderHeatmapHead(nn.Module):
    def __init__(self,config:Dict = None):
        super().__init__()
        # load with an arbitrary number of classes (e.g. 1) just to build the model
        backbone = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=config['num_pts']
        )
        
        self.backbone = backbone
        

    def forward(self, x):
        feats = self.backbone(x)        # returns feature map of shape (B, in_ch, H, W)
        
        return feats

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

        # --- Backbone: ResNet18 encoder ---
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # freeze the backbone layers
        for param in resnet.parameters():
            param.requires_grad = True
        # Unfreeze the last layer (layer4)
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        # Remove avgpool and fully-connected layers
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        # freeze the encoder layers

        self.feature_dim = 512  # final channel count of ResNet18

        # # --- Regression head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp =  nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 2),
            # nn.ReLU(inplace=True),
            # # nn.Dropout(dropout),
            # nn.Linear(mlp_hidden, mlp_hidden//2),
            # nn.ReLU(inplace=True),
            # # nn.Dropout(dropout),
            # nn.Linear(mlp_hidden//2, 2),
            
        )
        self.mlp2 =  nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 2),
            # nn.ReLU(inplace=True),
            # # nn.Dropout(dropout),
            # nn.Linear(mlp_hidden, mlp_hidden//2),
            # nn.ReLU(inplace=True),
            # # nn.Dropout(dropout),
            # nn.Linear(mlp_hidden//2, 2),
            
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Regression output of shape (B, num_spline_points, out_dim).
        """
 
        # Extract features
        feat = self.encoder(x)                # (B, feature_dim, H', W')
        pooled = self.avgpool(feat).view(feat.size(0), -1)  # (B, feature_dim)

        # MLP regression
        p1 = self.mlp(pooled)                  # (B, mlp_hidden//2)
        p2 = self.mlp2(pooled)                  # (B, mlp_hidden//2)
        x = torch.cat([p1, p2], dim=1)         # (B, mlp_hidden)

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
    
    
    
if __name__ == "__main__":
    # Example usage
    # model = EncoderHeatmapHead()
    model = DeepLabTransformer(num_pts=10, d_model=256, nhead=8, num_enc_layers=4, num_dec_layers=4)
    print(model)
    input_tensor = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image
    # Load one image
    path = "src/spline_dataset/ds_256x256_32splines_10pts_4-10ctrl_k3_s0p9_dim2/images/00001.png"  # Replace with your image path
    from PIL import Image
    import torchvision.transforms as transforms

    # Load and preprocess grayscale image
    image = Image.open(path).convert("L")  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    input_tensor = transform(image).unsqueeze(0)
    model.eval()  # Set the model to evaluation mode
    output = model(input_tensor)
    print("Output shape:", output.shape)
    
    
    # plot the heatmap
    import matplotlib.pyplot as plt
    # Convert output to numpy for plotting
    heatmap = output[0].detach().numpy()  # Shape: (2, 256, 256)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot first channel
    im1 = axes[0].imshow(heatmap[0], cmap='hot', interpolation='nearest')
    axes[0].set_title('Heatmap Channel 1')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Plot second channel
    im2 = axes[1].imshow(heatmap[1], cmap='hot', interpolation='nearest')
    axes[1].set_title('Heatmap Channel 2')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()