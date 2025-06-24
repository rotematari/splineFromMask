import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import models
import random
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_len, d_model)     # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)         # not a parameter

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)].unsqueeze(1)
        return x

class Mask2ControlPointsTransformer(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_steps: int,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 4,
    ):
        """
        grid_size: G (vocab size = G*G)
        num_steps: S (sequence length)
        """
        super().__init__()
        self.G = grid_size
        self.V = grid_size * grid_size
        self.S = num_steps
        self.d_model = d_model

        # 1) CNN encoder -> feature map
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 128×128
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                # 64×64
            nn.Conv2d(64, d_model, 3, padding=1), nn.ReLU(),
        )
        # flatten spatial dims -> sequence of length L=64*64
        self.memory_pos_enc = PositionalEncoding(d_model, max_len=64*64)

        # 2) token embedding + positional encoding for target sequence
        self.token_emb = nn.Embedding(self.V + 1, d_model)  # V for <SOS>
        self.tgt_pos_enc = PositionalEncoding(d_model, max_len=num_steps)

        # 3) Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers
        )

        # 4) final projection
        self.head = nn.Linear(d_model, self.V)

    def generate_square_subsequent_mask(self, sz):
        # standard causal mask
        mask = torch.triu(torch.ones(sz, sz), 1).bool()
        return mask  # (sz, sz)

    def forward(self, mask, target_ids=None):
        """
        mask:    (B,1,H,W)
        target_ids: (B, S) long, values in [0..V] where V is <SOS>
                    if None: runs in “inference mode” (greedy AR)
        returns logits: (B, S, V)
        """
        B = mask.size(0)

        # --- build memory ---
        feat = self.encoder(mask)                  # (B, d_model, 64,64)
        L = feat.shape[-2] * feat.shape[-1]        # 4096
        mem = feat.flatten(2)                      # (B, d_model, L)
        mem = mem.permute(2, 0, 1)                  # (L, B, d_model)
        mem = self.memory_pos_enc(mem)             # add spatial pos enc

        # --- prepare target sequence ---
        # start-of-seq tokens
        if target_ids is None:
            # initialize with all <SOS> for inference
            tgt_ids = torch.full((B, self.S), self.V, device=mask.device, dtype=torch.long)
        else:
            tgt_ids = target_ids

        # embed + add positional enc
        tgt = self.token_emb(tgt_ids)               # (B, S, d_model)
        tgt = tgt.permute(1, 0, 2)                  # (S, B, d_model)
        tgt = self.tgt_pos_enc(tgt)

        # causal mask so we can't “see” future tokens
        tgt_mask = self.generate_square_subsequent_mask(self.S).to(mask.device)

        # --- decode ---
        dec_out = self.decoder(
            tgt,             # (S, B, d_model)
            mem,             # (L, B, d_model)
            tgt_mask=tgt_mask
        )                   # (S, B, d_model)

        # --- project to logits ---
        dec_out = dec_out.permute(1, 0, 2)           # (B, S, d_model)
        logits  = self.head(dec_out)                 # (B, S, V)
        return logits

class Mask2ControlPoints(nn.Module):
    def __init__(self, grid_size=32, hidden_dim=256, num_steps=50):
        super().__init__()
        self.G = grid_size
        self.V = grid_size * grid_size    # vocabulary size
        self.num_steps = num_steps

        # 1) Simple CNN encoder → feature vector
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_proj = nn.Linear(32, hidden_dim)

        # 2) Token embedding for decoder input
        self.token_emb = nn.Embedding(self.V+1, hidden_dim)  
        # +1 for a “start‐of‐sequence” token at index V

        # 3) LSTM decoder
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # 4) Output head → logits over G² bins
        self.head = nn.Linear(hidden_dim, self.V)

        self.ctx_proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self,
                mask: torch.Tensor,
                target_ids: torch.LongTensor = None,
                teacher_forcing_ratio: float = 0.50
    ) -> torch.Tensor:
        """
        Args:
          mask: (B,1,H,W) input
          target_ids: (B, S) true token IDs, or None (inference mode)
          teacher_forcing_ratio: float in [0,1]
        Returns:
          logits: (B, S, V)
        """
        B = mask.size(0)
        S = self.num_steps
        V = self.V

        # --- encode image as before ---
        x = self.encoder(mask).view(B, -1)
        h = self.feat_proj(x)                
        c = torch.zeros_like(h)

        # start token = V
        tok = torch.full((B,), V, dtype=torch.long, device=mask.device)
        ctx = self.ctx_proj(self.feat_proj(x))   # (B,D)

        logits = []
        for t in range(S):
            emb = self.token_emb(tok)
            lstm_in = emb + ctx                  # re-inject context
            h, c = self.lstm(lstm_in, (h, c))
            logit = self.head(h)             # (B, V)
            logits.append(logit)

            # decide whether to teacher-force
            if target_ids is not None and random.random() < teacher_forcing_ratio:
                tok = target_ids[:, t]       # use ground-truth
            else:
                tok = logit.argmax(dim=-1)   # use model’s own prediction

        return torch.stack(logits, dim=1)   # (B, S, V)
class EncoderTransformerDecoder(nn.Module):
    """
    ResNet‑18 encoder + Transformer decoder for regressing spline control points.

    Args:
        num_spline_points (int): Number of spline control points (queries).
        in_channels (int): Input image channels (e.g., 1 for binary masks).
        pretrained (bool): Whether to load ImageNet pretrained weights for the backbone.
        d_model (int): Dimensionality of the Transformer feature embeddings.
        nhead (int): Number of attention heads in the Transformer.
        num_decoder_layers (int): Number of Transformer decoder layers.
        dim_feedforward (int): Hidden dimension of the feedforward network in the decoder.
        dropout (float): Dropout probability in the Transformer.
        out_dim (int): Output dimension per control point (e.g., 2D or 3D).
    """
    def __init__(
        self,
        num_spline_points: int,
        in_channels: int = 1,
        pretrained: bool = False,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        out_dim: int = 2,
    ):
        super().__init__()
        self.num_spline_points = num_spline_points
        self.out_dim = out_dim

        # --- Backbone encoder (ResNet‑18 without avgpool & fc) ---
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        if in_channels != 3:
            # Adapt first conv layer for in_channels
            resnet.conv1 = nn.Conv2d(
                in_channels,
                resnet.conv1.out_channels,
                kernel_size=resnet.conv1.kernel_size,
                stride=resnet.conv1.stride,
                padding=resnet.conv1.padding,
                bias=False
            )
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Remove avgpool & fc
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        self.feature_dim = 512  # ResNet‑18 last channel count

        # --- Project CNN features to Transformer dimension ---
        self.input_proj = nn.Conv2d(self.feature_dim, d_model, kernel_size=1)

        # --- Positional encoding for memory ---
        self.pos_embedding = nn.Parameter(torch.rand(1, d_model, 1))  # learned pos embedding per spatial location

        # --- Transformer decoder setup ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Learnable query embeddings (one per spline control point)
        self.query_embed = nn.Parameter(torch.rand(self.num_spline_points, d_model))

        # Final linear head to predict (x, y) or (x, y, z)
        self.linear_head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input mask tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Control-point coordinates of shape (B, num_spline_points, out_dim).
        """
        B = x.size(0)

        # 1) CNN encoder
        feat = self.encoder(x)  # (B, 512, H', W')

        # 2) Project to d_model and flatten spatial dims
        proj = self.input_proj(feat)  # (B, d_model, H', W')
        B, C, Hf, Wf = proj.shape
        # add positional embedding (broadcasted)
        proj = proj.flatten(2)  # (B, d_model, S)
        S = Hf * Wf
        # memory: (S, B, d_model)
        memory = proj.permute(2, 0, 1)

        # 3) Prepare queries as tgt
        # queries: (Q, B, d_model)
        tgt = self.query_embed.unsqueeze(1).repeat(1, B, 1)

        # 4) Transformer decoding (tgt plus memory)
        hs = self.transformer_decoder(tgt, memory)  # (Q, B, d_model)

        # 5) Predict control points
        hs = hs.permute(1, 0, 2)  # (B, Q, d_model)
        coords = self.linear_head(hs)  # (B, Q, out_dim)

        return coords

class EncoderRegressionHead(nn.Module):
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
        num_spline_points: int,
        in_channels: int = 1,
        pretrained: bool = False,
        out_dim: int = 2,
        mlp_hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_spline_points = num_spline_points
        self.out_dim = out_dim

        # --- Backbone: ResNet18 encoder ---
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Adapt first conv if input channels != 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels,
                resnet.conv1.out_channels,
                kernel_size=resnet.conv1.kernel_size,
                stride=resnet.conv1.stride,
                padding=resnet.conv1.padding,
                bias=False,
            )
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Remove avgpool and fully-connected layers
        modules = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*modules)
        self.feature_dim = 512  # final channel count of ResNet18

        # --- Regression head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.feature_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, num_spline_points * out_dim)

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
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = self.fc2(x)                       # (B, num_spline_points*out_dim)

        # Reshape to (B, num_spline_points, out_dim)
        return x.view(-1, self.num_spline_points, self.out_dim)
















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
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 32)
        self.down_convolution_2 = DownSample(32, 64)
        self.down_convolution_3 = DownSample(64, 128)
        self.down_convolution_4 = DownSample(128, 256)
        self.bottle_neck = DoubleConv(256, 256)
# --- your SoftArgmax2d from before ---
class SoftArgmax2d(nn.Module):
    """
    Differentiable soft‐argmax over 2D heatmaps.
    Input:  heatmaps (B, C, H, W)
    Output: coords   (B, C, 2) in normalized [0,1] coords
    """
    def __init__(self, beta: float = 1.0, normalize: bool = True):
        super().__init__()
        self.beta = beta
        self.normalize = normalize

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        B, C, H, W = heatmaps.shape
        flat = heatmaps.view(B, C, -1)               # (B, C, H*W)
        probs = F.softmax(flat * self.beta, dim=-1)  # (B, C, H*W)

        ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
        xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")       # (H, W)
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # (H*W, 2)

        coords = probs @ grid       # (B, C, 2)
        if self.normalize:
            coords[..., 0] /= (W - 1)
            coords[..., 1] /= (H - 1)
        return coords   # (B, C, 2)
    

class UNetSpline(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_points: int = 200,
                 beta: float = 10.0,
                 image_size: int = 256):
        super().__init__()
        # instantiate your original UNet so that its "out" conv
        # spits out `num_points` heatmaps instead of segmentation classes:
        self.unet = UNet(in_channels=in_channels,
                         num_classes=num_points)
        # differentiable collapse:
        self.softargmax = SoftArgmax2d(beta=beta)
        
        self.img_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, in_channels, 256, 256)
        returns (B, num_points, 2)  with normalized coords in [0,1]
        """
        heatmaps = self.unet(x)           # → (B, num_points, 256, 256)
        coords_norm   = self.softargmax(heatmaps)
        


        return coords_norm

if  __name__ == "__main__":
    model = UNetSpline(in_channels=1)
    # print(model)
    
    
    dummy_input = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image
    output = model(dummy_input)
    print("Output shape:", output.shape)
    
    
    
    
    import matplotlib.pyplot as plt
    
    
    output[:, :, 0] *= (256 - 1)  # x-coord
    output[:, :, 1] *= (256 - 1)  # y-coord
    plt.scatter(output[:, :, 0].detach().cpu().numpy(), output[:, :, 1].detach().cpu().numpy(), s=1)
    plt.xlim(0, 255)
    plt.ylim(0, 255)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("SoftArgmax2d Output")
    plt.show()
