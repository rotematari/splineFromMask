import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = ConvBlock(128, 128)

        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(64 + 128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(32 + 64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(16 + 32, 16)

        # Output
        self.out = nn.Conv2d(16, 1, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)         # [B, 32, 512, 512]
        e2 = self.enc2(self.pool(e1))  # [B, 64, 256, 256]
        e3 = self.enc3(self.pool(e2))  # [B, 128, 128, 128]

        # Bottleneck
        m = self.middle(self.pool(e3))  # [B, 128, 64, 64]

        # Decoder
        d3 = self.up3(m)  # [B, 64, 128, 128]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)  # [B, 32, 256, 256]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)  # [B, 16, 512, 512]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out(d1)  # [B, 1, 512, 512]
        return self.activation(out)
