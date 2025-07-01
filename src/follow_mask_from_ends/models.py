from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp


class EncoderEdgePointsHM(nn.Module):
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


    
    
if __name__ == "__main__":
    # Example usage
    # model = EncoderHeatmapHead()
    model = EncoderEdgePointsHM()
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