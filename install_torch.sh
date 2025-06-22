# Install PyTorch with CUDA 12.1 support
#!/bin/bash
source .venv/bin/activate

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
