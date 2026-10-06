# Creates .venv with CUDA 12.4 PyTorch (RTX 3090 / driver >= 550) and project deps.
$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch, open3d, timm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| open3d', open3d.__version__, '| timm', timm.__version__)"
