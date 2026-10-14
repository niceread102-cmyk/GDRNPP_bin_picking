# Windows: creates .venv with CUDA PyTorch and the project dependencies (requirements.txt).
#
#   powershell -ExecutionPolicy Bypass -File setup_venv.ps1                    # CUDA 12.4 (RTX 3090, driver >= 550)
#   powershell -ExecutionPolicy Bypass -File setup_venv.ps1 -TorchCuda cu130   # CUDA 13.0 (recent drivers / GPUs)
#   powershell -ExecutionPolicy Bypass -File setup_venv.ps1 -TorchCuda cpu     # no GPU
#   powershell -ExecutionPolicy Bypass -File setup_venv.ps1 -Recreate          # delete an existing .venv first
param(
    [ValidateSet("cu124", "cu130", "cpu")][string]$TorchCuda = "cu124",
    [string]$Python = "3.12",
    [switch]$Recreate
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$torch = @{
    "cu124" = @("torch==2.6.0", "torchvision==0.21.0")
    "cu130" = @("torch==2.14.1", "torchvision==0.29.1")
    "cpu"   = @("torch", "torchvision")
}[$TorchCuda]

if ($Recreate -and (Test-Path .venv)) { Remove-Item -Recurse -Force .venv }
if (-not (Test-Path .\.venv\Scripts\python.exe)) { py -$Python -m venv .venv }

$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install @torch --index-url "https://download.pytorch.org/whl/$TorchCuda"
& $py -m pip install -r requirements.txt
# make the venv selectable as a Jupyter kernel (VS Code / Jupyter) for notebooks/*.ipynb
& $py -m ipykernel install --user --name gdrnpp-bin6d --display-name "GDRNPP bin 6DoF (.venv)"
# git stores notebooks without outputs (.gitattributes); define that filter for this clone
if (Get-Command git -ErrorAction SilentlyContinue) {
    git config filter.strip-notebook-output.clean ".venv/Scripts/python.exe tools/strip_notebook_outputs.py"
}
& $py -c "import torch, open3d, timm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| open3d', open3d.__version__, '| timm', timm.__version__)"
