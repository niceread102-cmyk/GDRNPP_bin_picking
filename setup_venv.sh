#!/usr/bin/env bash
# Linux: creates .venv with PyTorch and the project dependencies (requirements.txt), registers the Jupyter
# kernel and the git filter that stores notebooks without outputs. Safe to re-run (keeps an existing .venv).
#
#   bash setup_venv.sh                    # PyTorch build picked from the NVIDIA driver (see below)
#   TORCH_CUDA=cu130 bash setup_venv.sh   # force CUDA 13.0 (torch 2.14.1; Blackwell GPUs such as GB10 / DGX Spark)
#   TORCH_CUDA=cu124 bash setup_venv.sh   # force CUDA 12.4 (torch 2.6.0; older drivers)
#   TORCH_CUDA=cpu bash setup_venv.sh     # no GPU
#   PYTHON=python3.11 bash setup_venv.sh  # another interpreter (default: python3.12, else python3; >= 3.10)
#   RECREATE=1 bash setup_venv.sh         # delete an existing .venv first
set -euo pipefail
cd "$(dirname "$0")"

# --- PyTorch build: CUDA 13.0 wheels need driver >= 580, CUDA 12.4 wheels driver >= 550
if [ -z "${TORCH_CUDA:-}" ]; then
    driver=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1 || true)"
    fi
    if [[ "$driver" =~ ^[0-9]+$ ]] && [ "$driver" -ge 580 ]; then
        TORCH_CUDA=cu130
    elif [[ "$driver" =~ ^[0-9]+$ ]] && [ "$driver" -ge 550 ]; then
        TORCH_CUDA=cu124
    else
        TORCH_CUDA=cpu
        echo "warning: no NVIDIA driver >= 550 found (nvidia-smi: '${driver:-none}'); installing CPU-only PyTorch." >&2
        echo "         Set TORCH_CUDA=cu130 or cu124 to override." >&2
    fi
    echo "NVIDIA driver ${driver:-none} -> TORCH_CUDA=$TORCH_CUDA"
fi
case "$TORCH_CUDA" in
    cu130) TORCH="torch==2.14.1 torchvision==0.29.1" ;;
    cu124) TORCH="torch==2.6.0 torchvision==0.21.0" ;;
    cpu)   TORCH="torch torchvision" ;;
    *)     echo "unknown TORCH_CUDA=$TORCH_CUDA (use cu130, cu124 or cpu)" >&2; exit 1 ;;
esac

# --- interpreter (absolute path: it is also stored in the git filter below)
PYTHON="$(command -v "${PYTHON:-python3.12}" || command -v python3 || true)"
if [ -z "$PYTHON" ]; then
    echo "error: no python3.12 / python3 found; install Python >= 3.10 or set PYTHON=/path/to/python" >&2
    exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {sys.version.split()[0]}"'

# --- virtual environment
if [ "${RECREATE:-0}" = "1" ] && [ -d .venv ]; then
    rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
    if ! "$PYTHON" -m venv .venv; then
        rm -rf .venv
        echo "error: '$PYTHON -m venv' failed. On Debian/Ubuntu install the venv module first, e.g." >&2
        echo "       sudo apt install python3.12-venv" >&2
        exit 1
    fi
else
    echo "reusing .venv ($(.venv/bin/python --version)); RECREATE=1 to rebuild it"
fi

.venv/bin/python -m pip install --upgrade pip
# shellcheck disable=SC2086
.venv/bin/python -m pip install $TORCH --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
.venv/bin/python -m pip install -r requirements.txt

# --- Jupyter kernel for notebooks/*.ipynb (VS Code / Jupyter: "GDRNPP bin 6DoF (.venv)")
.venv/bin/python -m ipykernel install --user --name gdrnpp-bin6d --display-name "GDRNPP bin 6DoF (.venv)"

# --- git stores notebooks without outputs (.gitattributes); define that filter for this clone.
#     Uses the system interpreter (stdlib only), so it keeps working if .venv is rebuilt.
if git rev-parse --git-dir >/dev/null 2>&1; then
    git config filter.strip-notebook-output.clean "\"$PYTHON\" tools/strip_notebook_outputs.py"
    echo "git filter strip-notebook-output -> $PYTHON tools/strip_notebook_outputs.py"
fi

# --- check
.venv/bin/python - "$TORCH_CUDA" <<'EOF'
import sys
import open3d, timm, torch
cuda = torch.cuda.is_available()
print(f"torch {torch.__version__}, cuda {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else "")
      + f" | open3d {open3d.__version__} | timm {timm.__version__}")
if sys.argv[1] != "cpu" and not cuda:
    print("warning: a CUDA build of PyTorch was installed but no GPU is usable (driver / CUDA version mismatch?)",
          file=sys.stderr)
EOF
