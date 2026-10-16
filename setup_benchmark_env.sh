#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${BENCHMARK_CONDA_ENV:-torch_ov}"

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Cannot find conda.sh under ~/miniforge3 or ~/anaconda3" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip "setuptools<81" wheel
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

python -m pip install \
    numpy==1.26.4 \
    pillow \
    openvino==2026.4.0 \
    "setuptools==80.10.2"

python - <<'PY'
import importlib.metadata
import openvino
import PIL
import torch

print("Shared environment validated:")
print("  torch", torch.__version__)
print("  openvino", openvino.__version__)
print("  numpy", importlib.metadata.version("numpy"))
print("  pillow", PIL.__version__)
PY

python "$ROOT_DIR/unet_benchmark.py" --check-only \
    --model-dir "$ROOT_DIR/UNet-models" \
    --openvino-dir "$ROOT_DIR/models_ov" \
    --image-dir "$ROOT_DIR/test_images"
