#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_BIN="${PIP_BIN:-uv pip}"

cat <<'EOF'
Installing PyTorch CUDA 12.1 wheels for an NVIDIA driver that reports CUDA 12.2.
This avoids newer cu126/cu128 wheels that may require a newer driver.
EOF

echo "Python: $(${PYTHON_BIN} --version)"
echo "Pip command: ${PIP_BIN}"

# Remove incompatible torch packages if they were installed from the default PyPI index.
${PIP_BIN} uninstall -y torch torchvision torchaudio || true

# Install the CUDA 12.1 PyTorch wheel. It is compatible with CUDA 12.2 drivers.
${PIP_BIN} install -r requirements-torch-cu121.txt

# Install the rest of the dependencies.
${PIP_BIN} install -r requirements.txt

cat <<'EOF'

Verifying CUDA from PyTorch...
EOF

${PYTHON_BIN} - <<'PY'
import torch
print('torch version:', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}:', torch.cuda.get_device_name(i))
PY
