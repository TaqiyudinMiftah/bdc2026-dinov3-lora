# CUDA 12.2 / NVIDIA Driver 535 PyTorch Setup

If `nvidia-smi` works but PyTorch says:

```text
CUDA initialization: The NVIDIA driver on your system is too old
CUDA available: False
```

then your virtual environment probably installed a PyTorch wheel built for a newer CUDA runtime, such as CUDA 12.6 or CUDA 12.8.

For this server:

```text
NVIDIA driver: 535.x
nvidia-smi CUDA Version: 12.2
GPU: NVIDIA L40
```

use the PyTorch CUDA 12.1 wheel:

```text
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
index: https://download.pytorch.org/whl/cu121
```

CUDA 12.1 wheels are safer for a driver that reports CUDA 12.2 than newer cu126/cu128 wheels.

## Fix your current virtualenv

From the repo root with `.venv` activated:

```bash
git pull
chmod +x scripts/install_torch_cuda122.sh
./scripts/install_torch_cuda122.sh
```

Or manually:

```bash
uv pip uninstall -y torch torchvision torchaudio
uv pip install -r requirements-torch-cu121.txt
uv pip install -r requirements.txt
```

## Verify

```bash
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

Expected:

```text
torch CUDA build: 12.1
CUDA available: True
Device: NVIDIA L40
```

## Pick a free GPU

Your `nvidia-smi` showed GPU 2 is already busy. Use GPU 0 or GPU 1:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora \
  --image-size 224 \
  --epochs 20 \
  --batch-size 4 \
  --valid-batch-size 8 \
  --grad-accum 4 \
  --use-class-weights
```

For a stronger L40 run, you can try larger resolution:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora_384 \
  --image-size 384 \
  --epochs 25 \
  --batch-size 2 \
  --valid-batch-size 4 \
  --grad-accum 8 \
  --use-class-weights \
  --early-stopping-patience 8
```
