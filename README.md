# BDC 2026 DINOv3 LoRA Pipeline

Training and inference pipeline for the BDC Satria Data 2026 waste image classification task.

The pipeline uses:

- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- LoRA fine-tuning with `r=16`, `alpha=32`
- corrected DINOv3 LoRA target modules: `q_proj`, `v_proj`
- Stratified 5-fold cross validation
- OOF Macro-F1 evaluation
- early stopping on validation Macro-F1
- class imbalance handling
- TTA + fold ensemble for final `submission.csv`

## Expected dataset structure

```text
BDC2026/
├── train/
│   ├── 0_Recyclable/
│   ├── 1_Electronic/
│   └── 2_Organic/
├── test/
└── submission.csv
```

Label mapping:

```text
0 = Recyclable
1 = Electronic
2 = Organic
```

## Setup

### Recommended: uv virtual environment

Install `uv` if it is not installed yet:

```bash
pip install uv
```

Create and activate a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
uv venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies inside the uv environment:

```bash
uv pip install -r requirements.txt
```

Check that PyTorch can see your GPU:

```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

If you are using Colab, you can also use:

```bash
pip install -r requirements.txt
```

DINOv3 may require accepting the model terms on Hugging Face and logging in:

```bash
huggingface-cli login
```

or set:

```bash
export HF_TOKEN=your_token_here
```

## Download dataset from Google Drive in Colab

```bash
python scripts/download_drive_dataset.py \
  --url "https://drive.google.com/drive/folders/1mVsWMnr2nmRotVjndbej9ONQmM6y5-q9" \
  --output /content/BDC2026
```

## Train 5-fold CV

```bash
python train.py \
  --data-root /content/BDC2026 \
  --output-dir ./outputs_dinov3_lora \
  --image-size 224 \
  --epochs 8 \
  --batch-size 4 \
  --valid-batch-size 8 \
  --grad-accum 4 \
  --use-class-weights \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 1e-4
```

For a stronger GPU run, try:

```bash
python train.py \
  --data-root /content/BDC2026 \
  --output-dir ./outputs_dinov3_lora_384 \
  --image-size 384 \
  --epochs 12 \
  --batch-size 2 \
  --valid-batch-size 4 \
  --grad-accum 8 \
  --use-class-weights \
  --early-stopping-patience 4 \
  --early-stopping-min-delta 1e-4
```

## Early stopping

Early stopping monitors **validation Macro-F1** for each fold.

Defaults:

```bash
--early-stopping-patience 3
--early-stopping-min-delta 1e-4
```

Meaning:

- `patience=3`: stop a fold after 3 consecutive epochs without meaningful Macro-F1 improvement.
- `min_delta=1e-4`: improvement must be greater than `0.0001` to reset patience.
- Set `--early-stopping-patience 0` to disable early stopping.

The best checkpoint for each fold is still saved as:

```text
outputs_dinov3_lora/fold0_best.pt
outputs_dinov3_lora/fold1_best.pt
...
```

## Predict final submission

```bash
python predict.py \
  --data-root /content/BDC2026 \
  --checkpoint-dir ./outputs_dinov3_lora \
  --output ./submission_NamaTim.csv \
  --tta
```

## Imbalance strategy

Recommended first run:

```bash
--use-class-weights --class-weight-mode inverse
```

If Electronic F1 is still low, try sampler instead:

```bash
--use-weighted-sampler --sampler-weight-mode sqrt_inverse
```

Avoid using test data for training, validation, model selection, or tuning. Use only OOF validation scores to choose your configuration.
