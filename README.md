# BDC 2026 DINOv3 LoRA Pipeline

Training and inference pipeline for the BDC Satria Data 2026 waste image classification task.

The pipeline uses:

- `facebook/dinov3-vitl16-pretrain-lvd1689m`
- LoRA fine-tuning with `r=16`, `alpha=32`
- corrected DINOv3 LoRA target modules: `q_proj`, `v_proj`
- Stratified 5-fold cross validation
- OOF Macro-F1 evaluation
- higher epoch budget with early stopping on validation Macro-F1
- ReduceLROnPlateau scheduler that lowers LR when validation loss plateaus
- class imbalance handling
- EDA, visualization, and cleaning reports
- exact duplicate, perceptual duplicate, and optional DINO-embedding duplicate detection
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

### Install dependencies on the Debian L40 server

This server reports NVIDIA driver 535 and CUDA 12.2 in `nvidia-smi`. Do **not** let pip install the newest default PyTorch wheel, because it may pull a CUDA 12.6/12.8 build that needs a newer driver.

Use the repo installer:

```bash
git pull
chmod +x scripts/install_torch_cuda122.sh
./scripts/install_torch_cuda122.sh
```

Or install manually:

```bash
uv pip uninstall -y torch torchvision torchaudio
uv pip install -r requirements-torch-cu121.txt
uv pip install -r requirements.txt
```

The PyTorch CUDA requirement file pins:

```text
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
CUDA wheel index: https://download.pytorch.org/whl/cu121
```

Check that PyTorch can see your GPU:

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

Expected on the L40 server:

```text
torch CUDA build: 12.1
CUDA available: True
Device: NVIDIA L40
```

More details are in:

```text
docs/cuda_setup.md
```

### Colab setup

If you are using Colab, you can use:

```bash
pip install -r requirements-torch-cu121.txt
pip install -r requirements.txt
```

## Environment variables and tokens

A safe template is provided in `.env.example`. Copy it to `.env` and fill in your real tokens locally:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
HF_TOKEN=your_huggingface_token_here
BDC2026_DRIVE_URL=https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7
BDC2026_DATA_ROOT=./BDC2026
BDC2026_OUTPUT_DIR=./outputs_dinov3_lora
BDC2026_EDA_OUTPUT_DIR=./eda_outputs
```

Use `./BDC2026` on Linux servers and local machines. Use `/content/BDC2026` only on Google Colab. The real `.env` file is ignored by Git and should not be committed. The training code automatically loads `.env` with `python-dotenv`, so `HF_TOKEN` can be used for gated Hugging Face models such as DINOv3.

DINOv3 may require accepting the model terms on Hugging Face and logging in. You can either use `.env` with `HF_TOKEN`, or use the current Hugging Face CLI:

```bash
hf auth login
```

Check login status with:

```bash
hf auth whoami
```

## Download dataset from Google Drive

### Linux server or local machine

Use a path you can write to, such as `./BDC2026`:

```bash
python scripts/download_drive_dataset.py \
  --url "https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7" \
  --output ./BDC2026
```

Or use the values from `.env`:

```bash
python scripts/download_drive_dataset.py
```

### Google Colab

Use `/content/BDC2026` only in Colab:

```bash
python scripts/download_drive_dataset.py \
  --url "https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7" \
  --output /content/BDC2026
```

## EDA, visualization, and data cleaning

Run the EDA pipeline first before training.

For Linux server/local path:

```bash
python scripts/eda_cleaning.py \
  --data-root ./BDC2026 \
  --output-dir ./eda_outputs
```

For Colab path:

```bash
python scripts/eda_cleaning.py \
  --data-root /content/BDC2026 \
  --output-dir ./eda_outputs
```

This creates:

```text
eda_outputs/
├── train_manifest.csv
├── corrupted_images.csv
├── exact_duplicate_groups.csv
├── image_md5.csv
├── phash_duplicate_pairs.csv
├── image_phash.csv
├── cleaning_candidates.csv
├── eda_summary.json
└── figures/
    ├── class_distribution.png
    ├── width_hist.png
    ├── height_hist.png
    ├── aspect_ratio_hist.png
    ├── file_size_bytes_hist.png
    └── samples_*.png
```

For stronger near-duplicate detection with DINO embeddings:

```bash
python scripts/eda_cleaning.py \
  --data-root ./BDC2026 \
  --output-dir ./eda_outputs_dino \
  --use-dino-duplicates \
  --embedding-batch-size 16 \
  --embedding-sim-threshold 0.985
```

DINO duplicate outputs:

```text
eda_outputs_dino/
├── dino_embeddings.npy
├── dino_embedding_index.csv
└── dino_duplicate_pairs.csv
```

Cleaning suggestions used by the script:

- Remove or repair corrupted/unreadable images.
- Remove exact byte duplicates when the duplicate group has the same label.
- Manually review cross-label duplicates because those may indicate noisy labels.
- Manually review DINO-embedding duplicates; they are semantic near-duplicates, so do not blindly delete all of them.
- Manually review extreme aspect-ratio or very small images.

To create a cleaned copy without modifying the original dataset:

```bash
python scripts/eda_cleaning.py \
  --data-root ./BDC2026 \
  --output-dir ./eda_outputs \
  --make-clean-copy \
  --clean-output ./BDC2026_clean \
  --copy-mode copy
```

Then train using:

```bash
--data-root ./BDC2026_clean
```

Important: the clean-copy mode only auto-removes corrupt files and safe exact duplicates. Cross-label and semantic duplicates are reported for manual review.

## Train 5-fold CV

Default training now uses a larger epoch budget and higher early-stopping patience.

GPU 2 may already be busy on the shared server. Use an idle GPU, for example GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora \
  --image-size 224 \
  --epochs 20 \
  --batch-size 4 \
  --valid-batch-size 8 \
  --grad-accum 4 \
  --use-class-weights \
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 2 \
  --plateau-threshold 1e-4 \
  --min-lr 1e-7 \
  --early-stopping-patience 6 \
  --early-stopping-min-delta 1e-4
```

For a stronger GPU run, try:

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
  --scheduler plateau \
  --plateau-factor 0.5 \
  --plateau-patience 3 \
  --plateau-threshold 1e-4 \
  --min-lr 1e-7 \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 1e-4
```

## Learning-rate scheduling

The default scheduler is:

```bash
--scheduler plateau
```

It monitors **validation loss**. If validation loss does not reduce enough for `--plateau-patience` epochs, it lowers both LoRA and classifier learning rates by `--plateau-factor`.

Defaults:

```bash
--scheduler plateau
--plateau-factor 0.5
--plateau-patience 2
--plateau-threshold 1e-4
--min-lr 1e-7
```

You can still use cosine scheduling:

```bash
--scheduler cosine
```

## Early stopping

Early stopping monitors **validation Macro-F1** for each fold.

Defaults:

```bash
--early-stopping-patience 6
--early-stopping-min-delta 1e-4
```

Meaning:

- `patience=6`: stop a fold after 6 consecutive epochs without meaningful Macro-F1 improvement.
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
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --data-root ./BDC2026 \
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
