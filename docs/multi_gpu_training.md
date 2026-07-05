# Auto Multi-GPU Training

Use this when the server has several GPUs but some are already busy.

The launcher checks `nvidia-smi`, selects only GPUs that satisfy your free-memory and utilization rules, sets `CUDA_VISIBLE_DEVICES`, and launches `train.py`.

Training uses `torch.nn.DataParallel` when more than one GPU is selected.

## Current server example

Your server showed:

```text
GPU 0: used by python, about 9.6 GB used, 100% utilization
GPU 1: almost empty
GPU 2: busy, about 32.5 GB used, 99% utilization
```

With the default safe rules, the launcher will probably select only GPU 1. It will skip GPU 0 and GPU 2 because their utilization is high.

## Dry run first

Run this before training:

```bash
python scripts/launch_train_auto_gpus.py \
  --max-gpus 3 \
  --min-gpus 1 \
  --min-free-mb 30000 \
  --max-utilization 30 \
  --dry-run \
  -- python train.py \
    --data-root ./BDC2026 \
    --output-dir ./outputs_dinov3_lora_multigpu \
    --image-size 224 \
    --epochs 20 \
    --batch-size 12 \
    --valid-batch-size 24 \
    --grad-accum 2 \
    --use-class-weights
```

It will print selected GPUs and the final launch command.

## Train with available GPUs

```bash
python scripts/launch_train_auto_gpus.py \
  --max-gpus 3 \
  --min-gpus 1 \
  --min-free-mb 30000 \
  --max-utilization 30 \
  -- python train.py \
    --data-root ./BDC2026 \
    --output-dir ./outputs_dinov3_lora_multigpu \
    --image-size 224 \
    --epochs 20 \
    --batch-size 12 \
    --valid-batch-size 24 \
    --grad-accum 2 \
    --use-class-weights \
    --scheduler plateau \
    --early-stopping-patience 6
```

If only one GPU is available, it trains on one GPU. If two or three GPUs are available, the launcher adds `--multi-gpu` automatically.

## Wait until more GPUs are free

Use this if you want at least 2 GPUs before starting:

```bash
python scripts/launch_train_auto_gpus.py \
  --max-gpus 3 \
  --min-gpus 2 \
  --min-free-mb 30000 \
  --max-utilization 30 \
  --wait \
  --poll-seconds 60 \
  -- python train.py \
    --data-root ./BDC2026 \
    --output-dir ./outputs_dinov3_lora_multigpu \
    --image-size 224 \
    --epochs 20 \
    --batch-size 12 \
    --valid-batch-size 24 \
    --grad-accum 2 \
    --use-class-weights
```

## Force only specific GPUs

Use only GPU 1:

```bash
python scripts/launch_train_auto_gpus.py \
  --include 1 \
  --max-gpus 1 \
  -- python train.py \
    --data-root ./BDC2026 \
    --output-dir ./outputs_dinov3_lora_gpu1 \
    --image-size 224 \
    --epochs 20 \
    --batch-size 4 \
    --valid-batch-size 8 \
    --grad-accum 4 \
    --use-class-weights
```

Use GPUs 0 and 1 only if they are available:

```bash
python scripts/launch_train_auto_gpus.py \
  --include 0,1 \
  --max-gpus 2 \
  --min-free-mb 30000 \
  --max-utilization 30 \
  -- python train.py \
    --data-root ./BDC2026 \
    --output-dir ./outputs_dinov3_lora_gpu01 \
    --image-size 224 \
    --epochs 20 \
    --batch-size 8 \
    --valid-batch-size 16 \
    --grad-accum 2 \
    --use-class-weights
```

Skip GPU 2:

```bash
python scripts/launch_train_auto_gpus.py \
  --exclude 2 \
  --max-gpus 3 \
  --min-free-mb 30000 \
  --max-utilization 30 \
  -- python train.py \
    --data-root ./BDC2026 \
    --output-dir ./outputs_dinov3_lora_no_gpu2 \
    --image-size 224 \
    --epochs 20 \
    --batch-size 8 \
    --valid-batch-size 16 \
    --grad-accum 2 \
    --use-class-weights
```

## Manual multi-GPU command

If you already know which GPUs are free:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train.py \
  --multi-gpu \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora_multigpu \
  --image-size 224 \
  --epochs 20 \
  --batch-size 12 \
  --valid-batch-size 24 \
  --grad-accum 2 \
  --use-class-weights
```

## Notes

- `--batch-size` is the total batch size across all selected GPUs.
- For 3 GPUs, `--batch-size 12` means about 4 images per GPU.
- For 2 GPUs, `--batch-size 8` means about 4 images per GPU.
- DataParallel is simpler than DistributedDataParallel but not always perfectly linear in speedup.
- The checkpoints are saved without the DataParallel `module.` prefix, so prediction still works normally with a single GPU.
