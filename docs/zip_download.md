# ZIP-based Google Drive Dataset Download

This is the recommended fallback when `gdown.download_folder` fails while downloading thousands of individual files.

## Key idea

`gdown` is more reliable when downloading **one ZIP file** than when downloading many individual images from a Google Drive folder.

However, `gdown` cannot automatically ask Google Drive to zip a folder for you. You need a real ZIP file in Drive, for example:

```text
BDC2026.zip
```

The ZIP should contain:

```text
BDC2026/
├── train/
│   ├── 0_Recyclable/
│   ├── 1_Electronic/
│   └── 2_Organic/
├── test/
└── submission.csv
```

or directly:

```text
train/
test/
submission.csv
```

The script will auto-detect the folder containing `submission.csv`.

## 1. Create the ZIP file

Create or ask the dataset owner to create:

```text
BDC2026.zip
```

Then upload it to Google Drive.

Recommended ZIP structure:

```text
BDC2026.zip
└── BDC2026/
    ├── train/
    ├── test/
    └── submission.csv
```

## 2. Share the ZIP file

Right click `BDC2026.zip` in Google Drive:

```text
Share -> General access -> Anyone with the link -> Viewer
```

Copy the ZIP file link. It should look like:

```text
https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
```

## 3. Download and extract with gdown

Run:

```bash
python scripts/download_drive_zip.py \
  --url "https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing" \
  --zip-path ./BDC2026.zip \
  --output ./BDC2026 \
  --force
```

The script will:

1. download the ZIP file,
2. test the ZIP integrity,
3. extract it,
4. auto-detect the dataset root,
5. run the dataset integrity checker.

## 4. Verify manually

```bash
python scripts/check_dataset_integrity.py --data-root ./BDC2026 --write-report
```

Expected:

```text
Dataset integrity status: OK
```

## 5. Use `.env` instead of command arguments

Add this to `.env`:

```bash
BDC2026_ZIP_URL=https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
BDC2026_ZIP_PATH=./BDC2026.zip
BDC2026_DATA_ROOT=./BDC2026
```

Then run:

```bash
python scripts/download_drive_zip.py --force
```

## 6. Train

```bash
python train.py \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora \
  --image-size 224 \
  --epochs 20 \
  --batch-size 4 \
  --valid-batch-size 8 \
  --grad-accum 4 \
  --use-class-weights
```
