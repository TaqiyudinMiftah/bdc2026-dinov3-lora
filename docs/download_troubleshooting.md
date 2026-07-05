# Google Drive Download Troubleshooting

## Filename order

Seeing files like this does not always mean files are missing:

```text
1.jpg
10.jpg
100.jpg
1000.jpg
```

Many file explorers sort filenames alphabetically. With alphabetical sorting, `2.jpg` appears after many files that start with `1`.

Use numeric sorting in the terminal:

```bash
ls -1v BDC2026/test | head -30
```

## Check whether the dataset is complete

Run:

```bash
python scripts/check_dataset_integrity.py --data-root ./BDC2026 --write-report
```

Expected counts:

```text
train/0_Recyclable: 9999
train/1_Electronic: 3961
train/2_Organic: 12567
test: 1458
```

The checker reports missing test IDs and writes:

```text
dataset_integrity_report.csv
```

Do not train until the checker prints:

```text
Dataset integrity status: OK
```

## Best fallback: download one ZIP file

If `gdown` folder download fails on individual files, create or ask for a single file:

```text
BDC2026.zip
```

Upload that ZIP to Google Drive and share it as viewer-accessible by link. Then run:

```bash
python scripts/download_drive_zip.py \
  --url "https://drive.google.com/file/d/<ZIP_FILE_ID>/view?usp=sharing" \
  --zip-path ./BDC2026.zip \
  --output ./BDC2026 \
  --force
```

More details:

```bash
cat docs/zip_download.md
```

## Local Linux folder download command

On a local Linux server, do not use `/content/BDC2026`. That path is for Google Colab only.

Use:

```bash
python scripts/download_drive_dataset.py \
  --url "https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7" \
  --output ./BDC2026
```

Or put this in `.env`:

```bash
BDC2026_DRIVE_URL=https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7
BDC2026_DATA_ROOT=./BDC2026
```

Then run:

```bash
python scripts/download_drive_dataset.py
```

## If gdown folder download stops before finishing

The download may stop if Google Drive blocks one file, the folder permissions are not fully public, or Google temporarily rate-limits anonymous downloads.

First pull the latest downloader:

```bash
git pull
```

Then rerun:

```bash
python scripts/download_drive_dataset.py \
  --url "https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7" \
  --output ./BDC2026
```

Then verify:

```bash
python scripts/check_dataset_integrity.py --data-root ./BDC2026 --write-report
```

If files are still missing and you cannot get a ZIP file, use authenticated Google Drive download with rclone:

```bash
cat docs/rclone_download.md
```

Quick rclone command after configuring the `gdrive` remote:

```bash
chmod +x scripts/download_with_rclone.sh
BDC2026_DRIVE_FOLDER_ID=1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7 \
BDC2026_DATA_ROOT=./BDC2026 \
REMOTE_NAME=gdrive \
./scripts/download_with_rclone.sh
```

If even rclone cannot access the file, open Google Drive and make sure the folder and files are accessible to the Google account used by rclone.
