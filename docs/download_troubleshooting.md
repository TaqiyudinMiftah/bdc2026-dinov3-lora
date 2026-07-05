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

## Local Linux download command

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

## If gdown stops before finishing

The download may stop if Google Drive blocks one file or the folder permissions are not fully public.

Try:

```bash
python scripts/download_drive_dataset.py \
  --url "https://drive.google.com/drive/folders/1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7" \
  --output ./BDC2026
```

Then verify again:

```bash
python scripts/check_dataset_integrity.py --data-root ./BDC2026 --write-report
```

If it still fails, open Google Drive and make sure the folder and files are shared as viewer-accessible by link.
