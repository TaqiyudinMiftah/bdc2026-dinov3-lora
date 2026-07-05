#!/usr/bin/env bash
set -euo pipefail

REMOTE_NAME="${REMOTE_NAME:-gdrive}"
FOLDER_ID="${BDC2026_DRIVE_FOLDER_ID:-1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7}"
OUTPUT_DIR="${BDC2026_DATA_ROOT:-./BDC2026}"

mkdir -p "${OUTPUT_DIR}"

echo "Remote: ${REMOTE_NAME}"
echo "Folder ID: ${FOLDER_ID}"
echo "Output: ${OUTPUT_DIR}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is not installed. Install it first:"
  echo "  curl https://rclone.org/install.sh | sudo bash"
  echo "or without sudo, download rclone from https://rclone.org/downloads/"
  exit 1
fi

echo "Copying Google Drive folder with rclone..."
rclone copy \
  "${REMOTE_NAME}:/" \
  "${OUTPUT_DIR}" \
  --drive-root-folder-id "${FOLDER_ID}" \
  --progress \
  --transfers 8 \
  --checkers 16 \
  --drive-acknowledge-abuse \
  --retries 10 \
  --low-level-retries 20

echo "Running integrity check..."
python scripts/check_dataset_integrity.py --data-root "${OUTPUT_DIR}" --write-report

echo "Done. Use this for training:"
echo "python train.py --data-root ${OUTPUT_DIR} ..."
