# Authenticated Google Drive Download with rclone

Use this when `gdown` fails on one or more files even though you can open the Drive folder in your browser.

`rclone` uses your own Google account login, so it is usually more reliable for large Google Drive folders than anonymous public-link downloads.

## 1. Install rclone

### Option A: no sudo access

Use the local installer included in this repo:

```bash
git pull
chmod +x scripts/install_rclone_user.sh
./scripts/install_rclone_user.sh
export PATH="$HOME/bin:$PATH"
rclone version
```

This installs the `rclone` binary to:

```text
~/bin/rclone
```

If `rclone` works only in the current terminal, reload your shell profile:

```bash
source ~/.bashrc
```

or manually run:

```bash
export PATH="$HOME/bin:$PATH"
```

### Option B: sudo access

If you have sudo access:

```bash
curl https://rclone.org/install.sh | sudo bash
```

Check:

```bash
rclone version
```

## 2. Configure a Google Drive remote

Run this on the server:

```bash
rclone config
```

Choose:

```text
n) New remote
name: gdrive
Storage: Google Drive
client_id: leave blank
client_secret: leave blank
scope: drive.readonly
root_folder_id: leave blank
service_account_file: leave blank
Edit advanced config: n
```

### If the server has a browser

Choose:

```text
Use auto config: y
```

A browser login will open. Login with the Google account that can access the BDC2026 Drive folder.

### If the server does not have a browser

Choose:

```text
Use auto config: n
```

The server will ask for `config_token` and show a command like this:

```bash
rclone authorize "drive" "eyJzY29wZSI6ImRyaXZlLnJlYWRvbmx5In0"
```

The `config_token` is the JSON output produced by running that command on another machine that has both `rclone` and a browser, for example your laptop.

On your laptop, install rclone and run the exact command shown by the server:

```bash
rclone authorize "drive" "eyJzY29wZSI6ImRyaXZlLnJlYWRvbmx5In0"
```

Your browser will open. Login with the Google account that can access the BDC2026 Drive folder. After login, the laptop terminal will print a JSON block similar to this:

```json
{"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}
```

Copy the entire JSON block and paste it into the server prompt:

```text
config_token> {"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}
```

Do not paste your Hugging Face token, GitHub token, or Google password into `config_token`.

Verify the remote:

```bash
rclone lsd gdrive:
```

## 3. Download the BDC2026 folder by folder ID

The current folder ID is:

```text
1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7
```

Run:

```bash
chmod +x scripts/download_with_rclone.sh
BDC2026_DRIVE_FOLDER_ID=1Wkn2KazyHsSqBQnONkI98SnN--k3gAT7 \
BDC2026_DATA_ROOT=./BDC2026 \
REMOTE_NAME=gdrive \
./scripts/download_with_rclone.sh
```

The script copies the Drive folder to `./BDC2026` and runs:

```bash
python scripts/check_dataset_integrity.py --data-root ./BDC2026 --write-report
```

## 4. Train only after integrity check is OK

Expected status:

```text
Dataset integrity status: OK
```

Then train:

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
