import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path

import gdown
from dotenv import load_dotenv


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("BDC2026_DRIVE_URL"),
        help="Google Drive folder URL. Can also be set with BDC2026_DRIVE_URL in .env.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("BDC2026_DATA_ROOT", "./BDC2026")),
        help="Dataset output directory. Use ./BDC2026 on local/Linux servers, /content/BDC2026 on Colab.",
    )
    parser.add_argument(
        "--use-cookies",
        action="store_true",
        help="Use browser cookies if available. Useful for private Drive files on local machines.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately on the first file download error. By default, gdown keeps going when supported.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip dataset integrity check after download.",
    )
    return parser.parse_args()


def call_gdown_download_folder(args):
    kwargs = {
        "url": args.url,
        "output": str(args.output),
        "quiet": False,
    }

    signature = inspect.signature(gdown.download_folder)
    if "use_cookies" in signature.parameters:
        kwargs["use_cookies"] = args.use_cookies
    if "remaining_ok" in signature.parameters:
        # False means: keep downloading remaining files even if one file fails.
        # This helps large Drive folders where one file temporarily blocks gdown.
        kwargs["remaining_ok"] = not args.strict

    print("gdown.download_folder kwargs:", {k: v for k, v in kwargs.items() if k != "url"})
    return gdown.download_folder(**kwargs)


def find_data_root(output: Path) -> Path:
    candidates = list(output.rglob("submission.csv"))
    if len(candidates) == 0:
        return output
    return candidates[0].parent


def run_integrity_check(data_root: Path):
    checker = Path(__file__).resolve().parent / "check_dataset_integrity.py"
    if not checker.exists():
        print("Integrity checker not found, skipping:", checker)
        return

    print("\nRunning dataset integrity check...")
    result = subprocess.run(
        [sys.executable, str(checker), "--data-root", str(data_root), "--write-report"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Dataset integrity check found issues. See dataset_integrity_report.csv. "
            "Usually this means the Google Drive download stopped before all files were retrieved."
        )


def main():
    args = parse_args()

    if not args.url:
        raise ValueError(
            "Missing Drive folder URL. Pass --url or set BDC2026_DRIVE_URL in your .env file."
        )

    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Downloading from:", args.url)
    print("Output folder:", args.output)
    print("Tip: if download fails because a file is not public, set the folder and files to 'Anyone with the link: Viewer'.")
    print("Tip: if it fails after many files, rerun the same command; existing files are usually reused/skipped by gdown.")
    print("Tip: if gdown still fails, use docs/rclone_download.md for authenticated Drive download.")

    try:
        call_gdown_download_folder(args)
    except Exception as e:
        print("\nDownload failed before all files were retrieved.")
        print("Reason:", repr(e))
        print("\nWhat to try next:")
        print("1. Pull latest repo changes: git pull")
        print("2. Rerun the same command. The downloader now keeps going when your gdown version supports it.")
        print("3. Run: python scripts/check_dataset_integrity.py --data-root", args.output, "--write-report")
        print("4. If files are still missing, use authenticated rclone download: docs/rclone_download.md")
        raise

    data_root = find_data_root(args.output)
    template_path = data_root / "submission.csv"
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not template_path.exists():
        raise FileNotFoundError(
            "submission.csv not found. The Drive folder may be private or the structure changed."
        )
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train folder: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    print("Dataset ready.")
    print("DATA_ROOT:", data_root)
    print("TRAIN_DIR:", train_dir)
    print("TEST_DIR:", test_dir)
    print("TEMPLATE_PATH:", template_path)

    if not args.no_check:
        run_integrity_check(data_root)

    print("\nUse this path for training:")
    print(f"python train.py --data-root {data_root} ...")


if __name__ == "__main__":
    main()
