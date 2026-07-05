import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import gdown
from dotenv import load_dotenv


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("BDC2026_ZIP_URL"),
        help="Google Drive URL for a single BDC2026.zip file. Can also be set with BDC2026_ZIP_URL in .env.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(os.environ.get("BDC2026_ZIP_PATH", "./BDC2026.zip")),
        help="Where to save the downloaded ZIP file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("BDC2026_DATA_ROOT", "./BDC2026")),
        help="Where to extract the dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output folder before extracting.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and only extract/check an existing ZIP file.",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip dataset integrity check after extraction.",
    )
    return parser.parse_args()


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
            "The ZIP may have the wrong folder structure or may be incomplete."
        )


def find_data_root(output: Path) -> Path:
    candidates = list(output.rglob("submission.csv"))
    if len(candidates) == 0:
        return output
    return candidates[0].parent


def download_zip(url: str, zip_path: Path):
    if not url:
        raise ValueError("Missing ZIP URL. Pass --url or set BDC2026_ZIP_URL in .env.")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading ZIP from:", url)
    print("Saving ZIP to:", zip_path)

    gdown.download(
        url=url,
        output=str(zip_path),
        quiet=False,
        fuzzy=True,
        resume=True,
    )

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        raise FileNotFoundError(f"ZIP download failed or empty file: {zip_path}")


def extract_zip(zip_path: Path, output: Path, force: bool):
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if force and output.exists():
        print("Removing existing output folder:", output)
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    print("Extracting:", zip_path)
    print("To:", output)

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP integrity test failed at file: {bad_file}")
        zf.extractall(output)


def main():
    args = parse_args()
    args.zip_path = args.zip_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.skip_download:
        download_zip(args.url, args.zip_path)
    else:
        print("Skipping download. Using existing ZIP:", args.zip_path)

    extract_zip(args.zip_path, args.output, force=args.force)

    data_root = find_data_root(args.output)
    print("\nDetected DATA_ROOT:", data_root)
    print("Expected contents:")
    print("  train/")
    print("  test/")
    print("  submission.csv")

    if not args.no_check:
        run_integrity_check(data_root)

    print("\nDataset ZIP pipeline complete.")
    print("Use this path for training:")
    print(f"python train.py --data-root {data_root} ...")


if __name__ == "__main__":
    main()
