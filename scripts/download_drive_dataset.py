import argparse
import os
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
    return parser.parse_args()


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

    gdown.download_folder(
        url=args.url,
        output=str(args.output),
        quiet=False,
        use_cookies=False,
    )

    candidates = list(args.output.rglob("submission.csv"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            "submission.csv not found. The Drive folder may be private or the structure changed. "
            "If using Colab, try mounting Google Drive and adding the shared folder as a shortcut."
        )

    template_path = candidates[0]
    data_root = template_path.parent
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train folder: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test folder: {test_dir}")

    print("Dataset ready.")
    print("DATA_ROOT:", data_root)
    print("TRAIN_DIR:", train_dir)
    print("TEST_DIR:", test_dir)
    print("TEMPLATE_PATH:", template_path)
    print("\nUse this path for training:")
    print(f"python train.py --data-root {data_root} ...")


if __name__ == "__main__":
    main()
