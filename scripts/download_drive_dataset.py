import argparse
from pathlib import Path

import gdown


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="Google Drive folder URL")
    parser.add_argument("--output", type=Path, default=Path("/content/BDC2026"))
    return parser.parse_args()


def main():
    args = parse_args()
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


if __name__ == "__main__":
    main()
