import argparse
import re
from pathlib import Path

import pandas as pd


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_TRAIN_COUNTS = {
    "0_Recyclable": 9999,
    "1_Electronic": 3961,
    "2_Organic": 12567,
}
EXPECTED_TEST_COUNT = 1458


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("./BDC2026"))
    parser.add_argument("--expected-test-count", type=int, default=EXPECTED_TEST_COUNT)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report-path", type=Path, default=Path("./dataset_integrity_report.csv"))
    return parser.parse_args()


def numeric_stem(path: Path):
    match = re.fullmatch(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: (numeric_stem(p) is None, numeric_stem(p) or p.stem, p.name),
    )


def check_test(test_dir: Path, expected_count: int):
    files = image_files(test_dir)
    ids = [numeric_stem(p) for p in files]
    numeric_ids = sorted([i for i in ids if i is not None])
    expected_ids = set(range(1, expected_count + 1))
    actual_ids = set(numeric_ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    duplicate_ids = sorted({x for x in numeric_ids if numeric_ids.count(x) > 1})

    print("\n========== TEST CHECK ==========")
    print("Test folder:", test_dir)
    print("Image files found:", len(files))
    print("Expected test images:", expected_count)
    print("First 30 files using numeric sort:")
    print([p.name for p in files[:30]])
    print("Missing numeric ids:", len(missing))
    if missing:
        print("First missing ids:", missing[:50])
    print("Extra numeric ids:", len(extra))
    if extra:
        print("First extra ids:", extra[:50])
    print("Duplicate numeric ids:", len(duplicate_ids))
    if duplicate_ids:
        print("Duplicate ids:", duplicate_ids[:50])

    rows = []
    for missing_id in missing:
        rows.append({"split": "test", "issue": "missing_id", "id": missing_id, "path": str(test_dir / f"{missing_id}.jpg")})
    for extra_id in extra:
        rows.append({"split": "test", "issue": "extra_id", "id": extra_id, "path": ""})
    for dup_id in duplicate_ids:
        rows.append({"split": "test", "issue": "duplicate_id", "id": dup_id, "path": ""})
    return rows


def check_train(train_dir: Path):
    rows = []
    print("\n========== TRAIN CHECK ==========")
    print("Train folder:", train_dir)
    total = 0
    for class_name, expected_count in EXPECTED_TRAIN_COUNTS.items():
        class_dir = train_dir / class_name
        files = image_files(class_dir)
        total += len(files)
        status = "OK" if len(files) == expected_count else "MISMATCH"
        print(f"{class_name}: found={len(files)} expected={expected_count} [{status}]")
        if len(files) != expected_count:
            rows.append({
                "split": "train",
                "issue": "class_count_mismatch",
                "id": "",
                "path": str(class_dir),
                "found": len(files),
                "expected": expected_count,
            })
    print("Train total found:", total)
    print("Train total expected:", sum(EXPECTED_TRAIN_COUNTS.values()))
    return rows


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    template_path = data_root / "submission.csv"

    print("Dataset root:", data_root)
    print("submission.csv exists:", template_path.exists())
    print("train exists:", train_dir.exists())
    print("test exists:", test_dir.exists())

    report_rows = []
    report_rows.extend(check_train(train_dir))
    report_rows.extend(check_test(test_dir, args.expected_test_count))

    if template_path.exists():
        template = pd.read_csv(template_path)
        print("\n========== SUBMISSION TEMPLATE ==========")
        print("Rows:", len(template))
        print("Columns:", list(template.columns))
        if len(template) != args.expected_test_count:
            report_rows.append({
                "split": "submission",
                "issue": "template_row_mismatch",
                "id": "",
                "path": str(template_path),
                "found": len(template),
                "expected": args.expected_test_count,
            })

    if args.write_report:
        report = pd.DataFrame(report_rows)
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report_path, index=False)
        print("\nSaved report:", args.report_path)

    if report_rows:
        print("\nDataset integrity status: ISSUES FOUND")
        raise SystemExit(1)

    print("\nDataset integrity status: OK")


if __name__ == "__main__":
    main()
