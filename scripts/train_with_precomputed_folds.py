import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train as base_train
from bdc2026.dataset import build_train_df, get_transforms
from bdc2026.utils import get_device, macro_f1_and_report, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINOv3 LoRA with a precomputed fold CSV."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs_dinov3_lora"))
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--valid-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--class-weight-mode", type=str, default="inverse")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-weight-mode", type=str, default="sqrt_inverse")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def make_path_key(path_value, class_name=None):
    path = Path(str(path_value))
    known = {"0_Recyclable", "1_Electronic", "2_Organic"}
    for part in path.parts:
        if part in known:
            return f"{part}/{path.name}"
    if class_name in known:
        return f"{class_name}/{path.name}"
    return path.name


def load_fold_map(fold_csv: Path):
    folds = pd.read_csv(fold_csv)
    if "fold" not in folds.columns:
        raise ValueError(f"Missing 'fold' column in {fold_csv}")

    if "path_key" in folds.columns:
        folds["_path_key"] = folds["path_key"].astype(str)
    elif "relative_path" in folds.columns:
        folds["_path_key"] = [
            make_path_key(value, row.get("class_name"))
            for value, (_, row) in zip(folds["relative_path"], folds.iterrows())
        ]
    elif "path" in folds.columns:
        folds["_path_key"] = [
            make_path_key(value, row.get("class_name"))
            for value, (_, row) in zip(folds["path"], folds.iterrows())
        ]
    else:
        raise ValueError("Fold CSV needs one of: path_key, relative_path, or path")

    duplicated = folds[folds["_path_key"].duplicated(keep=False)]
    if len(duplicated):
        raise ValueError(
            "Fold CSV has duplicate path keys:\n"
            + duplicated[["_path_key", "fold"]].head(20).to_string(index=False)
        )
    return folds.set_index("_path_key")


def main():
    args = parse_args()
    cfg = base_train.cfg_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    device = get_device()
    print("Device:", device)
    print("Precomputed fold CSV:", args.fold_csv)

    train_tfms, valid_tfms, _, _ = get_transforms(
        cfg.model_name, cfg.image_size, cfg.hf_token
    )
    train_df = build_train_df(cfg.train_dir, cfg.label2id, cfg.seed)
    train_df["path_key"] = [
        make_path_key(path, class_name)
        for path, class_name in zip(train_df["path"], train_df["class_name"])
    ]

    fold_map = load_fold_map(args.fold_csv)
    train_df["fold"] = train_df["path_key"].map(fold_map["fold"])
    missing = train_df[train_df["fold"].isna()]
    if len(missing):
        raise ValueError(
            f"{len(missing)} training images are missing from the fold CSV. Examples:\n"
            + missing[["path_key", "path"]].head(20).to_string(index=False)
        )

    train_df["fold"] = train_df["fold"].astype(int)
    found_folds = sorted(train_df["fold"].unique().tolist())
    expected_folds = list(range(cfg.n_splits))
    if found_folds != expected_folds:
        raise ValueError(f"Expected folds {expected_folds}, found {found_folds}")

    if "label" in fold_map.columns:
        expected_labels = train_df["path_key"].map(fold_map["label"]).astype(int)
        mismatch = train_df[expected_labels.to_numpy() != train_df["label"].to_numpy()]
        if len(mismatch):
            raise ValueError(
                "Label mismatch between clean dataset and fold CSV. Examples:\n"
                + mismatch[["path_key", "label"]].head(20).to_string(index=False)
            )

    train_df["original_index"] = np.arange(len(train_df))
    train_df.to_csv(cfg.output_dir / "train_folds.csv", index=False)

    fold_counts = train_df.groupby(["fold", "class_name"]).size().unstack(fill_value=0)
    print("Fold class counts:\n", fold_counts)

    oof_probs = np.zeros((len(train_df), cfg.num_classes), dtype=np.float32)
    oof_labels = train_df["label"].values.copy()
    fold_scores = []

    for fold in expected_folds:
        val_indices, val_probs, _, best_f1 = base_train.run_fold(
            fold, train_df, train_tfms, valid_tfms, cfg, device
        )
        oof_probs[val_indices] = val_probs
        fold_scores.append(best_f1)

    macro_f1, per_class_f1, report, cm = macro_f1_and_report(
        oof_labels, oof_probs, cfg.id2label
    )
    print("\n========== OOF RESULT ==========")
    print("Fold scores:", fold_scores)
    print("Mean fold F1:", np.mean(fold_scores))
    print("OOF Macro F1:", macro_f1)
    print("OOF per-class F1:", per_class_f1)
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", cm)

    np.save(cfg.output_dir / "oof_probs.npy", oof_probs)
    oof_df = train_df.copy()
    oof_df["oof_pred"] = oof_probs.argmax(axis=1)
    for class_id in range(cfg.num_classes):
        oof_df[f"prob_{cfg.id2label[class_id]}"] = oof_probs[:, class_id]
    oof_df.to_csv(cfg.output_dir / "oof_predictions.csv", index=False)

    cfg_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in cfg.__dict__.items()
        if not key.startswith("_") and key != "hf_token"
    }
    cfg_dict["fold_csv"] = str(args.fold_csv.resolve())
    with open(cfg.output_dir / "config_used.json", "w", encoding="utf-8") as handle:
        json.dump(cfg_dict, handle, indent=2)


if __name__ == "__main__":
    main()
