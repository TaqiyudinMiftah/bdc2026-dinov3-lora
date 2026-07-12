import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm


ID2NAME = {
    0: "0_Recyclable",
    1: "1_Electronic",
    2: "2_Organic",
}
DISPLAY_NAMES = {
    0: "Recyclable",
    1: "Electronic",
    2: "Organic",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze wrong OOF predictions from a completed training run.")
    parser.add_argument("--data-root", type=Path, default=Path("./BDC2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs_dinov3_lora_384"))
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--high-confidence", type=float, default=0.85)
    parser.add_argument("--very-high-confidence", type=float, default=0.95)
    parser.add_argument("--low-margin", type=float, default=0.15)
    parser.add_argument("--compute-image-metadata", action="store_true")
    parser.add_argument("--max-metadata-images", type=int, default=0, help="0 means all images.")
    parser.add_argument("--eda-dirs", nargs="*", type=Path, default=[Path("./eda_outputs"), Path("./eda_outputs_dino")])
    return parser.parse_args()


def probability_columns(df: pd.DataFrame) -> list[str]:
    expected = [f"prob_{ID2NAME[i]}" for i in range(len(ID2NAME))]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        fallback = sorted([c for c in df.columns if c.startswith("prob_")])
        if len(fallback) != len(ID2NAME):
            raise ValueError(
                f"Missing probability columns: {missing}. Available probability columns: {fallback}"
            )
        return fallback
    return expected


def resolve_image_path(path_value, class_name: str, data_root: Path) -> Path:
    original = Path(str(path_value))
    if original.exists():
        return original

    candidates = [
        data_root / "train" / str(class_name) / original.name,
        data_root / str(class_name) / original.name,
        data_root / "train" / original.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_oof(output_dir: Path, data_root: Path) -> pd.DataFrame:
    oof_path = output_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF predictions not found: {oof_path}. Expected a completed training output directory."
        )

    df = pd.read_csv(oof_path)
    required = {"path", "label", "oof_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {oof_path}: {missing}")

    if "class_name" not in df.columns:
        df["class_name"] = df["label"].map(ID2NAME)
    if "fold" not in df.columns:
        df["fold"] = -1

    df["resolved_path"] = [
        str(resolve_image_path(path, class_name, data_root))
        for path, class_name in zip(df["path"], df["class_name"])
    ]
    return df


def enrich_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prob_cols = probability_columns(df)
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)

    sorted_probs = np.sort(probs, axis=1)
    labels = df["label"].astype(int).to_numpy()
    preds = df["oof_pred"].astype(int).to_numpy()

    df["correct"] = labels == preds
    df["true_name"] = df["label"].map(DISPLAY_NAMES)
    df["pred_name"] = df["oof_pred"].map(DISPLAY_NAMES)
    df["confidence"] = probs.max(axis=1)
    df["second_probability"] = sorted_probs[:, -2]
    df["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    df["entropy"] = -(probs * np.log(probs)).sum(axis=1)
    df["normalized_entropy"] = df["entropy"] / math.log(probs.shape[1])
    df["true_probability"] = probs[np.arange(len(df)), labels]
    df["pred_probability"] = probs[np.arange(len(df)), preds]
    df["confidence_error"] = np.where(df["correct"], 1.0 - df["confidence"], df["confidence"])
    df["confusion_pair"] = df["true_name"] + " → " + df["pred_name"]
    return df


def build_metrics(df: pd.DataFrame):
    labels = sorted(ID2NAME)
    y_true = df["label"].astype(int)
    y_pred = df["oof_pred"].astype(int)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[DISPLAY_NAMES[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    class_metrics = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{DISPLAY_NAMES[i]}" for i in labels],
        columns=[f"pred_{DISPLAY_NAMES[i]}" for i in labels],
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return macro_f1, class_metrics, cm_df


def plot_confusion_matrix(cm_df: pd.DataFrame, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm_df.to_numpy(), interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(cm_df.columns)), [c.replace("pred_", "") for c in cm_df.columns], rotation=30)
    ax.set_yticks(range(len(cm_df.index)), [r.replace("true_", "") for r in cm_df.index])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("OOF confusion matrix")

    matrix = cm_df.to_numpy()
    threshold = matrix.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    return fig


def confusion_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    wrong = df[~df["correct"]]
    return (
        wrong.groupby(["label", "oof_pred", "true_name", "pred_name"], as_index=False)
        .agg(
            count=("path", "size"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_true_probability=("true_probability", "mean"),
            mean_margin=("margin", "mean"),
        )
        .sort_values(["count", "mean_confidence"], ascending=[False, False])
        .reset_index(drop=True)
    )


def load_histories(output_dir: Path) -> pd.DataFrame:
    files = sorted(output_dir.glob("fold*_history.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "fold" not in frame.columns:
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            frame["fold"] = int(digits) if digits else -1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_training_curves(history: pd.DataFrame, analysis_dir: Path):
    if history.empty:
        print("No fold history CSV files found. Skipping training curves.")
        return []

    figures = []
    for metric, title, filename in [
        ("macro_f1", "Validation Macro-F1 by fold", "training_macro_f1.png"),
        ("valid_loss", "Validation loss by fold", "training_valid_loss.png"),
        ("train_loss", "Training loss by fold", "training_train_loss.png"),
    ]:
        if metric not in history.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for fold, group in history.groupby("fold"):
            group = group.sort_values("epoch")
            ax.plot(group["epoch"], group[metric], marker="o", label=f"fold {fold}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir / filename, dpi=180, bbox_inches="tight")
        figures.append(fig)
    return figures


def show_image_grid(
    rows: pd.DataFrame,
    n: int = 20,
    cols: int = 5,
    sort_by: str | None = None,
    ascending: bool = False,
    title: str = "Image examples",
    save_path: Path | None = None,
):
    if sort_by is not None and sort_by in rows.columns:
        rows = rows.sort_values(sort_by, ascending=ascending)
    rows = rows.head(n).reset_index(drop=True)
    if rows.empty:
        print(f"No rows available for: {title}")
        return None

    grid_rows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3.4, grid_rows * 3.7))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, rows.iterrows()):
        path = Path(row["resolved_path"])
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                ax.imshow(image)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Cannot read\n{path.name}\n{exc}", ha="center", va="center")

        ax.set_title(
            f"T: {row['true_name']} | P: {row['pred_name']}\n"
            f"conf={row['confidence']:.3f} true_p={row['true_probability']:.3f}\n"
            f"fold={row.get('fold', '?')} | {path.name}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def image_metadata(path: str) -> dict:
    try:
        file_path = Path(path)
        with Image.open(file_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            thumbnail = image.copy()
            thumbnail.thumbnail((256, 256))
            gray = np.asarray(thumbnail.convert("L"), dtype=np.float32)

        gradient_x = np.diff(gray, axis=1)
        gradient_y = np.diff(gray, axis=0)
        sharpness = float(gradient_x.var() + gradient_y.var())
        return {
            "image_valid": True,
            "width": width,
            "height": height,
            "min_side": min(width, height),
            "aspect_ratio": width / max(height, 1),
            "file_size_bytes": file_path.stat().st_size,
            "brightness": float(gray.mean()),
            "contrast": float(gray.std()),
            "sharpness": sharpness,
            "image_error": "",
        }
    except Exception as exc:
        return {
            "image_valid": False,
            "width": np.nan,
            "height": np.nan,
            "min_side": np.nan,
            "aspect_ratio": np.nan,
            "file_size_bytes": np.nan,
            "brightness": np.nan,
            "contrast": np.nan,
            "sharpness": np.nan,
            "image_error": repr(exc),
        }


def add_image_metadata(df: pd.DataFrame, max_images: int = 0) -> pd.DataFrame:
    result = df.copy()
    indices = result.index.to_list()
    if max_images > 0:
        indices = indices[:max_images]

    metadata_rows = {}
    for index in tqdm(indices, desc="Image metadata"):
        metadata_rows[index] = image_metadata(result.at[index, "resolved_path"])

    metadata_df = pd.DataFrame.from_dict(metadata_rows, orient="index")
    for column in metadata_df.columns:
        result.loc[metadata_df.index, column] = metadata_df[column]
    return result


def normalize_path(value) -> str:
    return str(Path(str(value)).expanduser())


def collect_duplicate_flags(eda_dirs: Iterable[Path]):
    duplicate_paths = set()
    cross_label_paths = set()
    duplicate_sources = {}

    def add_path(path_value, source, cross_label=False):
        path = normalize_path(path_value)
        duplicate_paths.add(path)
        duplicate_sources.setdefault(path, set()).add(source)
        if cross_label:
            cross_label_paths.add(path)

    for directory in eda_dirs:
        if not directory.exists():
            continue

        exact_path = directory / "exact_duplicate_groups.csv"
        if exact_path.exists():
            exact = pd.read_csv(exact_path)
            for _, row in exact.iterrows():
                add_path(row["path"], "exact_md5", bool(row.get("cross_label", False)))

        for filename, source in [
            ("phash_duplicate_pairs.csv", "phash"),
            ("dino_duplicate_pairs.csv", "dino_embedding"),
        ]:
            pair_path = directory / filename
            if not pair_path.exists():
                continue
            pairs = pd.read_csv(pair_path)
            for _, row in pairs.iterrows():
                cross = bool(row.get("cross_label", False))
                add_path(row["path_a"], source, cross)
                add_path(row["path_b"], source, cross)

    return duplicate_paths, cross_label_paths, duplicate_sources


def add_duplicate_flags(df: pd.DataFrame, eda_dirs: Iterable[Path]) -> pd.DataFrame:
    result = df.copy()
    duplicate_paths, cross_label_paths, duplicate_sources = collect_duplicate_flags(eda_dirs)

    def lookup_sources(row):
        candidates = [normalize_path(row["path"]), normalize_path(row["resolved_path"])]
        sources = set()
        for candidate in candidates:
            sources.update(duplicate_sources.get(candidate, set()))
        return ",".join(sorted(sources))

    result["duplicate_sources"] = result.apply(lookup_sources, axis=1)
    result["is_duplicate_candidate"] = result["duplicate_sources"].str.len() > 0
    result["is_cross_label_duplicate"] = result.apply(
        lambda row: normalize_path(row["path"]) in cross_label_paths
        or normalize_path(row["resolved_path"]) in cross_label_paths,
        axis=1,
    )
    return result


def add_error_categories(
    df: pd.DataFrame,
    high_confidence: float = 0.85,
    very_high_confidence: float = 0.95,
    low_margin: float = 0.15,
) -> pd.DataFrame:
    result = df.copy()

    def categorize(row):
        if row["correct"]:
            return "correct"
        if bool(row.get("is_cross_label_duplicate", False)):
            return "cross_label_duplicate_review"
        if row["confidence"] >= very_high_confidence and row["true_probability"] <= 0.05:
            return "possible_label_noise"
        if bool(row.get("is_duplicate_candidate", False)):
            return "duplicate_or_near_duplicate"

        if bool(row.get("image_valid", True)) is False:
            return "corrupt_or_unreadable"

        quality_flags = []
        if pd.notna(row.get("min_side")) and row.get("min_side") < 100:
            quality_flags.append("small")
        if pd.notna(row.get("aspect_ratio")) and not 0.4 <= row.get("aspect_ratio") <= 2.5:
            quality_flags.append("extreme_aspect")
        if pd.notna(row.get("brightness")) and (row.get("brightness") < 35 or row.get("brightness") > 225):
            quality_flags.append("brightness")
        if pd.notna(row.get("contrast")) and row.get("contrast") < 20:
            quality_flags.append("low_contrast")
        if pd.notna(row.get("sharpness")) and row.get("sharpness") < 80:
            quality_flags.append("blur")
        if quality_flags:
            return "low_visual_quality:" + ",".join(quality_flags)

        if row["margin"] <= low_margin or row["normalized_entropy"] >= 0.80:
            return "intrinsic_ambiguity"
        if row["confidence"] >= high_confidence:
            return "systematic_model_confusion"
        return "general_model_error"

    result["error_category"] = result.apply(categorize, axis=1)
    return result


def error_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    wrong = df[~df["correct"]]
    if wrong.empty:
        return pd.DataFrame(columns=["error_category", "count", "share", "mean_confidence"])
    summary = (
        wrong.groupby("error_category", as_index=False)
        .agg(count=("path", "size"), mean_confidence=("confidence", "mean"), mean_true_probability=("true_probability", "mean"))
        .sort_values("count", ascending=False)
    )
    summary["share"] = summary["count"] / len(wrong)
    return summary


def quality_error_summary(df: pd.DataFrame, feature: str, bins=5) -> pd.DataFrame:
    valid = df.dropna(subset=[feature]).copy()
    if valid.empty:
        return pd.DataFrame()
    try:
        valid["bin"] = pd.qcut(valid[feature], q=bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    return (
        valid.groupby("bin", observed=False)
        .agg(images=("path", "size"), error_rate=("correct", lambda x: 1.0 - x.mean()), mean_confidence=("confidence", "mean"))
        .reset_index()
    )


def export_reports(
    df: pd.DataFrame,
    class_metrics: pd.DataFrame,
    cm_df: pd.DataFrame,
    pair_table: pd.DataFrame,
    category_summary: pd.DataFrame,
    analysis_dir: Path,
    macro_f1: float,
):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    wrong = df[~df["correct"]].sort_values(["confidence", "margin"], ascending=[False, False])
    high_conf_wrong = wrong[wrong["confidence"] >= 0.85]
    uncertain = df.sort_values(["margin", "normalized_entropy"], ascending=[True, False])
    label_review = wrong[
        wrong["error_category"].isin(["possible_label_noise", "cross_label_duplicate_review"])
    ]

    df.to_csv(analysis_dir / "oof_predictions_enriched.csv", index=False)
    wrong.to_csv(analysis_dir / "misclassified_all.csv", index=False)
    high_conf_wrong.to_csv(analysis_dir / "high_confidence_wrong.csv", index=False)
    uncertain.to_csv(analysis_dir / "most_uncertain_predictions.csv", index=False)
    label_review.to_csv(analysis_dir / "label_review_candidates.csv", index=False)
    class_metrics.to_csv(analysis_dir / "per_class_metrics.csv", index=False)
    cm_df.to_csv(analysis_dir / "confusion_matrix.csv")
    pair_table.to_csv(analysis_dir / "confusion_pairs.csv", index=False)
    category_summary.to_csv(analysis_dir / "error_category_summary.csv", index=False)

    summary = {
        "num_images": int(len(df)),
        "num_correct": int(df["correct"].sum()),
        "num_wrong": int((~df["correct"]).sum()),
        "accuracy": float(df["correct"].mean()),
        "macro_f1": float(macro_f1),
        "high_confidence_wrong": int(len(high_conf_wrong)),
        "label_review_candidates": int(len(label_review)),
    }
    with open(analysis_dir / "error_analysis_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    lines = [
        "# OOF Error Analysis Summary",
        "",
        f"- Images: {summary['num_images']}",
        f"- Correct: {summary['num_correct']}",
        f"- Wrong: {summary['num_wrong']}",
        f"- Accuracy: {summary['accuracy']:.5f}",
        f"- Macro-F1: {summary['macro_f1']:.5f}",
        f"- High-confidence wrong predictions: {summary['high_confidence_wrong']}",
        f"- Label-review candidates: {summary['label_review_candidates']}",
        "",
        "## Largest confusion pairs",
        "",
        pair_table.head(10).to_markdown(index=False) if not pair_table.empty else "No errors found.",
        "",
        "## Error categories",
        "",
        category_summary.to_markdown(index=False) if not category_summary.empty else "No errors found.",
    ]
    (analysis_dir / "error_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run_analysis(
    data_root: Path,
    output_dir: Path,
    analysis_dir: Path | None = None,
    compute_metadata: bool = False,
    max_metadata_images: int = 0,
    eda_dirs: Iterable[Path] = (Path("./eda_outputs"), Path("./eda_outputs_dino")),
    high_confidence: float = 0.85,
    very_high_confidence: float = 0.95,
    low_margin: float = 0.15,
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else output_dir / "error_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = load_oof(output_dir, data_root)
    df = enrich_predictions(df)
    df = add_duplicate_flags(df, [Path(p) for p in eda_dirs])
    if compute_metadata:
        df = add_image_metadata(df, max_images=max_metadata_images)
    df = add_error_categories(
        df,
        high_confidence=high_confidence,
        very_high_confidence=very_high_confidence,
        low_margin=low_margin,
    )

    macro_f1, class_metrics, cm_df = build_metrics(df)
    pair_table = confusion_pair_table(df)
    category_summary = error_category_summary(df)
    history = load_histories(output_dir)

    plot_confusion_matrix(cm_df, analysis_dir / "confusion_matrix.png")
    plot_training_curves(history, analysis_dir)

    show_image_grid(
        df[~df["correct"]],
        n=25,
        sort_by="confidence",
        ascending=False,
        title="Highest-confidence wrong predictions",
        save_path=analysis_dir / "high_confidence_wrong_grid.png",
    )
    show_image_grid(
        df,
        n=25,
        sort_by="margin",
        ascending=True,
        title="Most ambiguous OOF predictions",
        save_path=analysis_dir / "most_ambiguous_grid.png",
    )

    summary = export_reports(
        df,
        class_metrics,
        cm_df,
        pair_table,
        category_summary,
        analysis_dir,
        macro_f1,
    )

    return {
        "predictions": df,
        "class_metrics": class_metrics,
        "confusion_matrix": cm_df,
        "confusion_pairs": pair_table,
        "error_categories": category_summary,
        "history": history,
        "summary": summary,
        "analysis_dir": analysis_dir,
    }


def main():
    args = parse_args()
    result = run_analysis(
        data_root=args.data_root,
        output_dir=args.output_dir,
        analysis_dir=args.analysis_dir,
        compute_metadata=args.compute_image_metadata,
        max_metadata_images=args.max_metadata_images,
        eda_dirs=args.eda_dirs,
        high_confidence=args.high_confidence,
        very_high_confidence=args.very_high_confidence,
        low_margin=args.low_margin,
    )
    print(json.dumps(result["summary"], indent=2))
    print("Reports saved to:", result["analysis_dir"])


if __name__ == "__main__":
    main()
