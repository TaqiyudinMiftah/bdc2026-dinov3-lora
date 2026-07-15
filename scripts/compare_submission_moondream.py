#!/usr/bin/env python3
"""Compare a DINOv3 submission with Moondream test pseudo-labels.

The comparison is diagnostic only. Moondream predictions are not ground truth, and
this script deliberately does not create a competition submission.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

LABEL_NAMES = {0: "Recyclable", 1: "Electronic", 2: "Organic"}
CONFIDENCE_PRIORITY = {"high": 3, "medium": 2, "low": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moondream-predictions",
        type=Path,
        default=Path(
            "./leaderboard_pipeline_outputs/06b_moondream_test_comparison/"
            "moondream_predictions/moondream_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--dino-debug",
        type=Path,
        default=None,
        help="Optional test_predictions_debug.csv containing DINO probabilities.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "./leaderboard_pipeline_outputs/06b_moondream_test_comparison/comparison"
        ),
    )
    parser.add_argument("--grid-images", type=int, default=40)
    parser.add_argument("--grid-cols", type=int, default=5)
    return parser.parse_args()


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_path(value) -> str:
    return str(Path(str(value)).expanduser().resolve())


def attach_dino_probabilities(df: pd.DataFrame, debug_path: Path | None) -> pd.DataFrame:
    result = df.copy()
    if debug_path is None:
        return result
    if not debug_path.exists():
        raise FileNotFoundError(debug_path)

    debug = pd.read_csv(debug_path)
    probability_columns = sorted([c for c in debug.columns if c.startswith("prob_")])
    if len(probability_columns) != 3:
        raise ValueError(
            f"Expected three probability columns in {debug_path}, found {probability_columns}"
        )

    if "path" in debug.columns:
        debug["_path_key"] = debug["path"].map(normalize_path)
        result["_path_key"] = result["resolved_path"].map(normalize_path)
        debug_columns = ["_path_key", "predicted", *probability_columns]
        debug = debug[debug_columns].drop_duplicates("_path_key")
        result = result.merge(debug, on="_path_key", how="left", validate="one_to_one")
        result = result.drop(columns=["_path_key"])
    else:
        if len(debug) != len(result):
            raise ValueError(
                f"DINO debug rows={len(debug)} but Moondream rows={len(result)}"
            )
        for column in ["predicted", *probability_columns]:
            result[column] = debug[column].to_numpy()

    probs = result[probability_columns].to_numpy(dtype=float)
    sorted_probs = np.sort(probs, axis=1)
    result["dino_confidence"] = np.nanmax(probs, axis=1)
    result["dino_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    result["dino_entropy"] = -np.nansum(
        np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)), axis=1
    ) / math.log(3)
    return result


def build_agreement_matrix(valid: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(
        valid["submission_predicted"].astype(int),
        valid["moondream_suggested_label"].astype(int),
        rownames=["DINO_submission"],
        colnames=["Moondream"],
        dropna=False,
    )
    matrix = matrix.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    matrix.index = [f"DINO_{LABEL_NAMES[i]}" for i in matrix.index]
    matrix.columns = [f"Moon_{LABEL_NAMES[i]}" for i in matrix.columns]
    return matrix


def save_grid(rows: pd.DataFrame, output_path: Path, n: int, cols: int) -> None:
    rows = rows.head(n).reset_index(drop=True)
    if rows.empty:
        return
    grid_rows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3.3, grid_rows * 3.7))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, rows.iterrows()):
        path = Path(row["resolved_path"])
        try:
            with Image.open(path) as source:
                ax.imshow(ImageOps.exif_transpose(source).convert("RGB"))
        except Exception as exc:
            ax.text(0.5, 0.5, f"Cannot read\n{path.name}\n{exc}", ha="center", va="center")
        dino = LABEL_NAMES.get(int(row["submission_predicted"]), "?")
        moon_value = pd.to_numeric(
            pd.Series([row.get("moondream_suggested_label")]), errors="coerce"
        ).iloc[0]
        moon = LABEL_NAMES.get(int(moon_value), "null") if pd.notna(moon_value) else "null"
        dino_conf = row.get("dino_confidence")
        dino_text = f" | D conf={float(dino_conf):.3f}" if pd.notna(dino_conf) else ""
        ax.set_title(
            f"DINO: {dino}{dino_text}\nMoon: {moon} ({row.get('moondream_confidence', '')})\n"
            f"ambiguous={row.get('moondream_taxonomy_ambiguity', '')}\n{path.name}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle("DINOv3 submission vs Moondream disagreements", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.moondream_predictions.exists():
        raise FileNotFoundError(args.moondream_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.moondream_predictions)
    required = {
        "test_index",
        "resolved_path",
        "submission_predicted",
        "moondream_suggested_label",
        "moondream_confidence",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = attach_dino_probabilities(df, args.dino_debug)
    moon_numeric = pd.to_numeric(df["moondream_suggested_label"], errors="coerce")
    valid_mask = moon_numeric.isin([0, 1, 2])
    valid = df[valid_mask].copy()
    valid["moondream_suggested_label"] = moon_numeric[valid_mask].astype(int)
    valid["agreement"] = (
        valid["submission_predicted"].astype(int)
        == valid["moondream_suggested_label"].astype(int)
    )
    valid["moondream_confidence_priority"] = (
        valid["moondream_confidence"].astype(str).str.lower().map(CONFIDENCE_PRIORITY).fillna(0)
    )
    ambiguity_source = (
        valid["moondream_taxonomy_ambiguity"]
        if "moondream_taxonomy_ambiguity" in valid.columns
        else pd.Series(False, index=valid.index)
    )
    valid["moondream_taxonomy_ambiguity_bool"] = ambiguity_source.map(parse_bool)

    disagreements = valid[~valid["agreement"]].copy()
    sort_columns = ["moondream_taxonomy_ambiguity_bool", "moondream_confidence_priority"]
    ascending = [True, False]
    if "dino_confidence" in disagreements.columns:
        sort_columns.extend(["dino_confidence", "dino_margin"])
        ascending.extend([False, False])
    disagreements = disagreements.sort_values(sort_columns, ascending=ascending)

    high_conf_disagreements = disagreements[
        (disagreements["moondream_confidence"].astype(str).str.lower() == "high")
        & (~disagreements["moondream_taxonomy_ambiguity_bool"])
    ].copy()

    matrix = build_agreement_matrix(valid)
    distribution_rows = []
    for source, column in [
        ("DINO_submission", "submission_predicted"),
        ("Moondream_valid_only", "moondream_suggested_label"),
    ]:
        source_df = df if source == "DINO_submission" else valid
        counts = source_df[column].value_counts().reindex([0, 1, 2], fill_value=0)
        for label, count in counts.items():
            distribution_rows.append(
                {
                    "source": source,
                    "label": int(label),
                    "class_name": LABEL_NAMES[int(label)],
                    "count": int(count),
                    "share": float(count / max(len(source_df), 1)),
                }
            )
    distributions = pd.DataFrame(distribution_rows)

    summary = {
        "diagnostic_only": True,
        "moondream_is_ground_truth": False,
        "official_submission_created": False,
        "rows_processed": int(len(df)),
        "valid_moondream_labels": int(len(valid)),
        "null_or_failed_moondream_labels": int(len(df) - len(valid)),
        "agreement_count": int(valid["agreement"].sum()),
        "disagreement_count": int((~valid["agreement"]).sum()),
        "agreement_rate_on_valid_moondream_labels": (
            float(valid["agreement"].mean()) if len(valid) else None
        ),
        "high_confidence_unambiguous_disagreements": int(len(high_conf_disagreements)),
        "interpretation": (
            "Agreement measures model consistency, not test accuracy. Do not select or "
            "change a competition submission solely from this comparison."
        ),
    }

    df.to_csv(args.output_dir / "comparison_all.csv", index=False)
    valid.to_csv(args.output_dir / "comparison_valid_moondream_labels.csv", index=False)
    disagreements.to_csv(args.output_dir / "disagreements.csv", index=False)
    high_conf_disagreements.to_csv(
        args.output_dir / "high_confidence_unambiguous_disagreements.csv", index=False
    )
    matrix.to_csv(args.output_dir / "agreement_matrix.csv")
    distributions.to_csv(args.output_dir / "class_distributions.csv", index=False)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_grid(
        high_conf_disagreements if len(high_conf_disagreements) else disagreements,
        args.output_dir / "disagreement_grid.png",
        n=args.grid_images,
        cols=args.grid_cols,
    )

    print(json.dumps(summary, indent=2))
    print("Agreement matrix:\n", matrix)
    print("Reports saved to:", args.output_dir)
    print("No competition submission was created.")


if __name__ == "__main__":
    main()
