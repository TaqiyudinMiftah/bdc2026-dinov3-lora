#!/usr/bin/env python3
"""Merge only human-confirmed Moondream review decisions into the main review queue.

Moondream recommendations are never applied automatically. A row is merged only when
`human_action`, `human_reason`, and `human_reviewer` have been completed and validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ALLOWED_ACTIONS = {"keep", "relabel", "exclude", "needs_second_review"}
ALLOWED_LABELS = {0, 1, 2}
LABEL_TO_CLASS = {
    0: "0_Recyclable",
    1: "1_Electronic",
    2: "2_Organic",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-queue",
        type=Path,
        default=Path(
            "./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv"
        ),
    )
    parser.add_argument(
        "--moondream-queue",
        type=Path,
        default=Path(
            "./leaderboard_pipeline_outputs/02b_moondream_review/"
            "moondream_human_review_queue.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "./leaderboard_pipeline_outputs/02_label_review/"
            "ranked_review_queue_with_moondream_human_decisions.csv"
        ),
    )
    return parser.parse_args()


def get_key(row: pd.Series) -> str:
    """Use class folder plus basename so queues remain portable across machines."""
    class_name = row.get("class_name")
    if pd.isna(class_name) or not str(class_name).strip():
        try:
            class_name = LABEL_TO_CLASS[int(row.get("label"))]
        except (KeyError, TypeError, ValueError):
            class_name = "unknown_class"

    for column in ("resolved_path", "path"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return f"{class_name}/{Path(str(value)).name}"
    raise ValueError("Row has neither resolved_path nor path")


def validate_human_rows(review: pd.DataFrame) -> pd.DataFrame:
    action = review.get("human_action", pd.Series("", index=review.index)).fillna("")
    selected = review[action.astype(str).str.strip() != ""].copy()
    errors = []

    for index, row in selected.iterrows():
        human_action = str(row.get("human_action", "")).strip()
        reason = str(row.get("human_reason", "")).strip()
        reviewer = str(row.get("human_reviewer", "")).strip()

        if human_action not in ALLOWED_ACTIONS:
            errors.append(f"row {index}: invalid human_action={human_action!r}")
        if not reason:
            errors.append(f"row {index}: human_reason is required")
        if not reviewer:
            errors.append(f"row {index}: human_reviewer is required")

        if human_action == "relabel":
            try:
                new_label = int(row.get("human_new_label"))
            except (TypeError, ValueError):
                errors.append(
                    f"row {index}: relabel requires human_new_label 0, 1, or 2"
                )
            else:
                if new_label not in ALLOWED_LABELS:
                    errors.append(f"row {index}: invalid human_new_label={new_label}")

    if errors:
        preview = "\n".join(errors[:30])
        raise ValueError(f"Human decision validation failed:\n{preview}")
    return selected


def main():
    args = parse_args()
    if not args.base_queue.exists():
        raise FileNotFoundError(args.base_queue)
    if not args.moondream_queue.exists():
        raise FileNotFoundError(args.moondream_queue)

    base = pd.read_csv(args.base_queue)
    review = pd.read_csv(args.moondream_queue)
    selected = validate_human_rows(review)

    base["_merge_key"] = base.apply(get_key, axis=1)
    selected["_merge_key"] = selected.apply(get_key, axis=1)

    if selected["_merge_key"].duplicated().any():
        duplicates = selected.loc[
            selected["_merge_key"].duplicated(), "_merge_key"
        ].tolist()
        raise ValueError(f"Duplicate reviewed paths found: {duplicates[:10]}")

    decision_by_key = selected.set_index("_merge_key").to_dict(orient="index")
    merged_count = 0
    for index, row in base.iterrows():
        decision = decision_by_key.get(row["_merge_key"])
        if decision is None:
            continue

        action = str(decision["human_action"]).strip()
        base.at[index, "review_action"] = action
        base.at[index, "new_label"] = (
            int(decision["human_new_label"]) if action == "relabel" else ""
        )
        base.at[index, "review_reason"] = str(decision["human_reason"]).strip()
        base.at[index, "reviewer"] = str(decision["human_reviewer"]).strip()
        base.at[index, "second_review_required"] = (
            action == "needs_second_review"
            or str(decision.get("human_second_review_required", "")).strip().lower()
            in {"true", "yes", "1"}
        )
        base.at[index, "moondream_model"] = decision.get("moondream_model", "")
        base.at[index, "moondream_suggested_label"] = decision.get(
            "moondream_suggested_label", ""
        )
        base.at[index, "moondream_confidence"] = decision.get(
            "moondream_confidence", ""
        )
        base.at[index, "moondream_reason"] = decision.get("moondream_reason", "")
        merged_count += 1

    unmatched = sorted(set(decision_by_key) - set(base["_merge_key"]))
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} human-reviewed paths were not found in the base queue. "
            f"First keys: {unmatched[:5]}"
        )

    base = base.drop(columns=["_merge_key"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.output, index=False)

    summary = {
        "base_rows": int(len(base)),
        "human_decisions_merged": int(merged_count),
        "actions": {
            str(key): int(value)
            for key, value in selected["human_action"].value_counts().to_dict().items()
        },
        "automatic_moondream_decisions_applied": 0,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
