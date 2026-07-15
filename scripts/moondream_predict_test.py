#!/usr/bin/env python3
"""Run Moondream on BDC 2026 test images for diagnostic comparison only.

This script never reads test labels (none should exist), never modifies a submitted
CSV, never trains or tunes a model, and never creates an official submission.
Actual inference requires an explicit acknowledgement that the competition rules
allow external pretrained VLM inference on test images.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bdc2026.dataset import build_test_df
from scripts.moondream_label_review import (
    ModelAnswer,
    factual_prompt,
    load_json,
    load_model,
    normalize_bool,
    parse_json_answer,
    run_query,
    taxonomy_prompt,
    write_jsonl,
)

DEFAULT_MODEL = "moondream3.1-9B-A2B"
DEFAULT_RUBRIC = Path("./configs/moondream_test_prediction_rubric.json")
DEFAULT_OUTPUT_DIR = Path(
    "./leaderboard_pipeline_outputs/06b_moondream_test_comparison/moondream_predictions"
)
ALLOWED_LABELS = {0, 1, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test-only Moondream pseudo-labels for diagnostic comparison."
    )
    parser.add_argument("--data-root", type=Path, default=Path("./BDC2026"))
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="The already-generated DINOv3 submission to compare against.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--backend", choices=["photon", "transformers"], default="photon")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all test images.")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate alignment and paths without loading Moondream.",
    )
    parser.add_argument(
        "--confirm-rules-allow-external-vlm-test-inference",
        action="store_true",
        help=(
            "Required for actual inference. Confirms that you checked the competition "
            "rules and external pretrained VLM inference on test images is permitted."
        ),
    )
    return parser.parse_args()


def validate_submission(
    data_root: Path, submission_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    template_path = data_root / "submission.csv"
    test_root = (data_root / "test").resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Submission template not found: {template_path}")
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission to compare not found: {submission_path}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    template = pd.read_csv(template_path)
    submission = pd.read_csv(submission_path)
    if "predicted" not in template.columns or "predicted" not in submission.columns:
        raise ValueError("Both template and submission must contain a 'predicted' column.")
    if len(template) != len(submission):
        raise ValueError(
            f"Template rows={len(template)} but submission rows={len(submission)}."
        )

    identity_columns = [column for column in template.columns if column != "predicted"]
    missing_identity = [column for column in identity_columns if column not in submission]
    if missing_identity:
        raise ValueError(
            f"Submission is missing template identity columns: {missing_identity}"
        )
    for column in identity_columns:
        left = template[column].astype(str).reset_index(drop=True)
        right = submission[column].astype(str).reset_index(drop=True)
        if not left.equals(right):
            mismatch = (left != right).to_numpy().nonzero()[0][:10].tolist()
            raise ValueError(
                f"Submission order/content differs from template in column {column!r}; "
                f"first mismatched rows: {mismatch}"
            )

    numeric_pred = pd.to_numeric(submission["predicted"], errors="coerce")
    if numeric_pred.isna().any():
        bad = submission.loc[numeric_pred.isna()].head(10)
        raise ValueError(f"Non-numeric predictions found:\n{bad}")
    submission["predicted"] = numeric_pred.astype(int)
    invalid = sorted(set(submission["predicted"]) - ALLOWED_LABELS)
    if invalid:
        raise ValueError(f"Submission contains invalid labels: {invalid}")

    test_df = build_test_df(test_root)
    if len(test_df) != len(template):
        raise ValueError(
            f"Test images={len(test_df)}, template rows={len(template)}. "
            "Cannot guarantee row alignment."
        )

    manifest = test_df.copy().reset_index(drop=True)
    manifest["test_index"] = range(len(manifest))
    manifest["resolved_path"] = manifest["path"].map(
        lambda value: str(Path(str(value)).resolve())
    )
    for path in manifest["resolved_path"]:
        resolved = Path(path)
        if test_root != resolved and test_root not in resolved.parents:
            raise ValueError(f"Refusing path outside test directory: {resolved}")

    manifest["submission_predicted"] = submission["predicted"].to_numpy()
    for column in identity_columns:
        manifest[f"template_{column}"] = template[column].to_numpy()
    return manifest, template, submission


def prepare_query_image(model: Any, image: Image.Image) -> Any:
    """Encode once when supported so the two queries can reuse image features."""
    encoder = getattr(model, "encode_image", None)
    if encoder is None:
        return image
    try:
        return encoder(image)
    except Exception:
        return image


def flatten_test_result(
    row: pd.Series,
    model_name: str,
    backend: str,
    facts: dict[str, Any] | None,
    facts_status: str,
    decision: dict[str, Any] | None,
    decision_status: str,
    factual_raw: str,
    decision_raw: str,
    factual_seconds: float,
    decision_seconds: float,
) -> dict[str, Any]:
    suggested_label = decision.get("suggested_label") if decision else None
    try:
        suggested_label = int(suggested_label) if suggested_label is not None else None
    except (TypeError, ValueError):
        suggested_label = None
    if suggested_label not in ALLOWED_LABELS:
        suggested_label = None

    submission_label = int(row["submission_predicted"])
    result = row.to_dict()
    result.update(
        {
            "moondream_model": model_name,
            "moondream_backend": backend,
            "moondream_facts_json": (
                json.dumps(facts, ensure_ascii=False) if facts is not None else ""
            ),
            "moondream_facts_parse_status": facts_status,
            "moondream_suggested_label": suggested_label,
            "moondream_suggested_class_name": (
                decision.get("suggested_class_name", "") if decision else ""
            ),
            "moondream_confidence": decision.get("confidence", "") if decision else "",
            "moondream_taxonomy_ambiguity": normalize_bool(
                decision.get("taxonomy_ambiguity") if decision else None
            ),
            "moondream_recommended_action": (
                decision.get("recommended_action", "") if decision else ""
            ),
            "moondream_reason": decision.get("reason", "") if decision else "",
            "moondream_rubric_rule_used": (
                decision.get("rubric_rule_used", "") if decision else ""
            ),
            "moondream_decision_json": (
                json.dumps(decision, ensure_ascii=False)
                if decision is not None
                else ""
            ),
            "moondream_decision_parse_status": decision_status,
            "moondream_factual_raw": factual_raw,
            "moondream_decision_raw": decision_raw,
            "moondream_factual_seconds": factual_seconds,
            "moondream_decision_seconds": decision_seconds,
            "submission_agrees_moondream": (
                suggested_label is not None and submission_label == suggested_label
            ),
            "diagnostic_only": True,
        }
    )
    return result


def checkpoint(records: list[dict[str, Any]], evidence_path: Path) -> None:
    evidence = pd.DataFrame(records)
    if not evidence.empty and "test_index" in evidence.columns:
        evidence = (
            evidence.sort_values("test_index")
            .drop_duplicates("test_index", keep="last")
            .reset_index(drop=True)
        )
    evidence.to_csv(evidence_path, index=False)


def build_summary(evidence: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    valid = evidence[
        pd.to_numeric(
            evidence.get("moondream_suggested_label", pd.Series(dtype=float)),
            errors="coerce",
        ).isin(ALLOWED_LABELS)
    ].copy()
    agreement = (
        valid["submission_agrees_moondream"].fillna(False).astype(bool)
        if len(valid)
        else pd.Series(dtype=bool)
    )
    summary = {
        "diagnostic_only": True,
        "official_submission_created": False,
        "model": args.model,
        "backend": args.backend,
        "submission_compared": str(args.submission),
        "test_images_total": int(len(evidence)),
        "moondream_valid_labels": int(len(valid)),
        "moondream_null_or_failed": int(len(evidence) - len(valid)),
        "agreement_count": int(agreement.sum()) if len(valid) else 0,
        "disagreement_count": int((~agreement).sum()) if len(valid) else 0,
        "agreement_rate_on_valid_moondream_labels": (
            float(agreement.mean()) if len(valid) else None
        ),
        "rules_confirmation_received": bool(
            args.confirm_rules_allow_external_vlm_test_inference
        ),
    }
    for column in (
        "submission_predicted",
        "moondream_suggested_label",
        "moondream_confidence",
        "moondream_taxonomy_ambiguity",
    ):
        if column in evidence.columns:
            summary[f"{column}_counts"] = {
                str(key): int(value)
                for key, value in evidence[column]
                .fillna("null")
                .astype(str)
                .value_counts()
                .to_dict()
                .items()
            }
    return summary


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.submission = args.submission.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.rubric = args.rubric.expanduser().resolve()

    manifest, _, _ = validate_submission(args.data_root, args.submission)
    selected = manifest.iloc[args.start :].copy()
    if args.limit > 0:
        selected = selected.head(args.limit)
    selected = selected.reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "selected_test_manifest.csv"
    evidence_path = args.output_dir / "moondream_test_predictions.csv"
    raw_jsonl_path = args.output_dir / "moondream_test_raw_responses.jsonl"
    summary_path = args.output_dir / "moondream_test_prediction_summary.json"
    rubric_snapshot_path = args.output_dir / "rubric_snapshot.json"

    rubric = load_json(args.rubric)
    selected.to_csv(selected_path, index=False)
    rubric_snapshot_path.write_text(
        json.dumps(rubric, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Selected test images:", len(selected))
    print("Selection:", selected_path)
    print("Submission distribution:")
    print(selected["submission_predicted"].value_counts().sort_index())

    if args.dry_run:
        print("Dry run complete. No model was loaded and no test image was inferred.")
        return

    if not args.confirm_rules_allow_external_vlm_test_inference:
        raise RuntimeError(
            "Actual test inference is blocked. First verify the competition rules, then "
            "rerun with --confirm-rules-allow-external-vlm-test-inference."
        )

    existing_records: list[dict[str, Any]] = []
    completed_indices: set[int] = set()
    if evidence_path.exists() and not args.overwrite:
        existing = pd.read_csv(evidence_path)
        existing_records = existing.to_dict(orient="records")
        if "test_index" in existing.columns:
            completed_indices = set(
                pd.to_numeric(existing["test_index"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
        print(f"Resume mode: {len(completed_indices)} test rows already completed.")

    model = load_model(args.backend, args.model, args.compile)
    records = list(existing_records)

    for _, row in tqdm(
        selected.iterrows(), total=len(selected), desc="Moondream test inference"
    ):
        test_index = int(row["test_index"])
        if test_index in completed_indices and not args.overwrite:
            continue

        try:
            with Image.open(row["resolved_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            query_image = prepare_query_image(model, image)

            factual_result = run_query(model, args.backend, query_image, factual_prompt())
            facts, facts_status = parse_json_answer(factual_result.answer)

            if facts is None:
                decision_result = ModelAnswer(answer="", elapsed_seconds=0.0)
                decision = None
                decision_status = "skipped_facts_parse_failed"
            else:
                decision_result = run_query(
                    model,
                    args.backend,
                    query_image,
                    taxonomy_prompt(facts, rubric),
                )
                decision, decision_status = parse_json_answer(decision_result.answer)

            record = flatten_test_result(
                row=row,
                model_name=args.model,
                backend=args.backend,
                facts=facts,
                facts_status=facts_status,
                decision=decision,
                decision_status=decision_status,
                factual_raw=factual_result.answer,
                decision_raw=decision_result.answer,
                factual_seconds=factual_result.elapsed_seconds,
                decision_seconds=decision_result.elapsed_seconds,
            )
        except Exception as exc:
            record = row.to_dict()
            record.update(
                {
                    "moondream_model": args.model,
                    "moondream_backend": args.backend,
                    "moondream_error": repr(exc),
                    "moondream_facts_parse_status": "runtime_error",
                    "moondream_decision_parse_status": "runtime_error",
                    "moondream_suggested_label": None,
                    "submission_agrees_moondream": False,
                    "diagnostic_only": True,
                }
            )

        records.append(record)
        write_jsonl(raw_jsonl_path, record)
        completed_indices.add(test_index)

        if (
            args.checkpoint_every > 0
            and len(records) % args.checkpoint_every == 0
        ):
            checkpoint(records, evidence_path)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    checkpoint(records, evidence_path)
    evidence = pd.read_csv(evidence_path)
    summary = build_summary(evidence, args)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("Predictions:", evidence_path)
    print("Raw responses:", raw_jsonl_path)
    print(
        "No official submission was created. Use the comparison script/notebook for "
        "diagnostics; do not treat Moondream labels as test ground truth."
    )


if __name__ == "__main__":
    main()
