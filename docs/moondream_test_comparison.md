# Moondream Test Prediction Comparison

This pipeline compares an existing DINOv3 competition submission with independent Moondream pseudo-labels on the 1,458 test images.

## What it is for

Use it to inspect:

- overall agreement between DINOv3 and Moondream;
- directional disagreements such as DINO `Recyclable` versus Moondream `Electronic`;
- class-distribution differences;
- high-confidence, non-ambiguous disagreements;
- disagreement images where DINOv3 is also highly confident.

Moondream outputs are not test labels. Agreement is not accuracy, and disagreement does not prove that DINOv3 is wrong.

## Competition-rule gate

Actual test inference is blocked unless this flag is supplied:

```bash
--confirm-rules-allow-external-vlm-test-inference
```

Supply it only after checking that the competition permits external pretrained VLM inference on test images. The dry run does not need the flag.

The pipeline never:

- reads hidden test labels;
- trains or fine-tunes on test images;
- fits thresholds or class biases from test pseudo-labels;
- changes the submitted CSV;
- creates an official competition submission.

## Files

```text
configs/moondream_test_prediction_rubric.json
scripts/moondream_predict_test.py
scripts/compare_submission_moondream.py
notebooks/leaderboard_pipeline/06b_moondream_test_comparison.ipynb
```

## Install

```bash
git pull
source .venv/bin/activate
uv pip install -r requirements-moondream.txt
```

## 1. Dry-run validation

Replace the submission path with the exact CSV that received 98.42.

```bash
python scripts/moondream_predict_test.py \
  --data-root ./BDC2026 \
  --submission ./submission_NamaTim_384.csv \
  --dry-run
```

This checks:

- row count;
- submission-template identity and order;
- labels restricted to `0`, `1`, and `2`;
- one test image for every template row;
- every image path is inside `BDC2026/test`.

## 2. Five-image smoke test

For Moondream 3.1 through Photon:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/moondream_predict_test.py \
  --data-root ./BDC2026 \
  --submission ./submission_NamaTim_384.csv \
  --output-dir ./leaderboard_pipeline_outputs/06b_moondream_test_comparison/smoke_test \
  --backend photon \
  --model moondream3.1-9B-A2B \
  --limit 5 \
  --checkpoint-every 1 \
  --confirm-rules-allow-external-vlm-test-inference
```

For the user-requested preview model:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/moondream_predict_test.py \
  --data-root ./BDC2026 \
  --submission ./submission_NamaTim_384.csv \
  --output-dir ./leaderboard_pipeline_outputs/06b_moondream_test_comparison/smoke_preview \
  --backend transformers \
  --model moondream/moondream3-preview \
  --compile \
  --limit 5 \
  --checkpoint-every 1 \
  --confirm-rules-allow-external-vlm-test-inference
```

Inspect the raw and parsed outputs before running all images.

## 3. Full test inference

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/moondream_predict_test.py \
  --data-root ./BDC2026 \
  --submission ./submission_NamaTim_384.csv \
  --output-dir ./leaderboard_pipeline_outputs/06b_moondream_test_comparison/moondream_predictions \
  --backend photon \
  --model moondream3.1-9B-A2B \
  --limit 0 \
  --checkpoint-every 5 \
  --confirm-rules-allow-external-vlm-test-inference
```

The script resumes completed `test_index` values when rerun.

Outputs:

```text
leaderboard_pipeline_outputs/06b_moondream_test_comparison/moondream_predictions/
├── selected_test_manifest.csv
├── rubric_snapshot.json
├── moondream_test_raw_responses.jsonl
├── moondream_test_predictions.csv
└── moondream_test_prediction_summary.json
```

The submitted DINO label is attached only after Moondream inference. It is never included in the factual or taxonomy prompt.

## 4. Compare predictions

Locate the `test_predictions_debug.csv` produced alongside the submitted DINOv3 CSV. It adds DINO confidence, probability margin, and entropy to the comparison.

```bash
python scripts/compare_submission_moondream.py \
  --moondream-predictions ./leaderboard_pipeline_outputs/06b_moondream_test_comparison/moondream_predictions/moondream_test_predictions.csv \
  --dino-debug ./test_predictions_debug.csv \
  --output-dir ./leaderboard_pipeline_outputs/06b_moondream_test_comparison/comparison
```

Outputs:

```text
comparison/
├── comparison_all.csv
├── comparison_valid_moondream_labels.csv
├── disagreements.csv
├── high_confidence_unambiguous_disagreements.csv
├── agreement_matrix.csv
├── class_distributions.csv
├── comparison_summary.json
└── disagreement_grid.png
```

## How to interpret it

Useful observations:

- A very large shift in class distribution may indicate that Moondream is applying a different taxonomy.
- Concentrated disagreement in one direction can reveal a packaging-versus-content or electronic-accessory convention mismatch.
- High-confidence agreement from both models is still not proof of correctness.
- High-confidence disagreement should lead to visual inspection and taxonomy analysis, not automatic prediction replacement.

With only two submissions remaining, select S02 and S03 from training OOF evidence. Do not use Moondream test agreement as the optimization score.
