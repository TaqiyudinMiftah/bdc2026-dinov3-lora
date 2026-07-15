# Moondream-Assisted Label Review

This stage uses Moondream to help a human inspect suspicious **training** labels. It is not an automatic relabeling system and must not be used on the competition test set.

## Why use it

The OOF error grid contains examples where the model predicts an obvious electronic object while the training label says Organic or Recyclable. Moondream can add an independent factual description and taxonomy suggestion so the human reviewer can prioritize the strongest cases.

The evidence order is:

```text
OOF disagreement
+ duplicate or DINO-neighbor evidence
+ Moondream factual description
+ human visual inspection
+ written competition rubric
= human decision
```

## Model and backend

The default is local Moondream 3.1 through Photon:

```text
model: moondream3.1-9B-A2B
backend: photon
```

Install the optional package inside the existing environment:

```bash
uv pip install -r requirements-moondream.txt
```

Verify PyTorch and CUDA after installation:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

## Editable competition rubric

Review this file before running inference:

```text
configs/moondream_labeling_rubric.json
```

The included definitions are conservative assumptions. Replace them with verified competition rules when available, especially for:

- organic contents inside recyclable packaging;
- mixed-material objects;
- several objects in one image;
- advertisements, illustrations, and product catalog photos.

## Dry run

Validate candidate paths without loading Moondream:

```bash
python scripts/moondream_label_review.py \
  --data-root ./BDC2026 \
  --candidates ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv \
  --tiers A,B,C,D \
  --limit 300 \
  --dry-run
```

The runner refuses any resolved path outside:

```text
BDC2026/train
```

## Smoke test

Choose a free GPU from `nvidia-smi` and run five images:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/moondream_label_review.py \
  --data-root ./BDC2026 \
  --candidates ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv \
  --output-dir ./leaderboard_pipeline_outputs/02b_moondream_review/smoke_test \
  --model moondream3.1-9B-A2B \
  --tiers A,B,C,D \
  --limit 5 \
  --checkpoint-every 1
```

Inspect the raw and parsed responses before a larger run.

## Process the first 300 strong candidates

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/moondream_label_review.py \
  --data-root ./BDC2026 \
  --candidates ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv \
  --output-dir ./leaderboard_pipeline_outputs/02b_moondream_review \
  --model moondream3.1-9B-A2B \
  --tiers A,B,C,D \
  --limit 300 \
  --checkpoint-every 5
```

The runner resumes from the existing evidence CSV. Add `--overwrite` only when intentionally regenerating all selected responses.

## Optional preview backend

The original preview model can be used through Transformers:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/moondream_label_review.py \
  --backend transformers \
  --model moondream/moondream3-preview \
  --compile \
  --data-root ./BDC2026 \
  --candidates ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv \
  --output-dir ./leaderboard_pipeline_outputs/02b_moondream_preview_review \
  --tiers A,B,C,D \
  --limit 5
```

Keep model outputs from different model versions in separate directories.

## Outputs

```text
leaderboard_pipeline_outputs/02b_moondream_review/
├── selected_candidates.csv
├── rubric_snapshot.json
├── moondream_raw_responses.jsonl
├── moondream_review_evidence.csv
├── moondream_human_review_queue.csv
└── moondream_review_summary.json
```

The script asks two questions per image:

1. neutral factual visual analysis;
2. class suggestion using the rubric.

The factual question does not contain the original label or OOF prediction, reducing anchoring.

## Human review

Open:

```text
leaderboard_pipeline_outputs/02b_moondream_review/moondream_human_review_queue.csv
```

Complete:

```text
human_action
human_new_label
human_reason
human_reviewer
human_second_review_required
```

Allowed actions:

```text
keep
relabel
exclude
needs_second_review
```

A relabel requires `human_new_label` in `{0, 1, 2}`. Every completed decision requires a written reason and reviewer name.

## Merge only human decisions

```bash
python scripts/merge_moondream_human_decisions.py \
  --base-queue ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv \
  --moondream-queue ./leaderboard_pipeline_outputs/02b_moondream_review/moondream_human_review_queue.csv \
  --output ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue_with_moondream_human_decisions.csv
```

The merge script ignores unreviewed rows and never converts `moondream_recommended_action` directly into a cleaning action.

Notebook 04 automatically prefers the merged review queue when it exists.

## Decision standard

A strong relabel candidate should normally have several agreeing signals:

```text
human sees an unambiguous object
OOF confidently predicts another class
Moondream independently supports that class
competition rubric clearly supports that class
no unresolved packaging or mixed-object ambiguity
```

Moondream confidence is language generated by the model, not a calibrated probability.

## Competition safeguard

Confirm that external VLM-assisted annotation is permitted before training on relabeled data. The safe fallback is to use Moondream only to organize human review, while the final decision remains a documented human annotation based on the provided training image and competition taxonomy.
