# Leaderboard Improvement Pipeline

This pipeline is designed around the current competition situation:

- current public Macro-F1: **98.42**
- current rank when recorded: **29**
- submission limit: **3**
- submissions already used: **1**
- submissions remaining: **2**

Do not spend another submission until a candidate passes OOF, per-class, and fold-stability checks.

## Implemented notebooks

Run these notebooks in order:

```text
00_project_config_and_submission_budget.ipynb
01_baseline_oof_diagnostics.ipynb
02_manual_label_review_queue.ipynb
02b_moondream_assisted_label_review.ipynb   # optional assistant, human decision required
03_duplicate_groups_and_leakage.ipynb
04_build_reviewed_clean_dataset.ipynb
05_duplicate_aware_folds.ipynb
06_experiment_matrix.ipynb
```

## 00 — Project contract and submission budget

Verifies:

- 26,527 training images
- 1,458 test images
- completed `outputs_dinov3_lora_384`
- five fold histories and checkpoints
- OOF predictions and probabilities
- Python, PyTorch, CUDA, package, and Git versions

Creates:

```text
leaderboard_pipeline_outputs/00_config/
├── environment.json
├── dataset_and_output_checks.json
├── submission_budget.csv
└── submission_gate.json
```

## 01 — Baseline OOF diagnostics

Creates the official baseline analysis:

- Macro-F1
- per-class precision, recall, and F1
- confusion matrix
- directional confusion pairs
- fold-level metrics
- confidence calibration
- training curves
- high-confidence error grid

Outputs:

```text
leaderboard_pipeline_outputs/01_baseline/
```

## 02 — Ranked manual label review

Creates a prioritized review queue. It does not automatically relabel anything.

Priority tiers:

```text
A: confidence >= 0.99 and true-label probability <= 0.01
B: confidence >= 0.97 and true-label probability <= 0.03
C: confidence >= 0.95 and true-label probability <= 0.05
D: cross-label duplicate evidence
E: duplicate or near-duplicate evidence
F: low-margin intrinsic ambiguity
G: general model error
```

Allowed human actions:

```text
keep
relabel
exclude
needs_second_review
```

Start with:

```text
leaderboard_pipeline_outputs/02_label_review/priority_review_first_300.csv
```

The full editable audit file is:

```text
leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv
```

## 02b — Moondream-assisted review

This optional stage uses Moondream as a **review assistant**, not an automatic labeler.

Safeguards:

- only paths inside `BDC2026/train` are accepted;
- no test images are processed;
- the first prompt requests neutral visual facts without showing the original label or OOF prediction;
- the second prompt uses the editable competition rubric;
- packaging-versus-contents and mixed-object cases are sent to second review;
- Moondream recommendations are never merged automatically;
- a human reviewer, written reason, and explicit action are required.

Install the optional runtime:

```bash
uv pip install -r requirements-moondream.txt
```

Review the editable rubric before inference:

```text
configs/moondream_labeling_rubric.json
```

Dry run without loading the model:

```bash
python scripts/moondream_label_review.py \
  --data-root ./BDC2026 \
  --candidates ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv \
  --tiers A,B,C,D \
  --limit 300 \
  --dry-run
```

Five-image smoke test on a free GPU:

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

Process the first 300 strong candidates:

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

Outputs:

```text
leaderboard_pipeline_outputs/02b_moondream_review/
├── selected_candidates.csv
├── rubric_snapshot.json
├── moondream_raw_responses.jsonl
├── moondream_review_evidence.csv
├── moondream_human_review_queue.csv
└── moondream_review_summary.json
```

Complete these columns in `moondream_human_review_queue.csv`:

```text
human_action
human_new_label
human_reason
human_reviewer
human_second_review_required
```

Then merge only the completed human decisions:

```bash
python scripts/merge_moondream_human_decisions.py \
  --base-queue ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue.csv \
  --moondream-queue ./leaderboard_pipeline_outputs/02b_moondream_review/moondream_human_review_queue.csv \
  --output ./leaderboard_pipeline_outputs/02_label_review/ranked_review_queue_with_moondream_human_decisions.csv
```

Before notebook 04, use the merged review queue as the reviewed decision source. Do not use Moondream output if the competition rules prohibit external VLM-assisted annotation.

## 03 — Duplicate groups and fold leakage

Combines:

- exact MD5 duplicate groups
- pHash candidate pairs
- DINO embedding candidate pairs
- manually confirmed DINO duplicate decisions
- original fold assignments

It produces:

- **strict groups**: exact duplicates and manually confirmed DINO duplicates
- **candidate groups**: pHash and unreviewed DINO similarity evidence

Only strict groups are automatically used for duplicate-aware cross-validation.

Outputs:

```text
leaderboard_pipeline_outputs/03_duplicates/
├── unified_duplicate_edges.csv
├── oof_with_duplicate_groups.csv
├── strict_group_leakage.csv
├── candidate_group_leakage.csv
└── duplicate_leakage_summary.json
```

## 04 — Build reviewed clean dataset

Combines human decisions with conservative automatic cleaning.

Automatic exclusions are limited to:

```text
corrupt or unreadable files
redundant same-label exact MD5 duplicates
```

The notebook creates an audit first and starts with:

```python
APPLY_CLEAN_COPY = False
```

Review these files before changing it to `True`:

```text
leaderboard_pipeline_outputs/04_clean_dataset/
├── cleaning_audit.csv
├── excluded_images.csv
├── relabeled_images.csv
├── pending_second_review.csv
└── clean_manifest.csv
```

The clean dataset is created as:

```text
BDC2026_clean_v1/
├── train/
├── test -> original test
└── submission.csv -> original template
```

## 05 — Duplicate-aware folds

Uses `StratifiedGroupKFold` to build five folds while keeping every strict duplicate group in a single fold.

Output:

```text
leaderboard_pipeline_outputs/05_folds/train_folds_duplicate_aware.csv
```

Training with these folds uses:

```text
scripts/train_with_precomputed_folds.py
```

## 06 — Controlled experiment matrix

Defines the minimum experiment sequence:

```text
E00: completed 384 baseline
E01: clean 384 seed 42 — first S02 candidate
E02: clean 384 seed 123 — ensemble diversity
E03: optional weighted-sampler 384 model
E04: optional 224-resolution diversity model
```

It generates:

```text
leaderboard_pipeline_outputs/06_experiments/
├── experiment_registry.csv
├── generated_commands.sh
└── S02_decision_checklist.csv
```

Run **E01 first**. Do not run optional experiments until E01 OOF analysis is complete.

## Run in Jupyter

```bash
git pull
source .venv/bin/activate
jupyter lab notebooks/leaderboard_pipeline/
```

Execute the fully automatic analysis notebooks:

```bash
for notebook in \
  00_project_config_and_submission_budget \
  01_baseline_oof_diagnostics \
  02_manual_label_review_queue \
  03_duplicate_groups_and_leakage; do
  jupyter nbconvert --to notebook --execute --inplace \
    "notebooks/leaderboard_pipeline/${notebook}.ipynb"
done
```

Notebook 02b deliberately leaves model execution disabled until the rubric, GPU selection, and candidate count have been reviewed. Notebook 04 requires human decisions before creating the clean dataset.

## Train E01

```bash
python scripts/launch_train_auto_gpus.py \
  --max-gpus 3 \
  --min-gpus 1 \
  --min-free-mb 30000 \
  --max-utilization 30 \
  -- python scripts/train_with_precomputed_folds.py \
    --data-root ./BDC2026_clean_v1 \
    --fold-csv ./leaderboard_pipeline_outputs/05_folds/train_folds_duplicate_aware.csv \
    --output-dir ./experiments/E01_clean_v1_384_seed42 \
    --image-size 384 \
    --epochs 25 \
    --batch-size 2 \
    --valid-batch-size 4 \
    --grad-accum 8 \
    --seed 42 \
    --use-class-weights \
    --scheduler plateau \
    --early-stopping-patience 8
```

## Submission strategy

### Submission S02

Reserve for the strongest cleaned, duplicate-aware single model. Gate:

```text
OOF Macro-F1 gain >= 0.0010
at least 3 of 5 folds improve
weakest-class F1 does not drop by more than 0.0005
prediction file and submission schema are reproducible
```

### Submission S03

Reserve for the final diverse ensemble. Gate:

```text
OOF gain over the best single model >= 0.0005
at least 3 of 5 folds improve
ensemble weights optimized only from OOF predictions
model disagreement analysis confirms useful diversity
submission order and schema are validated
```

## Planned final phase

```text
07_model_disagreement_analysis.ipynb
08_oof_ensemble_optimization.ipynb
09_class_bias_calibration.ipynb
10_final_submission_pipeline.ipynb
11_experiment_report.ipynb
```

Do not upload S02 until E01 has been compared against the baseline using the same OOF and fold-level checks.
