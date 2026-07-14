# Leaderboard Improvement Pipeline

This pipeline is designed around the current competition situation:

- current public Macro-F1: **98.42**
- current rank when recorded: **29**
- submission limit: **3**
- submissions already used: **1**
- submissions remaining: **2**

The pipeline therefore uses a strict rule: do not spend another submission until a candidate passes OOF, per-class, and fold-stability checks.

## Implemented phase

Run these notebooks in order:

```text
00_project_config_and_submission_budget.ipynb
01_baseline_oof_diagnostics.ipynb
02_manual_label_review_queue.ipynb
03_duplicate_groups_and_leakage.ipynb
```

### Notebook 00

Verifies:

- 26,527 training images
- 1,458 test images
- completed `outputs_dinov3_lora_384`
- 5 fold histories
- 5 fold checkpoints
- OOF predictions and probabilities
- Python, PyTorch, CUDA, package, and Git versions

It creates:

```text
leaderboard_pipeline_outputs/00_config/
├── environment.json
├── dataset_and_output_checks.json
├── submission_budget.csv
└── submission_gate.json
```

### Notebook 01

Creates the official OOF baseline:

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

### Notebook 02

Creates a ranked manual-review queue. It does not automatically relabel anything.

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

Review actions:

```text
keep
relabel
exclude
needs_second_review
```

The first file to review is:

```text
leaderboard_pipeline_outputs/02_label_review/priority_review_first_300.csv
```

Do not relabel only because the model is confident. Use the image, competition taxonomy, duplicate evidence, and a written review reason.

### Notebook 03

Combines:

- exact MD5 duplicate groups
- pHash candidate pairs
- DINO embedding candidate pairs
- manually confirmed DINO duplicate decisions
- original fold assignments

It produces two types of groups:

- **strict groups**: exact duplicates and manually confirmed DINO duplicates
- **candidate groups**: pHash and unreviewed DINO similarity evidence

Only strict groups should be used automatically for duplicate-aware cross-validation.

Outputs:

```text
leaderboard_pipeline_outputs/03_duplicates/
├── unified_duplicate_edges.csv
├── oof_with_duplicate_groups.csv
├── strict_group_leakage.csv
├── candidate_group_leakage.csv
└── duplicate_leakage_summary.json
```

## Run in Jupyter

From the repository root:

```bash
git pull
source .venv/bin/activate
jupyter lab notebooks/leaderboard_pipeline/
```

Run notebooks in numerical order.

## Execute notebooks from the terminal

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/leaderboard_pipeline/00_project_config_and_submission_budget.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  notebooks/leaderboard_pipeline/01_baseline_oof_diagnostics.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  notebooks/leaderboard_pipeline/02_manual_label_review_queue.ipynb

jupyter nbconvert --to notebook --execute --inplace \
  notebooks/leaderboard_pipeline/03_duplicate_groups_and_leakage.ipynb
```

## Submission strategy

### Submission S02

Reserve for the strongest cleaned, duplicate-aware single model. Suggested gate:

```text
OOF Macro-F1 gain >= 0.0010
at least 3 of 5 folds improve
weakest-class F1 does not drop by more than 0.0005
prediction file and submission schema are reproducible
```

### Submission S03

Reserve for the final diverse ensemble. Suggested gate:

```text
OOF gain over the best single model >= 0.0005
at least 3 of 5 folds improve
ensemble weights optimized only from OOF predictions
model disagreement analysis confirms useful diversity
submission order and schema are validated
```

## Planned next implementation phase

```text
04_duplicate_aware_folds.ipynb
05_build_clean_dataset.ipynb
06_experiment_matrix.ipynb
07_model_disagreement_analysis.ipynb
08_oof_ensemble_optimization.ipynb
09_class_bias_calibration.ipynb
10_final_submission_pipeline.ipynb
11_experiment_report.ipynb
```

The next phase should begin only after the priority review CSV and strict duplicate groups have been inspected.
