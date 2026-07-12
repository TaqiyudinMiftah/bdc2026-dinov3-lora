# OOF Error Analysis

Use the error-analysis notebook after training finishes and the output directory contains:

```text
outputs_dinov3_lora_384/
├── oof_predictions.csv
├── fold0_history.csv
├── fold1_history.csv
├── fold2_history.csv
├── fold3_history.csv
├── fold4_history.csv
└── ...
```

## Launch the notebook

Install notebook dependencies after pulling the latest repository:

```bash
git pull
uv pip install -r requirements.txt
```

Start JupyterLab:

```bash
jupyter lab notebooks/oof_error_analysis.ipynb
```

In VS Code, you can also open:

```text
notebooks/oof_error_analysis.ipynb
```

and select the repository `.venv` Python kernel.

## Run the same analysis from the terminal

Quick analysis without reading every image:

```bash
python scripts/error_analysis.py \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora_384
```

Full image-quality analysis:

```bash
python scripts/error_analysis.py \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora_384 \
  --compute-image-metadata
```

Use custom thresholds:

```bash
python scripts/error_analysis.py \
  --data-root ./BDC2026 \
  --output-dir ./outputs_dinov3_lora_384 \
  --high-confidence 0.85 \
  --very-high-confidence 0.95 \
  --low-margin 0.15 \
  --compute-image-metadata
```

## What the analysis checks

- Overall OOF Macro-F1 and per-class metrics.
- Confusion matrix and directional confusion pairs.
- Highest-confidence wrong predictions.
- Most ambiguous predictions using probability margin and entropy.
- Error rate by fold.
- Training and validation curves for each fold.
- Possible label-noise candidates.
- Exact, perceptual-hash, and DINO duplicate conflicts when EDA reports exist.
- Image size, aspect ratio, brightness, contrast, and sharpness when metadata analysis is enabled.
- Optional PCA view of DINO embeddings.

## Exported reports

Reports are saved under:

```text
outputs_dinov3_lora_384/error_analysis/
```

Important files:

```text
misclassified_all.csv
high_confidence_wrong.csv
most_uncertain_predictions.csv
label_review_candidates.csv
per_class_metrics.csv
confusion_matrix.csv
confusion_pairs.csv
error_category_summary.csv
error_analysis_summary.json
error_analysis_summary.md
high_confidence_wrong_grid.png
most_ambiguous_grid.png
training_macro_f1.png
training_valid_loss.png
training_train_loss.png
```

## How to interpret common error categories

- `possible_label_noise`: the model is extremely confident in another class and assigns almost zero probability to the dataset label. Review manually before relabeling.
- `cross_label_duplicate_review`: duplicate or near-duplicate images appear with conflicting labels.
- `duplicate_or_near_duplicate`: the image appears in an exact, pHash, or DINO duplicate report.
- `low_visual_quality`: image size, aspect ratio, brightness, contrast, or sharpness may make recognition difficult.
- `intrinsic_ambiguity`: the top two class probabilities are close or predictive entropy is high.
- `systematic_model_confusion`: the model is confident but no obvious data-quality issue was detected.
- `general_model_error`: a normal error that does not meet the stronger diagnostic rules.

Do not automatically remove or relabel images based only on these heuristics. Review the image, its duplicates, and its class context first. Retrain with the same folds after any cleaning so the OOF comparison remains fair.
