# Cleaning Pipeline Notebooks

Run the notebooks in order:

1. `00_manifest_and_corrupt_images.ipynb`
2. `01_geometry_and_quality_review.ipynb`
3. `02_exact_duplicates_md5.ipynb`
4. `03_near_duplicates_phash.ipynb`
5. `04_embedding_similarity_dinov3.ipynb`
6. `05_decision_policy_and_clean_copy.ipynb`

Intermediate outputs are written to:

```text
eda_outputs/notebook_pipeline/
```

The policy is conservative:

- automatically exclude only corrupt files and redundant same-label exact duplicates;
- manually review cross-label duplicates, pHash pairs, DINO similarity pairs, and quality anomalies;
- never modify the original `BDC2026` directory;
- create `BDC2026_clean` only after reviewing `review_decisions.csv`.

Launch:

```bash
jupyter lab notebooks/cleaning_pipeline/
```
