# Moondream Photon model-registry troubleshooting

The error

```text
ValueError: Unknown model 'moondream3.1-9B-A2B'. Known models: moondream2, moondream3-preview
```

means the installed Photon/Kestrel runtime predates the registry entry for Moondream 3.1. It is not a dataset, CUDA, or submission-alignment error.

## Inspect installed versions and registry support

```bash
python scripts/check_moondream_runtime.py --model moondream3.1-9B-A2B
```

## Upgrade the local runtime

```bash
git pull
source .venv/bin/activate
chmod +x scripts/install_moondream_runtime.sh
./scripts/install_moondream_runtime.sh moondream3.1-9B-A2B
```

Equivalent manual command:

```bash
uv pip install --upgrade --reinstall "moondream>=1.3.0" kestrel
python scripts/check_moondream_runtime.py --model moondream3.1-9B-A2B
```

Only rerun the full 1,458-image job after the registry check passes.

## Immediate explicit fallback

The traceback confirms that the current runtime recognizes `moondream3-preview`. To continue without upgrading:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/moondream_predict_test.py \
  --data-root ./BDC2026 \
  --submission ./submission_NamaTim.csv \
  --output-dir ./leaderboard_pipeline_outputs/06b_moondream_test_comparison/moondream3_preview_predictions \
  --backend photon \
  --model moondream3-preview \
  --limit 5 \
  --checkpoint-every 1 \
  --confirm-rules-allow-external-vlm-test-inference
```

Inspect the five-image smoke test, then change `--limit 5` to `--limit 0`.

Keep different models in different output directories. Do not mix resumed predictions from `moondream3-preview` and `moondream3.1-9B-A2B`.
