#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-moondream3.1-9B-A2B}"

if command -v uv >/dev/null 2>&1; then
  uv pip install --upgrade --reinstall "moondream>=1.3.0" kestrel
else
  python -m pip install --upgrade --force-reinstall "moondream>=1.3.0" kestrel
fi

python scripts/check_moondream_runtime.py --model "$MODEL"
