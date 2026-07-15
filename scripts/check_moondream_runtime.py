#!/usr/bin/env python3
"""Check whether the installed Photon/Kestrel runtime recognizes a Moondream model."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moondream3.1-9B-A2B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Python:", sys.version.split()[0])
    print("moondream:", package_version("moondream"))
    print("kestrel:", package_version("kestrel"))

    try:
        from kestrel.models.registry import get_spec
    except Exception as exc:
        raise RuntimeError(
            "Could not import the Photon/Kestrel model registry. Reinstall the runtime with:\n"
            "  uv pip install --upgrade --reinstall 'moondream>=1.3.0' kestrel"
        ) from exc

    try:
        spec = get_spec(args.model)
    except ValueError as exc:
        message = str(exc)
        raise RuntimeError(
            f"The installed Photon/Kestrel runtime does not recognize {args.model!r}.\n"
            f"Registry response: {message}\n\n"
            "Upgrade both packages, then run this check again:\n"
            "  uv pip install --upgrade --reinstall 'moondream>=1.3.0' kestrel\n"
            f"  python scripts/check_moondream_runtime.py --model {args.model}\n\n"
            "Immediate supported fallback from the reported registry:\n"
            "  --backend photon --model moondream3-preview\n"
            "Do not silently fall back when comparing models; record the actual model used."
        ) from exc

    print(f"Runtime recognizes model: {args.model}")
    print("Model spec:", spec)


if __name__ == "__main__":
    main()
