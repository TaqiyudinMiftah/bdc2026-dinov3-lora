import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_gpu: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select free GPUs with nvidia-smi, set CUDA_VISIBLE_DEVICES, "
            "and launch a training command. Put the train command after --."
        )
    )
    parser.add_argument("--max-gpus", type=int, default=3, help="Maximum GPUs to use.")
    parser.add_argument("--min-gpus", type=int, default=1, help="Minimum GPUs required to launch.")
    parser.add_argument("--min-free-mb", type=int, default=30000, help="Minimum free VRAM per selected GPU.")
    parser.add_argument("--max-utilization", type=int, default=30, help="Maximum GPU utilization percent for selected GPUs.")
    parser.add_argument("--include", type=str, default="", help="Comma-separated GPU ids allowed, e.g. 0,1,2.")
    parser.add_argument("--exclude", type=str, default="", help="Comma-separated GPU ids to exclude, e.g. 2.")
    parser.add_argument("--wait", action="store_true", help="Wait until at least --min-gpus are available.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Seconds between checks when --wait is used.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected GPUs and command, but do not run.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --, for example: -- python train.py ...")
    return parser.parse_args()


def parse_id_list(value: str):
    if not value:
        return None
    return {int(x.strip()) for x in value.split(",") if x.strip() != ""}


def query_gpus() -> List[GPUInfo]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("nvidia-smi not found. This launcher requires NVIDIA GPUs.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed: {e}") from e

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, total, used, free, util = parts
        gpus.append(
            GPUInfo(
                index=int(index),
                name=name,
                memory_total_mb=int(total),
                memory_used_mb=int(used),
                memory_free_mb=int(free),
                utilization_gpu=int(util),
            )
        )
    return gpus


def print_gpu_table(gpus: List[GPUInfo]):
    print("\nDetected GPUs:")
    print("idx | free MB | used MB | util % | name")
    print("----|---------|---------|--------|----------------")
    for gpu in gpus:
        print(
            f"{gpu.index:>3} | {gpu.memory_free_mb:>7} | {gpu.memory_used_mb:>7} | "
            f"{gpu.utilization_gpu:>6} | {gpu.name}"
        )


def select_gpus(gpus: List[GPUInfo], args) -> List[GPUInfo]:
    include = parse_id_list(args.include)
    exclude = parse_id_list(args.exclude) or set()

    candidates = []
    for gpu in gpus:
        if include is not None and gpu.index not in include:
            continue
        if gpu.index in exclude:
            continue
        if gpu.memory_free_mb < args.min_free_mb:
            continue
        if gpu.utilization_gpu > args.max_utilization:
            continue
        candidates.append(gpu)

    # Prefer the freest GPUs.
    candidates.sort(key=lambda g: (g.memory_free_mb, -g.utilization_gpu), reverse=True)
    selected = candidates[: args.max_gpus]
    selected.sort(key=lambda g: g.index)
    return selected


def clean_command(command):
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = ["python", "train.py"]
    return command


def maybe_add_multi_gpu_flag(command, num_selected: int):
    if num_selected <= 1:
        return command
    joined = " ".join(command)
    if "train.py" in joined and "--multi-gpu" not in command:
        return command + ["--multi-gpu"]
    return command


def main():
    args = parse_args()
    command = clean_command(args.command)

    while True:
        gpus = query_gpus()
        print_gpu_table(gpus)
        selected = select_gpus(gpus, args)

        if len(selected) >= args.min_gpus:
            break

        print(
            f"\nOnly {len(selected)} GPU(s) available, but --min-gpus={args.min_gpus}. "
            f"Criteria: min_free_mb={args.min_free_mb}, max_utilization={args.max_utilization}."
        )
        if not args.wait:
            raise SystemExit(1)
        print(f"Waiting {args.poll_seconds} seconds before checking again...")
        time.sleep(args.poll_seconds)

    selected_ids = [str(g.index) for g in selected]
    command = maybe_add_multi_gpu_flag(command, len(selected))

    print("\nSelected GPU(s):", ",".join(selected_ids))
    print("Launch command:", " ".join(shlex.quote(x) for x in command))
    print("CUDA_VISIBLE_DEVICES:", ",".join(selected_ids))

    if len(selected) < args.max_gpus:
        print(
            f"Note: requested up to {args.max_gpus} GPU(s), but selected {len(selected)} because the rest are busy or below free-memory threshold."
        )

    if args.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_ids)
    result = subprocess.run(command, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
