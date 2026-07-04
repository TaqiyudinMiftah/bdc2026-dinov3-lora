import os
import re
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import WeightedRandomSampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(path):
    path = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_image_files(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files, key=natural_key)


def compute_class_weights(labels, num_classes: int, mode="inverse", beta=0.999):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    if mode == "inverse":
        weights = counts.sum() / (num_classes * counts)
    elif mode == "sqrt_inverse":
        weights = np.sqrt(counts.sum() / (num_classes * counts))
    elif mode == "effective":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

    return weights.astype(np.float32)


def make_weighted_sampler(labels, num_classes: int, mode="sqrt_inverse"):
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    if mode == "inverse":
        class_weights = 1.0 / counts
    elif mode == "sqrt_inverse":
        class_weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown sampler mode: {mode}")

    sample_weights = torch.DoubleTensor(class_weights[labels])
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def macro_f1_and_report(y_true, probs, id2label: dict):
    preds = probs.argmax(axis=1)
    macro_f1 = f1_score(y_true, preds, average="macro")
    per_class_f1 = f1_score(
        y_true,
        preds,
        average=None,
        labels=list(range(len(id2label))),
    )
    report = classification_report(
        y_true,
        preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=5,
    )
    cm = confusion_matrix(y_true, preds)
    return macro_f1, per_class_f1, report, cm
