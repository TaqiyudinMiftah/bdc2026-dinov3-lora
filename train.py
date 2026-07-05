import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import get_cosine_schedule_with_warmup

from bdc2026.config import TrainConfig
from bdc2026.dataset import WasteDataset, build_train_df, get_transforms
from bdc2026.model import (
    Dinov3LoraClassifier,
    count_trainable_params,
    get_optimizer,
    get_trainable_state_dict,
    unwrap_model,
)
from bdc2026.utils import (
    seed_everything,
    get_device,
    compute_class_weights,
    make_weighted_sampler,
    macro_f1_and_report,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs_dinov3_lora"))
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--valid-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine"])
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-7)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--class-weight-mode", type=str, default="inverse")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-weight-mode", type=str, default="sqrt_inverse")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true", help="Use torch.nn.DataParallel over all visible CUDA devices.")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def cfg_from_args(args):
    cfg = TrainConfig(data_root=args.data_root)
    cfg.output_dir = args.output_dir
    cfg.model_name = args.model_name
    cfg.image_size = args.image_size
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.valid_batch_size = args.valid_batch_size
    cfg.grad_accum = args.grad_accum
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.n_splits = args.n_splits
    cfg.lr_lora = args.lr_lora
    cfg.lr_head = args.lr_head
    cfg.weight_decay = args.weight_decay
    cfg.label_smoothing = args.label_smoothing
    cfg.early_stopping_patience = args.early_stopping_patience
    cfg.early_stopping_min_delta = args.early_stopping_min_delta
    cfg.scheduler = args.scheduler
    cfg.plateau_factor = args.plateau_factor
    cfg.plateau_patience = args.plateau_patience
    cfg.plateau_threshold = args.plateau_threshold
    cfg.min_lr = args.min_lr
    cfg.lora_r = args.lora_r
    cfg.lora_alpha = args.lora_alpha
    cfg.lora_dropout = args.lora_dropout
    cfg.lora_target_modules = args.lora_target_modules
    cfg.use_class_weights = args.use_class_weights
    cfg.class_weight_mode = args.class_weight_mode
    cfg.use_weighted_sampler = args.use_weighted_sampler
    cfg.sampler_weight_mode = args.sampler_weight_mode
    cfg.gradient_checkpointing = args.gradient_checkpointing
    cfg.multi_gpu = args.multi_gpu
    cfg.use_amp = not args.no_amp
    return cfg


def make_criterion(labels, cfg, device):
    weights = None
    if cfg.use_class_weights:
        weights_np = compute_class_weights(labels, cfg.num_classes, mode=cfg.class_weight_mode)
        print("Class counts:", np.bincount(labels, minlength=cfg.num_classes))
        print("Class weights:", weights_np)
        weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)


def get_current_lrs(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def format_lrs(lrs):
    return ", ".join(f"{lr:.2e}" for lr in lrs)


def maybe_wrap_multi_gpu(model, cfg, device):
    if device.type != "cuda":
        if cfg.multi_gpu:
            print("--multi-gpu was requested, but CUDA is not available. Training on CPU.")
        return model

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    num_visible = torch.cuda.device_count()
    print(f"Visible CUDA devices: {visible}")
    print(f"torch.cuda.device_count(): {num_visible}")

    if cfg.multi_gpu and num_visible > 1:
        print(f"Using DataParallel across {num_visible} visible GPU(s).")
        print("Note: --batch-size and --valid-batch-size are total batch sizes split across visible GPUs.")
        return nn.DataParallel(model)

    if cfg.multi_gpu and num_visible <= 1:
        print("--multi-gpu was requested, but only one CUDA device is visible. Training on single GPU.")

    return model


def build_scheduler(optimizer, train_loader, cfg):
    if cfg.scheduler == "cosine":
        num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
        total_training_steps = cfg.epochs * num_update_steps_per_epoch
        warmup_steps = int(cfg.warmup_ratio * total_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
        return scheduler, "batch"

    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            threshold=cfg.plateau_threshold,
            threshold_mode="abs",
            min_lr=cfg.min_lr,
        )
        return scheduler, "epoch"

    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def train_one_epoch(model, loader, optimizer, scheduler, scheduler_step_mode, scaler, criterion, cfg, device, epoch):
    model.train()
    running_loss = 0.0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels) / cfg.grad_accum

        scaler.scale(loss).backward()
        should_step = ((step + 1) % cfg.grad_accum == 0) or ((step + 1) == len(loader))
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler_step_mode == "batch":
                scheduler.step()

        bs = labels.size(0)
        running_loss += loss.item() * cfg.grad_accum * bs
        total += bs
        pbar.set_postfix(loss=running_loss / max(total, 1), lr=format_lrs(get_current_lrs(optimizer)))

    return running_loss / max(total, 1)


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, cfg, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Valid", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.use_amp and device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        bs = labels.size(0)
        running_loss += loss.item() * bs
        total += bs

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds = all_probs.argmax(axis=1)
    macro_f1 = f1_score(all_labels, preds, average="macro")
    per_class_f1 = f1_score(all_labels, preds, average=None, labels=list(range(cfg.num_classes)))
    return running_loss / max(total, 1), macro_f1, per_class_f1, all_probs, all_labels


def is_improved(current_score: float, best_score: float, min_delta: float) -> bool:
    return current_score > best_score + min_delta


def run_fold(fold, train_df, train_tfms, valid_tfms, cfg, device):
    print(f"\n========== Fold {fold} ==========")
    trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

    train_ds = WasteDataset(trn_df, transform=train_tfms, is_test=False)
    valid_ds = WasteDataset(val_df, transform=valid_tfms, is_test=False)

    if cfg.use_weighted_sampler:
        train_sampler = make_weighted_sampler(trn_df["label"].values, cfg.num_classes, cfg.sampler_weight_mode)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = Dinov3LoraClassifier(cfg).to(device)
    model = maybe_wrap_multi_gpu(model, cfg, device)
    if fold == 0:
        count_trainable_params(model)

    optimizer = get_optimizer(model, cfg)
    scheduler, scheduler_step_mode = build_scheduler(optimizer, train_loader, cfg)
    criterion = make_criterion(trn_df["label"].values, cfg, device)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    print(f"Scheduler: {cfg.scheduler}")
    if cfg.scheduler == "plateau":
        print(
            "ReduceLROnPlateau: monitor=valid_loss, "
            f"factor={cfg.plateau_factor}, patience={cfg.plateau_patience}, "
            f"threshold={cfg.plateau_threshold}, min_lr={cfg.min_lr}"
        )
    print(f"Initial LRs: {format_lrs(get_current_lrs(optimizer))}")

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    ckpt_path = cfg.output_dir / f"fold{fold}_best.pt"
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scheduler_step_mode,
            scaler,
            criterion,
            cfg,
            device,
            epoch,
        )
        valid_loss, valid_f1, per_class_f1, val_probs, val_labels = valid_one_epoch(model, valid_loader, criterion, cfg, device)

        lr_before = get_current_lrs(optimizer)
        if scheduler_step_mode == "epoch":
            scheduler.step(valid_loss)
        lr_after = get_current_lrs(optimizer)

        print(
            f"Fold {fold} | Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.5f} | valid_loss={valid_loss:.5f} | macro_f1={valid_f1:.5f} | "
            f"F1 Rec={per_class_f1[0]:.4f} | F1 Elec={per_class_f1[1]:.4f} | F1 Org={per_class_f1[2]:.4f} | "
            f"LR={format_lrs(lr_after)}"
        )
        if lr_after != lr_before:
            print(f"LR reduced: {format_lrs(lr_before)} -> {format_lrs(lr_after)}")

        improved = is_improved(valid_f1, best_f1, cfg.early_stopping_min_delta)
        if improved:
            best_f1 = valid_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "fold": fold,
                "best_epoch": best_epoch,
                "best_f1": best_f1,
                "model": get_trainable_state_dict(model),
                "lora_target_modules": cfg.lora_target_modules,
                "image_size": cfg.image_size,
                "scheduler": cfg.scheduler,
                "multi_gpu": cfg.multi_gpu,
                "early_stopping_patience": cfg.early_stopping_patience,
                "early_stopping_min_delta": cfg.early_stopping_min_delta,
            }, ckpt_path)
            print(f"Saved best fold {fold}: macro_f1={best_f1:.5f}")
        else:
            epochs_without_improvement += 1
            print(
                f"No Macro-F1 improvement for {epochs_without_improvement}/"
                f"{cfg.early_stopping_patience} epochs "
                f"(min_delta={cfg.early_stopping_min_delta})."
            )

        history.append({
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "macro_f1": valid_f1,
            "f1_recyclable": float(per_class_f1[0]),
            "f1_electronic": float(per_class_f1[1]),
            "f1_organic": float(per_class_f1[2]),
            "lr_lora": lr_after[0],
            "lr_head": lr_after[1] if len(lr_after) > 1 else lr_after[0],
            "epochs_without_improvement": epochs_without_improvement,
        })

        if cfg.early_stopping_patience > 0 and epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping fold {fold} at epoch {epoch}. "
                f"Best epoch={best_epoch}, best Macro-F1={best_f1:.5f}."
            )
            break

    pd.DataFrame(history).to_csv(cfg.output_dir / f"fold{fold}_history.csv", index=False)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=False)
    valid_loss, valid_f1, per_class_f1, val_probs, val_labels = valid_one_epoch(model, valid_loader, criterion, cfg, device)
    val_indices = val_df["original_index"].values

    print(f"Best Fold {fold}: epoch={best_epoch}, macro_f1={best_f1:.5f}")
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return val_indices, val_probs, val_labels, best_f1


def main():
    args = parse_args()
    cfg = cfg_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    device = get_device()
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: You are on CPU. DINOv3 ViT-L training will be very slow. Use a GPU runtime.")

    train_tfms, valid_tfms, _, _ = get_transforms(cfg.model_name, cfg.image_size, cfg.hf_token)
    train_df = build_train_df(cfg.train_dir, cfg.label2id, cfg.seed)
    train_df["original_index"] = np.arange(len(train_df))

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        train_df.loc[val_idx, "fold"] = fold

    train_df.to_csv(cfg.output_dir / "train_folds.csv", index=False)

    oof_probs = np.zeros((len(train_df), cfg.num_classes), dtype=np.float32)
    oof_labels = train_df["label"].values.copy()
    fold_scores = []

    for fold in range(cfg.n_splits):
        val_indices, val_probs, val_labels, best_f1 = run_fold(fold, train_df, train_tfms, valid_tfms, cfg, device)
        oof_probs[val_indices] = val_probs
        fold_scores.append(best_f1)

    macro_f1, per_class_f1, report, cm = macro_f1_and_report(oof_labels, oof_probs, cfg.id2label)
    print("\n========== OOF RESULT ==========")
    print("Fold scores:", fold_scores)
    print("Mean fold F1:", np.mean(fold_scores))
    print("OOF Macro F1:", macro_f1)
    print("OOF per-class F1:", per_class_f1)
    print("\nClassification report:\n", report)
    print("\nConfusion matrix:\n", cm)

    np.save(cfg.output_dir / "oof_probs.npy", oof_probs)
    oof_df = train_df.copy()
    oof_df["oof_pred"] = oof_probs.argmax(axis=1)
    for c in range(cfg.num_classes):
        oof_df[f"prob_{cfg.id2label[c]}"] = oof_probs[:, c]
    oof_df.to_csv(cfg.output_dir / "oof_predictions.csv", index=False)

    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in cfg.__dict__.items()
        if not k.startswith("_") and k != "hf_token"
    }
    with open(cfg.output_dir / "config_used.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)


if __name__ == "__main__":
    main()
