import argparse
import gc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bdc2026.config import TrainConfig
from bdc2026.dataset import WasteDataset, build_test_df, get_transforms
from bdc2026.model import Dinov3LoraClassifier
from bdc2026.utils import seed_everything, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission_NamaTim.csv"))
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--valid-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def cfg_from_args(args):
    cfg = TrainConfig(data_root=args.data_root)
    cfg.model_name = args.model_name
    cfg.image_size = args.image_size
    cfg.n_splits = args.n_splits
    cfg.valid_batch_size = args.valid_batch_size
    cfg.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.lora_r = args.lora_r
    cfg.lora_alpha = args.lora_alpha
    cfg.lora_dropout = args.lora_dropout
    cfg.lora_target_modules = args.lora_target_modules
    cfg.gradient_checkpointing = args.gradient_checkpointing
    cfg.use_amp = not args.no_amp
    return cfg


@torch.no_grad()
def predict_proba(model, loader, cfg, device):
    model.eval()
    all_probs = []
    all_indices = []

    for images, indices in tqdm(loader, desc="Predict", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(cfg.use_amp and device.type == "cuda"),
        ):
            logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu().numpy())
        all_indices.append(indices.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    ordered = np.zeros_like(all_probs)
    ordered[all_indices] = all_probs
    return ordered


def predict_one_fold(fold, test_df, test_tfms_list, cfg, checkpoint_dir, device):
    ckpt_path = checkpoint_dir / f"fold{fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"\nPredicting fold {fold}: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "lora_target_modules" in checkpoint:
        cfg.lora_target_modules = checkpoint["lora_target_modules"]
    if "image_size" in checkpoint:
        cfg.image_size = checkpoint["image_size"]

    model = Dinov3LoraClassifier(cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    fold_probs = np.zeros((len(test_df), cfg.num_classes), dtype=np.float32)
    for i, tfm in enumerate(test_tfms_list):
        print(f"Fold {fold} | TTA {i + 1}/{len(test_tfms_list)}")
        ds = WasteDataset(test_df, transform=tfm, is_test=True)
        loader = DataLoader(
            ds,
            batch_size=cfg.valid_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        fold_probs += predict_proba(model, loader, cfg, device) / len(test_tfms_list)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return fold_probs


def main():
    args = parse_args()
    cfg = cfg_from_args(args)
    seed_everything(cfg.seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    device = get_device()
    print("Device:", device)
    if device.type == "cpu":
        print("WARNING: You are on CPU. Prediction will be slow with DINOv3 ViT-L.")

    _, _, test_tfms, test_flip_tfms = get_transforms(cfg.model_name, cfg.image_size, cfg.hf_token)
    test_tfms_list = [test_tfms, test_flip_tfms] if args.tta else [test_tfms]

    test_df = build_test_df(cfg.test_dir)
    test_probs = np.zeros((len(test_df), cfg.num_classes), dtype=np.float32)

    for fold in range(cfg.n_splits):
        fold_probs = predict_one_fold(fold, test_df, test_tfms_list, cfg, args.checkpoint_dir, device)
        test_probs += fold_probs / cfg.n_splits

    test_preds = test_probs.argmax(axis=1).astype(int)

    template = pd.read_csv(cfg.template_path)
    if len(template) != len(test_preds):
        raise ValueError(f"Template rows={len(template)} but predictions={len(test_preds)}")
    if "predicted" not in template.columns:
        raise ValueError("submission.csv template must contain a 'predicted' column")

    template["predicted"] = test_preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(args.output, index=False)

    debug_df = test_df.copy()
    debug_df["predicted"] = test_preds
    for c in range(cfg.num_classes):
        debug_df[f"prob_{cfg.id2label[c]}"] = test_probs[:, c]
    debug_df.to_csv(args.output.parent / "test_predictions_debug.csv", index=False)
    np.save(args.output.parent / "test_probs.npy", test_probs)

    print("Saved submission:", args.output)
    print("Submission distribution:")
    print(template["predicted"].value_counts().sort_index())


if __name__ == "__main__":
    main()
