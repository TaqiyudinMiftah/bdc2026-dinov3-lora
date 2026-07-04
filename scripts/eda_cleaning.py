import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, ImageOps
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

from bdc2026.config import TrainConfig
from bdc2026.utils import list_image_files, seed_everything, get_device

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self):
        out = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return [v for v in out.values() if len(v) > 1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./eda_outputs"))
    parser.add_argument("--model-name", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-class", type=int, default=16)
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--aspect-low", type=float, default=0.35)
    parser.add_argument("--aspect-high", type=float, default=2.80)
    parser.add_argument("--min-side", type=int, default=80)

    parser.add_argument("--use-dino-duplicates", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-neighbors", type=int, default=6)
    parser.add_argument("--embedding-sim-threshold", type=float, default=0.985)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))

    parser.add_argument("--make-clean-copy", action="store_true")
    parser.add_argument("--clean-output", type=Path, default=Path("./BDC2026_clean"))
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    return parser.parse_args()


def build_manifest(cfg: TrainConfig):
    rows = []
    for class_name, label in cfg.label2id.items():
        class_dir = cfg.train_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in list_image_files(class_dir):
            rows.append({"path": str(path), "class_name": class_name, "label": label})
    return pd.DataFrame(rows)


def safe_image_info(path: str):
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size
            mode = img.mode
            img.verify()
        return {
            "is_valid": True,
            "width": width,
            "height": height,
            "mode": mode,
            "error": "",
        }
    except Exception as e:
        return {
            "is_valid": False,
            "width": np.nan,
            "height": np.nan,
            "mode": "",
            "error": repr(e),
        }


def file_md5(path: str, chunk_size: int = 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: str):
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            return str(imagehash.phash(img))
    except Exception:
        return None


def save_class_distribution(df, output_dir):
    counts = df["class_name"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)


def save_numeric_hist(df, column, output_dir, title):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / f"{column}_hist.png", dpi=160)
    plt.close(fig)


def save_sample_grids(df, output_dir, sample_per_class, seed):
    rng = np.random.default_rng(seed)
    for class_name, sub in df[df["is_valid"]].groupby("class_name"):
        n = min(sample_per_class, len(sub))
        if n == 0:
            continue
        sample_idx = rng.choice(sub.index.values, size=n, replace=False)
        sample = sub.loc[sample_idx]
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).reshape(-1)

        for ax in axes:
            ax.axis("off")

        for ax, (_, row) in zip(axes, sample.iterrows()):
            try:
                with Image.open(row["path"]) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(Path(row["path"]).name, fontsize=7)
                    ax.axis("off")
            except Exception:
                ax.axis("off")

        fig.suptitle(f"Sample images: {class_name}")
        fig.tight_layout()
        safe_name = class_name.replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"samples_{safe_name}.png", dpi=160)
        plt.close(fig)


def exact_duplicate_groups(df):
    rows = []
    valid = df[df["is_valid"]].copy()
    md5s = []
    for path in tqdm(valid["path"], desc="MD5 exact hashes"):
        try:
            md5s.append(file_md5(path))
        except Exception:
            md5s.append(None)
    valid["md5"] = md5s

    group_id = 0
    for md5, group in valid.dropna(subset=["md5"]).groupby("md5"):
        if len(group) <= 1:
            continue
        labels = sorted(group["label"].unique().tolist())
        for _, row in group.iterrows():
            rows.append({
                "duplicate_type": "exact_md5",
                "group_id": group_id,
                "path": row["path"],
                "class_name": row["class_name"],
                "label": row["label"],
                "signature": md5,
                "cross_label": len(labels) > 1,
            })
        group_id += 1
    return pd.DataFrame(rows), valid[["path", "md5"]]


def phash_duplicate_pairs(df, threshold):
    valid = df[df["is_valid"]].copy()
    phashes = []
    for path in tqdm(valid["path"], desc="pHash"):
        phashes.append(compute_phash(path))
    valid["phash"] = phashes
    valid = valid.dropna(subset=["phash"]).reset_index(drop=True)

    # Blocking by first hex characters keeps this fast. DINO embedding duplicates are more reliable for broad near-duplicate search.
    blocks = defaultdict(list)
    for i, h in enumerate(valid["phash"]):
        blocks[h[:3]].append(i)

    pairs = []
    seen = set()
    for block_indices in tqdm(blocks.values(), desc="pHash near pairs"):
        if len(block_indices) < 2:
            continue
        for a_pos in range(len(block_indices)):
            for b_pos in range(a_pos + 1, len(block_indices)):
                i = block_indices[a_pos]
                j = block_indices[b_pos]
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                dist = imagehash.hex_to_hash(valid.loc[i, "phash"]) - imagehash.hex_to_hash(valid.loc[j, "phash"])
                if dist <= threshold:
                    pairs.append({
                        "path_a": valid.loc[i, "path"],
                        "class_a": valid.loc[i, "class_name"],
                        "label_a": int(valid.loc[i, "label"]),
                        "path_b": valid.loc[j, "path"],
                        "class_b": valid.loc[j, "class_name"],
                        "label_b": int(valid.loc[j, "label"]),
                        "phash_distance": int(dist),
                        "cross_label": int(valid.loc[i, "label"]) != int(valid.loc[j, "label"]),
                    })
    return pd.DataFrame(pairs), valid[["path", "phash"]]


@torch.no_grad()
def compute_dino_embeddings(df, model_name, batch_size, hf_token, device):
    processor = AutoImageProcessor.from_pretrained(model_name, token=hf_token)
    model = AutoModel.from_pretrained(model_name, token=hf_token).to(device)
    model.eval()

    paths = df["path"].tolist()
    embeddings = []

    for start in tqdm(range(0, len(paths), batch_size), desc="DINO embeddings"):
        batch_paths = paths[start:start + batch_size]
        images = []
        keep = []
        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    images.append(ImageOps.exif_transpose(img).convert("RGB"))
                    keep.append(p)
            except Exception:
                pass

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state[:, 0]
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
        embeddings.append(feats.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def dino_duplicate_pairs(df, embeddings, n_neighbors, sim_threshold):
    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(df)),
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    pairs = []
    seen = set()
    for i in tqdm(range(len(df)), desc="DINO duplicate pairs"):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            sim = 1.0 - float(dist)
            if sim < sim_threshold:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "path_a": df.loc[i, "path"],
                "class_a": df.loc[i, "class_name"],
                "label_a": int(df.loc[i, "label"]),
                "path_b": df.loc[j, "path"],
                "class_b": df.loc[j, "class_name"],
                "label_b": int(df.loc[j, "label"]),
                "cosine_similarity": sim,
                "cross_label": int(df.loc[i, "label"]) != int(df.loc[j, "label"]),
            })
    return pd.DataFrame(pairs)


def duplicate_groups_from_pairs(df, pairs_df, path_col_a="path_a", path_col_b="path_b"):
    if pairs_df.empty:
        return []
    path_to_idx = {p: i for i, p in enumerate(df["path"].tolist())}
    uf = UnionFind(len(df))
    for _, row in pairs_df.iterrows():
        a = path_to_idx.get(row[path_col_a])
        b = path_to_idx.get(row[path_col_b])
        if a is not None and b is not None:
            uf.union(a, b)
    return uf.groups()


def build_cleaning_candidates(df, exact_df, phash_pairs, dino_pairs, args):
    rows = []

    for _, row in df[~df["is_valid"]].iterrows():
        rows.append({
            "path": row["path"],
            "class_name": row["class_name"],
            "label": row["label"],
            "reason": "corrupt_or_unreadable",
            "recommended_action": "remove_or_repair",
            "detail": row.get("error", ""),
        })

    for _, row in df[df["is_valid"]].iterrows():
        reasons = []
        if min(row["width"], row["height"]) < args.min_side:
            reasons.append(f"small_image_min_side_lt_{args.min_side}")
        if row["aspect_ratio"] < args.aspect_low or row["aspect_ratio"] > args.aspect_high:
            reasons.append("extreme_aspect_ratio")
        for reason in reasons:
            rows.append({
                "path": row["path"],
                "class_name": row["class_name"],
                "label": row["label"],
                "reason": reason,
                "recommended_action": "manual_review",
                "detail": f"width={row['width']}, height={row['height']}, aspect_ratio={row['aspect_ratio']:.4f}",
            })

    if not exact_df.empty:
        grouped = exact_df.groupby("group_id")
        for group_id, group in grouped:
            group = group.reset_index(drop=True)
            cross_label = bool(group["cross_label"].iloc[0])
            for pos, row in group.iterrows():
                action = "manual_review_cross_label_duplicate" if cross_label else ("keep_first" if pos == 0 else "remove_duplicate")
                rows.append({
                    "path": row["path"],
                    "class_name": row["class_name"],
                    "label": row["label"],
                    "reason": "exact_md5_duplicate",
                    "recommended_action": action,
                    "detail": f"group_id={group_id}, cross_label={cross_label}",
                })

    for source_name, pairs in [("phash_near_duplicate", phash_pairs), ("dino_embedding_duplicate", dino_pairs)]:
        if pairs is None or pairs.empty:
            continue
        for _, row in pairs.iterrows():
            action = "manual_review_cross_label_duplicate" if row["cross_label"] else "manual_review_or_remove_one"
            rows.append({
                "path": row["path_a"],
                "class_name": row["class_a"],
                "label": row["label_a"],
                "reason": source_name,
                "recommended_action": action,
                "detail": f"paired_with={row['path_b']}",
            })
            rows.append({
                "path": row["path_b"],
                "class_name": row["class_b"],
                "label": row["label_b"],
                "reason": source_name,
                "recommended_action": action,
                "detail": f"paired_with={row['path_a']}",
            })

    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=[
        "path", "class_name", "label", "reason", "recommended_action", "detail"
    ])


def make_clean_copy(df, cleaning_candidates, output_root, copy_mode):
    output_root.mkdir(parents=True, exist_ok=True)
    remove_paths = set(
        cleaning_candidates[
            cleaning_candidates["recommended_action"].isin(["remove_or_repair", "remove_duplicate"])
        ]["path"].tolist()
    )

    kept = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating clean copy"):
        src = Path(row["path"])
        if str(src) in remove_paths:
            continue
        rel = Path(row["class_name"]) / src.name
        dst = output_root / "train" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if copy_mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
        kept += 1

    print(f"Clean copy created at {output_root}. Kept {kept} images.")


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(data_root=args.data_root)
    cfg.hf_token = args.hf_token
    cfg.model_name = args.model_name

    print("Building manifest...")
    df = build_manifest(cfg)

    print("Reading image metadata and checking corrupt files...")
    infos = [safe_image_info(p) for p in tqdm(df["path"], desc="Image checks")]
    info_df = pd.DataFrame(infos)
    df = pd.concat([df, info_df], axis=1)
    df["aspect_ratio"] = df["width"] / df["height"]
    df["min_side"] = df[["width", "height"]].min(axis=1)
    df["file_size_bytes"] = df["path"].map(lambda p: Path(p).stat().st_size if Path(p).exists() else np.nan)

    df.to_csv(args.output_dir / "train_manifest.csv", index=False)
    df[~df["is_valid"]].to_csv(args.output_dir / "corrupted_images.csv", index=False)

    print("Saving EDA plots...")
    save_class_distribution(df, figures_dir)
    save_numeric_hist(df, "width", figures_dir, "Image width distribution")
    save_numeric_hist(df, "height", figures_dir, "Image height distribution")
    save_numeric_hist(df, "aspect_ratio", figures_dir, "Aspect ratio distribution")
    save_numeric_hist(df, "file_size_bytes", figures_dir, "File size distribution")
    save_sample_grids(df, figures_dir, args.sample_per_class, args.seed)

    print("Detecting exact duplicates...")
    exact_df, md5_df = exact_duplicate_groups(df)
    exact_df.to_csv(args.output_dir / "exact_duplicate_groups.csv", index=False)
    md5_df.to_csv(args.output_dir / "image_md5.csv", index=False)

    print("Detecting perceptual hash near duplicates...")
    phash_pairs, phash_df = phash_duplicate_pairs(df, args.phash_threshold)
    phash_pairs.to_csv(args.output_dir / "phash_duplicate_pairs.csv", index=False)
    phash_df.to_csv(args.output_dir / "image_phash.csv", index=False)

    dino_pairs = pd.DataFrame()
    if args.use_dino_duplicates:
        print("Detecting DINO embedding duplicates...")
        device = get_device()
        valid_df = df[df["is_valid"]].reset_index(drop=True)
        embeddings = compute_dino_embeddings(
            valid_df,
            model_name=args.model_name,
            batch_size=args.embedding_batch_size,
            hf_token=args.hf_token,
            device=device,
        )
        np.save(args.output_dir / "dino_embeddings.npy", embeddings)
        valid_df[["path", "class_name", "label"]].to_csv(args.output_dir / "dino_embedding_index.csv", index=False)
        dino_pairs = dino_duplicate_pairs(
            valid_df,
            embeddings,
            n_neighbors=args.embedding_neighbors,
            sim_threshold=args.embedding_sim_threshold,
        )
        dino_pairs.to_csv(args.output_dir / "dino_duplicate_pairs.csv", index=False)
    else:
        print("Skipping DINO embedding duplicate detection. Add --use-dino-duplicates to enable it.")

    print("Building cleaning candidate report...")
    cleaning_candidates = build_cleaning_candidates(df, exact_df, phash_pairs, dino_pairs, args)
    cleaning_candidates.to_csv(args.output_dir / "cleaning_candidates.csv", index=False)

    summary = {
        "num_images": int(len(df)),
        "num_valid": int(df["is_valid"].sum()),
        "num_corrupt": int((~df["is_valid"]).sum()),
        "class_counts": df["class_name"].value_counts().to_dict(),
        "exact_duplicate_rows": int(len(exact_df)),
        "phash_duplicate_pairs": int(len(phash_pairs)),
        "dino_duplicate_pairs": int(len(dino_pairs)),
        "cleaning_candidate_rows": int(len(cleaning_candidates)),
        "notes": [
            "Do not delete the original dataset directly.",
            "Cross-label duplicates should be manually reviewed, not automatically removed.",
            "DINO embedding duplicates are semantic near-duplicates; inspect examples before removing them.",
        ],
    }
    with open(args.output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.make_clean_copy:
        make_clean_copy(df, cleaning_candidates, args.clean_output, args.copy_mode)

    print("EDA and cleaning pipeline finished.")
    print("Outputs saved to:", args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
