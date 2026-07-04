from pathlib import Path

import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

from .utils import list_image_files

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WasteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.is_test:
            return image, idx

        return image, int(row["label"])


def build_train_df(train_dir: Path, label2id: dict, seed: int):
    rows = []
    for folder_name, label_id in label2id.items():
        class_dir = train_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        for path in list_image_files(class_dir):
            rows.append({
                "path": str(path),
                "label": int(label_id),
                "class_name": folder_name,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print("Train size:", len(df))
    print(df["class_name"].value_counts())
    return df


def build_test_df(test_dir: Path):
    image_paths = list_image_files(test_dir)
    df = pd.DataFrame({"path": [str(p) for p in image_paths]})
    print("Test size:", len(df))
    return df


def get_transforms(model_name: str, image_size: int, hf_token=None):
    processor = AutoImageProcessor.from_pretrained(model_name, token=hf_token)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=mean, std=std)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.70, 1.00),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            )
        ], p=0.5),
        transforms.RandomRotation(
            degrees=10,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize(
            int(image_size * 1.15),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    flip_tfms = transforms.Compose([
        transforms.Resize(
            int(image_size * 1.15),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.Lambda(lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tfms, valid_tfms, valid_tfms, flip_tfms
