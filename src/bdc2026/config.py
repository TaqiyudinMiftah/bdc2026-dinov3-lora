from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv

# Load local .env automatically when present.
# This keeps secrets out of Git while allowing HF_TOKEN and other config values locally.
load_dotenv()


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path = Path(os.environ.get("BDC2026_OUTPUT_DIR", "./outputs_dinov3_lora"))
    model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    hf_token: str | None = os.environ.get("HF_TOKEN")

    seed: int = 42
    n_splits: int = 5
    num_classes: int = 3

    image_size: int = 224
    epochs: int = 20
    batch_size: int = 4
    valid_batch_size: int = 8
    grad_accum: int = 4
    num_workers: int = 2

    lr_lora: float = 1e-4
    lr_head: float = 7e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05

    # Early stopping monitors validation Macro-F1.
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4

    # LR scheduler. Default uses validation loss plateau to reduce LR.
    scheduler: str = "plateau"  # plateau or cosine
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_threshold: float = 1e-4
    min_lr: float = 1e-7

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    dropout: float = 0.1
    gradient_checkpointing: bool = False

    use_amp: bool = True
    use_class_weights: bool = True
    class_weight_mode: str = "inverse"  # inverse, sqrt_inverse, effective
    use_weighted_sampler: bool = False
    sampler_weight_mode: str = "sqrt_inverse"

    label2id: dict = field(default_factory=lambda: {
        "0_Recyclable": 0,
        "1_Electronic": 1,
        "2_Organic": 2,
    })

    @property
    def id2label(self) -> dict:
        return {v: k for k, v in self.label2id.items()}

    @property
    def train_dir(self) -> Path:
        return self.data_root / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_root / "test"

    @property
    def template_path(self) -> Path:
        return self.data_root / "submission.csv"
