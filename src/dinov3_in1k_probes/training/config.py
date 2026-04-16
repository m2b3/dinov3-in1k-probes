"""Configuration for DINOv3 IN1K CLS probe extraction and training."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dinov3_in1k_probes.repos import VARIANTS, dinov3_backbone_repo


def _env_path(var: str, fallback: str | None = None) -> Path:
    val = os.environ.get(var, fallback)
    if val is None:
        raise ValueError(f"${var} not set")
    return Path(val)


Objective = Literal["softmax", "sigmoid"]

# All DINOv3 backbones available for extraction (includes 7B, which has no probe).
DINOV3_REPOS: dict[str, str] = {
    v: dinov3_backbone_repo(v) for v in VARIANTS
} | {"vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m"}


_DEFAULT_BACKBONE = dinov3_backbone_repo("vitb16")


@dataclass
class ExtractionConfig:
    """CLS token extraction from a frozen DINOv3 backbone."""

    model_repo: str = _DEFAULT_BACKBONE
    image_size: int = 512
    split: Literal["train", "val"] = "val"

    batch_size: int = 256
    checkpoint_every_n_batches: int = 1000
    max_batches: int | None = None
    dry_run: bool = False
    accumulation_device: str = "cuda"

    imagenet_root: Path = field(default_factory=lambda: _env_path("IMAGENET_ROOT"))
    features_dir: Path = field(default_factory=lambda: _env_path("FEATURES_DIR"))


@dataclass
class TrainConfig:
    """Optuna-based linear probe training on pre-extracted CLS tokens."""

    model_repo: str = _DEFAULT_BACKBONE
    image_size: int = 512

    max_batch_size: int = 2048
    outer_epochs: int = 2
    objective: Objective = "sigmoid"
    n_trials: int = 100

    features_dir: Path = field(default_factory=lambda: _env_path("FEATURES_DIR"))
    checkpoint_dir: Path = field(default_factory=lambda: _env_path("CHECKPOINTS_DIR", "checkpoints"))

    comet_project: str = "dv3-in1k-linear-probing"
    comet_workspace: str = field(default_factory=lambda: os.environ.get("COMET_WORKSPACE", ""))
