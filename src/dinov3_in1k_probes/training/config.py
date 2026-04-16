"""Configuration for DINOv3 IN1K CLS probe extraction and training."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _env_path(var: str, fallback: str | None = None) -> Path:
    val = os.environ.get(var, fallback)
    if val is None:
        raise ValueError(f"${var} not set")
    return Path(val)


Objective = Literal["softmax", "sigmoid"]

# DINOv3 model repos on HuggingFace.
DINOV3_REPOS: dict[str, str] = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


@dataclass
class ExtractionConfig:
    """CLS token extraction from a frozen DINOv3 backbone."""

    model_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
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

    model_repo: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 512

    max_batch_size: int = 2048
    outer_epochs: int = 2
    objective: Objective = "sigmoid"
    n_trials: int = 100

    features_dir: Path = field(default_factory=lambda: _env_path("FEATURES_DIR"))
    checkpoint_dir: Path = field(default_factory=lambda: _env_path("CHECKPOINTS_DIR", "checkpoints"))

    comet_project: str = "dv3-in1k-linear-probing"
    comet_workspace: str = "m2b3-ava"
