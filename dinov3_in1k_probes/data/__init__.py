"""ImageNet-1k constants, class labels, preprocessing, and ImageNet-ReAL evaluation."""

import json
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from torchvision import transforms

NUM_CLASSES = 1000
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_val_transform(image_size: int) -> transforms.Compose:
    """Resize shortest side → center crop → normalize. Matches DALI val pipeline."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

_DATA_DIR = Path(__file__).resolve().parent / "in1k"
_REAL_LABELS_PATH = _DATA_DIR / "real.json"
_CLASSES_PATH = _DATA_DIR / "classes.json"

_VAL_NUM_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG")


def load_class_names() -> list[str]:
    with open(_CLASSES_PATH) as f:
        names = json.load(f)
    assert len(names) == NUM_CLASSES
    return names


def load_real_labels() -> list[list[int]]:
    """Load ImageNet-ReAL multi-labels. Index i → valid class IDs for val image i+1."""
    with open(_REAL_LABELS_PATH) as f:
        return json.load(f)


def extract_val_num(filename: str) -> int:
    """'n01440764/ILSVRC2012_val_00000293.JPEG' → 293."""
    m = _VAL_NUM_RE.search(filename)
    assert m, f"Not an IN1k val filename: {filename}"
    return int(m.group(1))


def real_accuracy(predictions: NDArray[np.intp], filenames: list[str], real_labels: list[list[int]]) -> float:
    """ImageNet-ReAL top-1 accuracy for predictions not in standard val order."""
    correct = []
    for pred, fn in zip(predictions, filenames):
        valid = real_labels[extract_val_num(fn) - 1]
        if valid:
            correct.append(int(pred) in valid)
    return float(np.mean(correct)) if correct else 0.0
