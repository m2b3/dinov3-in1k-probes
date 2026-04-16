"""Loss functions for IN1K classification probe training."""

from typing import Callable, Literal

import torch.nn.functional as F
from torch import Tensor

from dinov3_in1k_probes.data import NUM_CLASSES

LossName = Literal["softmax", "sigmoid"]
LossFn = Callable[..., Tensor]


def softmax_loss(logits: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    return F.cross_entropy(logits, labels.long(), reduction=reduction)


def sigmoid_loss(logits: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    """Sigmoid CE treating each class as independent (Beyer et al. 2020)."""
    targets = F.one_hot(labels.long(), NUM_CLASSES).float()
    return F.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


_REGISTRY: dict[LossName, LossFn] = {"softmax": softmax_loss, "sigmoid": sigmoid_loss}


def get_loss_fn(name: LossName) -> LossFn:
    return _REGISTRY[name]
