"""Tests for training-side code: data utils, loss, eval, naming."""

import numpy as np
import pytest
import torch

from dinov3_in1k_probes.data import (
    extract_val_num,
    load_class_names,
    load_real_labels,
    real_accuracy,
)
from dinov3_in1k_probes.repos import model_name_from_repo
from dinov3_in1k_probes.training.eval import evaluate
from dinov3_in1k_probes.training.loss import sigmoid_loss, softmax_loss


def test_real_labels_and_class_names():
    labels = load_real_labels()
    names = load_class_names()
    assert len(labels) == 50_000 and len(names) == 1000


def test_extract_val_num():
    assert extract_val_num("n01440764/ILSVRC2012_val_00000293.JPEG") == 293


def test_real_accuracy():
    real_labels = load_real_labels()
    fnames = [f"n0/ILSVRC2012_val_{i+1:08d}.JPEG" for i in range(50_000)]

    perfect = np.array([ls[0] if ls else 0 for ls in real_labels], dtype=np.intp)
    assert real_accuracy(perfect, fnames, real_labels) > 0.99

    garbage = np.full(50_000, 9999, dtype=np.intp)
    assert real_accuracy(garbage, fnames, real_labels) == 0.0


def test_softmax_and_sigmoid_loss():
    logits = torch.randn(8, 1000)
    labels = torch.randint(0, 1000, (8,))
    assert softmax_loss(logits, labels).item() > 0
    assert sigmoid_loss(logits, labels).item() > 0


def test_evaluate_overfit():
    torch.manual_seed(42)
    N, D, C = 16, 32, 10
    features = torch.randn(N, D)
    labels = torch.randint(0, C, (N,))

    model = torch.nn.Linear(D, C)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        softmax_loss(model(features), labels).backward()
        opt.step()

    result = evaluate(model, features=features, labels=labels, batch_size=8, loss_fn=softmax_loss)
    assert result.top1 == 1.0
    assert result.predictions.shape == (N,)


def test_model_name_from_repo():
    assert model_name_from_repo("facebook/dinov3-vitb16-pretrain-lvd1689m") == "dinov3_vitb16"
    assert model_name_from_repo("facebook/dinov3-vits16-pretrain-lvd1689m") == "dinov3_vits16"
    assert model_name_from_repo("facebook/dinov3-vith16plus-pretrain-lvd1689m") == "dinov3_vith16plus"

    with pytest.raises(AssertionError, match="'-pretrain'"):
        model_name_from_repo("some/random-model-name")

    with pytest.raises(AssertionError, match="doesn't look like"):
        model_name_from_repo("facebook/resnet50-pretrain-imagenet")
