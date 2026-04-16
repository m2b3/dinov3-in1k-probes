"""Optuna-based linear probe training on pre-extracted DINOv3 CLS tokens.

Phase 2 of the two-phase pipeline: load cached features into GPU memory,
then run Optuna HP search over {optimizer, LR, batch_size, WD, betas}.
"""

import logging
import math
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple

import comet_ml
import optuna
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from dinov3_in1k_probes.data import (
    NUM_CLASSES,
    load_class_names,
    load_real_labels,
    real_accuracy,
)
from dinov3_in1k_probes.repos import model_name_from_repo
from dinov3_in1k_probes.training.config import TrainConfig
from dinov3_in1k_probes.training.eval import evaluate
from dinov3_in1k_probes.training.extract import CLS_SUFFIX, FILENAMES_SUFFIX, LABELS_SUFFIX
from dinov3_in1k_probes.training.loss import get_loss_fn

log = logging.getLogger(__name__)

WARMUP_FRACTION = 0.1
BATCH_SIZE_MIN_DIVISOR = 4


# --- Types ---


class _FullResults(NamedTuple):
    loss: float
    top1: float
    top5: float
    real_top1: float


# --- Data loading ---


def _split_dir(cfg: TrainConfig, split: str) -> Path:
    return cfg.features_dir / model_name_from_repo(cfg.model_repo) / str(cfg.image_size) / split


def load_train_features(cfg: TrainConfig) -> tuple[list[Tensor], list[Tensor], int]:
    """Load all pre-extracted train feature files. Returns (cls_list, labels_list, embed_dim)."""
    d = _split_dir(cfg, "train")
    assert d.is_dir(), f"Train dir not found: {d}"

    files = sorted(d.glob(f"*{CLS_SUFFIX}"))
    assert files, f"No *{CLS_SUFFIX} in {d}"

    cls_list, labels_list = [], []
    embed_dim: int | None = None

    for cls_file in tqdm(files, desc="Loading train features"):
        lbl_file = cls_file.parent / cls_file.name.replace(CLS_SUFFIX, LABELS_SUFFIX)
        assert lbl_file.exists(), f"Missing: {lbl_file}"

        cls = torch.load(cls_file, weights_only=True)
        labels = torch.load(lbl_file, weights_only=True).squeeze().to(torch.int32)
        N, D = cls.shape
        assert labels.shape == (N,)

        if embed_dim is None:
            embed_dim = D
        assert D == embed_dim, f"Inconsistent embed_dim: {embed_dim} vs {D}"

        cls_list.append(cls)
        labels_list.append(labels)

    assert embed_dim is not None
    total = sum(t.shape[0] for t in cls_list)
    gb = sum(t.nbytes for t in cls_list) / 1e9
    log.info("Loaded %d train files: %d samples, embed_dim=%d, %.2f GB", len(cls_list), total, embed_dim, gb)
    return cls_list, labels_list, embed_dim


def load_val_features(cfg: TrainConfig, embed_dim: int) -> tuple[Tensor, Tensor, list[str]]:
    d = _split_dir(cfg, "val")
    cls = torch.load(d / f"features{CLS_SUFFIX}", weights_only=True)
    labels = torch.load(d / f"features{LABELS_SUFFIX}", weights_only=True).squeeze().to(torch.int32)
    filenames = torch.load(d / f"features{FILENAMES_SUFFIX}", weights_only=False)

    N, D = cls.shape
    assert D == embed_dim, f"Val embed_dim {D} != train {embed_dim}"
    assert labels.shape == (N,) and len(filenames) == N
    log.info("Val: %d samples, embed_dim=%d", N, embed_dim)
    return cls, labels, filenames


# --- Compiled forward-backward ---


def _forward_backward(
    model: nn.Module, features: Tensor, labels: Tensor, *, loss_fn: Callable,
) -> tuple[Tensor, Tensor, Tensor]:
    logits = model(features)
    loss = loss_fn(logits, labels, reduction="mean")
    loss.backward()
    top1 = (logits.argmax(1) == labels.long()).sum()
    top5 = (logits.topk(5, 1).indices == labels.long().unsqueeze(1)).any(1).sum()
    return loss, top1, top5


def _make_compiled_step(loss_fn: Callable) -> Callable:
    return torch.compile(partial(_forward_backward, loss_fn=loss_fn))


# --- Optuna seed trials ---

_SEED_TRIALS_TEMPLATE: list[dict] = [
    {"optimizer": "adamw", "ref_lr": 4e-6, "beta1": 0.9, "beta2_gap_fraction": 0.99,
     "weight_decay": 0.0, "use_dinov3_init": False},
    {"optimizer": "adamw", "ref_lr": 5e-6, "beta1": 0.9, "beta2_gap_fraction": 0.99,
     "weight_decay": 1e-2, "use_dinov3_init": False},
    {"optimizer": "adamw", "ref_lr": 4e-6, "beta1": 0.7, "beta2_gap_fraction": 0.99,
     "weight_decay": 0.0, "use_dinov3_init": False},
    {"optimizer": "sgd", "ref_lr": 5e-4, "momentum": 0.85, "use_dinov3_init": False},
]


def _seed_trials(max_batch_size: int) -> list[dict]:
    bs_exp = int(math.log2(max_batch_size))
    return [{**t, "batch_size_exp": bs_exp} for t in _SEED_TRIALS_TEMPLATE]


# --- Checkpoint ---


def _save_checkpoint(
    model: nn.Module, val: _FullResults,
    trial_params: dict, cfg: TrainConfig, comet_key: str,
) -> Path:
    model_name = model_name_from_repo(cfg.model_repo)
    ckpt_dir = cfg.checkpoint_dir / "dinov3" / "probes" / "in1k" / model_name / str(cfg.image_size)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"real{val.real_top1:.4f}_{comet_key}.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "val_results": val._asdict(),
        "trial_params": trial_params,
        "config_metadata": {
            "model_name": model_name, "image_size": cfg.image_size,
            "outer_epochs": cfg.outer_epochs,
        },
        "timestamp": datetime.now().isoformat(),
        "comet_experiment_key": comet_key,
    }, path)
    log.info("Saved checkpoint: %s", path)
    return path


# --- Optimizer + scheduler ---


def _sample_optimizer(
    trial: optuna.Trial, model: nn.Module, *, peak_lr: float,
) -> tuple[torch.optim.Optimizer, dict]:
    opt_type = trial.suggest_categorical("optimizer", ["adamw", "sgd"])

    if opt_type == "adamw":
        weight_decay = trial.suggest_float("weight_decay", 0, 0.1)
        beta1 = trial.suggest_float("beta1", 0.1, 0.99, log=True)
        beta2_gap = trial.suggest_float("beta2_gap_fraction", 0.01, 1.0, log=True)
        beta2 = beta1 + beta2_gap * (1 - beta1)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=peak_lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )
        return optimizer, {"optimizer": "adamw", "beta1": beta1, "beta2": beta2, "weight_decay": weight_decay}

    momentum = trial.suggest_float("momentum", 0, 0.99)
    optimizer = torch.optim.SGD(model.parameters(), lr=peak_lr, momentum=momentum)
    return optimizer, {"optimizer": "sgd", "momentum": momentum}


def _make_scheduler(
    optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0.0)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)


# --- Single Optuna trial ---


def _run_trial(
    trial: optuna.Trial,
    cfg: TrainConfig,
    train_cls: list[Tensor],
    train_labels: list[Tensor],
    val_cls: Tensor,
    val_labels: Tensor,
    val_filenames: list[str],
    device: torch.device,
    embed_dim: int,
    real_labels: list[list[int]],
) -> float:
    ref_lr = trial.suggest_float("ref_lr", 1e-7, 1e-2, log=True)

    max_bs_exp = int(math.log2(cfg.max_batch_size))
    min_bs_exp = max_bs_exp - int(math.log2(BATCH_SIZE_MIN_DIVISOR))
    bs_exp = trial.suggest_int("batch_size_exp", min_bs_exp, max_bs_exp)
    batch_size = 2 ** bs_exp
    use_dinov3_init = trial.suggest_categorical("use_dinov3_init", [True, False])

    peak_lr = ref_lr * batch_size
    log.info("Trial %d: BS=%d ref_lr=%.2e peak_lr=%.2e", trial.number, batch_size, ref_lr, peak_lr)

    model = nn.Linear(embed_dim, NUM_CLASSES, bias=True).to(device)
    if use_dinov3_init:
        model.weight.data.normal_(mean=0.0, std=0.01)
        model.bias.data.zero_()

    loss_fn = get_loss_fn(cfg.objective)
    step_fn = _make_compiled_step(loss_fn)

    optimizer, opt_hp = _sample_optimizer(trial, model, peak_lr=peak_lr)

    n_train_epochs = len(train_cls)
    samples_per_epoch = len(train_cls[0])
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    total_epochs = n_train_epochs * cfg.outer_epochs
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = int(WARMUP_FRACTION * total_steps)

    scheduler = _make_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    experiment = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace)
    trial_params: dict = {
        "objective": cfg.objective, "ref_lr": ref_lr,
        "peak_lr": peak_lr, "batch_size": batch_size, "batch_size_exp": bs_exp,
        "outer_epochs": cfg.outer_epochs, "warmup_fraction": WARMUP_FRACTION,
        "warmup_steps": warmup_steps, "n_train_epochs": n_train_epochs,
        "steps_per_epoch": steps_per_epoch, "total_epochs": total_epochs,
        "total_steps": total_steps, "use_dinov3_init": use_dinov3_init,
        "probe_params": sum(p.numel() for p in model.parameters()),
        **opt_hp,
    }
    experiment.log_parameters(trial_params)
    experiment.add_tag(f"trial_{trial.number}")

    init_clf = evaluate(model, features=val_cls, labels=val_labels, batch_size=batch_size, loss_fn=loss_fn)

    pbar = tqdm(total=total_steps, desc=f"T{trial.number}")
    epoch = 0
    val_results: _FullResults | None = None

    for _outer in range(cfg.outer_epochs):
        for epoch_idx in range(n_train_epochs):
            model.train()
            epoch_cls = train_cls[epoch_idx]
            epoch_labels = train_labels[epoch_idx]
            n_samples = len(epoch_cls)

            pbar.set_description(f"T{trial.number} E{epoch}/{total_epochs - 1}")

            total_loss = torch.tensor(0.0, device=device)
            total_top1 = torch.tensor(0, device=device)
            total_top5 = torch.tensor(0, device=device)
            grad_norms = torch.full((steps_per_epoch,), float("nan"), device=device)

            for step_idx, i in enumerate(range(0, n_samples, batch_size)):
                feat = epoch_cls[i : i + batch_size]
                lbl = epoch_labels[i : i + batch_size]

                optimizer.zero_grad()
                loss, c1, c5 = step_fn(model, feat, lbl)
                grad_norms[step_idx] = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_top1 += c1
                total_top5 += c5
                pbar.update(1)

            clf = evaluate(model, features=val_cls, labels=val_labels, batch_size=batch_size, loss_fn=loss_fn)
            r_top1 = real_accuracy(clf.predictions.cpu().numpy(), val_filenames, real_labels)
            val_results = _FullResults(loss=clf.loss, top1=clf.top1, top5=clf.top5, real_top1=r_top1)

            experiment.log_metrics({
                "train_loss": (total_loss / steps_per_epoch).item(),
                "train_top1": (total_top1 / n_samples).item(),
                "train_top5": (total_top5 / n_samples).item(),
                "grad_norm_mean": grad_norms.nanmean().item(),
                "grad_norm_max": grad_norms.max().item(),
                "lr": scheduler.get_last_lr()[0],
                "val_loss": val_results.loss, "val_top1": val_results.top1,
                "val_top5": val_results.top5, "val_real_top1": val_results.real_top1,
            }, epoch=epoch)

            if val_results.loss > init_clf.loss:
                pbar.write(f"Trial {trial.number} pruned: val_loss={val_results.loss:.4f} > init={init_clf.loss:.4f}")
                pbar.close()
                experiment.end()
                raise optuna.TrialPruned()

            trial.report(val_results.real_top1, step=epoch)
            epoch += 1

    pbar.close()
    assert val_results is not None

    log.info("Trial %d done: ReAL=%.4f, top1=%.4f, top5=%.4f",
             trial.number, val_results.real_top1, val_results.top1, val_results.top5)

    _save_checkpoint(model, val_results, trial_params, cfg, comet_key=experiment.get_key())
    experiment.end()
    return val_results.real_top1


# --- Entry point ---


def run_training(cfg: TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    assert cfg.max_batch_size > 0 and (cfg.max_batch_size & (cfg.max_batch_size - 1)) == 0, \
        f"max_batch_size must be power of 2, got {cfg.max_batch_size}"

    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    device = torch.device("cuda")

    log.info("=" * 60)
    log.info("DINOv3 IN1K Linear Probe — Optuna HP Search")
    log.info("  model: %s @ %dpx", cfg.model_repo, cfg.image_size)
    log.info("  max_batch_size=%d  outer_epochs=%d  trials=%d  objective=%s",
             cfg.max_batch_size, cfg.outer_epochs, cfg.n_trials, cfg.objective)
    log.info("=" * 60)

    train_cls, train_labels, embed_dim = load_train_features(cfg)
    val_cls, val_labels, val_filenames = load_val_features(cfg, embed_dim)

    real_labels = load_real_labels()
    log.info("ReAL labels: %d entries", len(real_labels))

    class_names = load_class_names()
    for i in [0, 1, 2]:
        log.info("  val[%d] %s → class %d (%s)", i, val_filenames[i], val_labels[i], class_names[val_labels[i]])

    train_cls = [t.to(device) for t in train_cls]
    train_labels = [t.to(device) for t in train_labels]
    val_cls = val_cls.to(device)
    val_labels = val_labels.to(device)

    study = optuna.create_study(direction="maximize", study_name="linear_probe_lr_sweep")
    for params in _seed_trials(cfg.max_batch_size):
        study.enqueue_trial(params)

    def _log_best(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        log.info("Best so far: trial %d, ReAL=%.4f, params=%s",
                 study.best_trial.number, study.best_value, study.best_params)

    study.optimize(
        lambda trial: _run_trial(
            trial, cfg, train_cls, train_labels,
            val_cls, val_labels, val_filenames,
            device, embed_dim, real_labels,
        ),
        n_trials=cfg.n_trials,
        callbacks=[_log_best],
    )

    log.info("=" * 60)
    log.info("DONE. Best trial %d: ReAL=%.4f", study.best_trial.number, study.best_value)
    log.info("  params: %s", study.best_params)
    log.info("=" * 60)
