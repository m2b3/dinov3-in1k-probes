"""DINOv3 CLS token extraction from ImageNet-1K.

Phase 1 of the two-phase pipeline: run the frozen DINOv3 backbone over
ImageNet train/val splits and cache CLS tokens to disk as .pt files.

Output layout:
  {features_dir}/{model_name}/{image_size}/{split}/
    *_cls_tokens.pt   (N, embed_dim) float32
    *_labels.pt       (N, 1) int32
    *_filenames.pt    list[str]  (val only)

Train files get a unique timestamp+UUID prefix (for concurrent SLURM jobs).
Val gets a fixed "features" prefix (single deterministic pass).
"""

import logging
import math
import uuid
from datetime import datetime
from itertools import islice
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

from dinov3_in1k_probes.repos import model_name_from_repo
from dinov3_in1k_probes.training.backbone import extract_cls, load_dinov3
from dinov3_in1k_probes.training.config import ExtractionConfig
from dinov3_in1k_probes.training.dali_loader import create_loader

log = logging.getLogger(__name__)

CLS_SUFFIX = "_cls_tokens.pt"
LABELS_SUFFIX = "_labels.pt"
FILENAMES_SUFFIX = "_filenames.pt"

_NAN = float("nan")
_LABEL_CANARY = -1


def _atomic_save(obj: object, path: Path, *, dry_run: bool) -> None:
    if dry_run:
        log.info("[DRY RUN] Would save → %s", path)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _split_dir(features_dir: Path, *, model_repo: str, image_size: int, split: str) -> Path:
    return features_dir / model_name_from_repo(model_repo) / str(image_size) / split


def _output_prefix(features_dir: Path, *, model_repo: str, image_size: int, split: str) -> Path:
    d = _split_dir(features_dir, model_repo=model_repo, image_size=image_size, split=split)
    d.mkdir(parents=True, exist_ok=True)
    if split == "train":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        return d / f"{ts}_{uid}"
    return d / "features"


def _unlink_files(files: list[Path], *, dry_run: bool) -> None:
    for f in files:
        if not dry_run:
            f.unlink()


def run_extraction(cfg: ExtractionConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    assert cfg.split in ("train", "val")
    device = torch.device("cuda")
    is_val = cfg.split == "val"

    model = load_dinov3(cfg.model_repo, device=device)
    embed_dim: int = model.config.hidden_size  # type: ignore[assignment]
    log.info("  embed_dim=%d", embed_dim)

    data_root = str(cfg.imagenet_root / cfg.split)
    assert Path(data_root).is_dir(), f"Not found: {data_root}"

    loader = create_loader(
        data_root=data_root,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        training=not is_val,
        return_filenames=is_val,
    )

    total_images = loader.images_per_epoch
    n_batches = math.ceil(total_images / cfg.batch_size)
    if cfg.max_batches is not None:
        n_batches = min(n_batches, cfg.max_batches)
        total_images = min(total_images, cfg.max_batches * cfg.batch_size)

    log.info("%s: %d images, %d batches", cfg.split, total_images, n_batches)

    acc_device = torch.device(cfg.accumulation_device)
    cls_acc = torch.full((total_images, embed_dim), _NAN, dtype=torch.float32, device=acc_device)
    lbl_acc = torch.full((total_images, 1), _LABEL_CANARY, dtype=torch.int32, device=acc_device)
    filenames: list[str] = []

    prefix = _output_prefix(cfg.features_dir, model_repo=cfg.model_repo, image_size=cfg.image_size, split=cfg.split)
    wip_files: list[Path] = []

    it = iter(loader)
    if cfg.max_batches is not None:
        it = islice(it, cfg.max_batches)

    written = 0
    with tqdm(total=total_images, desc=cfg.split) as pbar:
        for batch_idx, batch in enumerate(it):
            start = batch_idx * cfg.batch_size
            remaining = total_images - start
            B = min(batch["images"].shape[0], remaining)

            images = batch["images"][:B]
            labels = batch["labels"][:B]
            assert labels.min() >= 0 and labels.max() <= 999

            cls_token = extract_cls(model, images)
            assert cls_token.shape == (B, embed_dim)

            cls_acc[start : start + B] = cls_token.to(acc_device, non_blocking=True)
            lbl_acc[start : start + B] = labels.to(acc_device, dtype=torch.int32, non_blocking=True)

            if batch.get("filenames") is not None:
                filenames.extend(batch["filenames"][:B])

            written = start + B
            pbar.update(B)

            if (batch_idx + 1) % cfg.checkpoint_every_n_batches == 0:
                _unlink_files(wip_files, dry_run=cfg.dry_run)
                tag = f"_batch{batch_idx + 1}of{n_batches}.wip"
                wip_cls = Path(f"{prefix}{CLS_SUFFIX}{tag}.pt")
                wip_lbl = Path(f"{prefix}{LABELS_SUFFIX}{tag}.pt")
                torch.cuda.synchronize()
                _atomic_save(cls_acc[:written].cpu(), wip_cls, dry_run=cfg.dry_run)
                _atomic_save(lbl_acc[:written].cpu(), wip_lbl, dry_run=cfg.dry_run)
                wip_files = [wip_cls, wip_lbl]
                log.info("  checkpoint @ batch %d/%d (%d samples)", batch_idx + 1, n_batches, written)

    assert written == total_images

    torch.cuda.synchronize()
    cls_final = cls_acc.cpu()
    lbl_final = lbl_acc.cpu()
    assert not torch.isnan(cls_final).any(), "CLS contains NaN — incomplete extraction"
    assert (lbl_final != _LABEL_CANARY).all(), "Labels contain canary — incomplete extraction"

    _atomic_save(cls_final, Path(f"{prefix}{CLS_SUFFIX}"), dry_run=cfg.dry_run)
    _atomic_save(lbl_final, Path(f"{prefix}{LABELS_SUFFIX}"), dry_run=cfg.dry_run)
    if is_val:
        assert len(filenames) == total_images
        _atomic_save(filenames, Path(f"{prefix}{FILENAMES_SUFFIX}"), dry_run=cfg.dry_run)

    _unlink_files(wip_files, dry_run=cfg.dry_run)

    gb = cls_final.nbytes / 1e9
    log.info("Done: %d samples, %s, %.2f GB → %s_*.pt", total_images, tuple(cls_final.shape), gb, prefix)

    if torch.cuda.is_available():
        log.info("Peak CUDA memory: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
