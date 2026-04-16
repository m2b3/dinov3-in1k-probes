# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dinov3-in1k-probes @ git+https://github.com/yberreby/dinov3-in1k-probes.git",
#     "torch==2.9.1",
#     "transformers==4.57.1",
#     "torchvision==0.24.1",
#     "tqdm==4.67.1",
# ]
# ///
"""IN1K evaluation for DINOv3 linear probes.

Loads DINOv3 backbone + linear probe from HuggingFace Hub,
runs IN1K validation with correct preprocessing,
reports standard top-1, top-5, and ImageNet-ReAL top-1.

Image size and backbone are derived from the probe's HF config metadata.

Usage:
    uv run eval_in1k.py --imagenet-val /path/to/val
    uv run eval_in1k.py --imagenet-val /path/to/val --variant vitb16
    uv run eval_in1k.py --imagenet-val /path/to/val --probe some-org/custom-probe
"""

import argparse
import json
import logging
import time
from itertools import islice
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoModel

from dinov3_in1k_probes import DINOv3LinearClassificationHead, VARIANTS, probe_repo
from dinov3_in1k_probes.data import NUM_CLASSES, load_real_labels, real_accuracy
from dinov3_in1k_probes.repos import dinov3_repo_from_model_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_val_transform(image_size: int) -> transforms.Compose:
    """Match the DALI val pipeline: resize shortest side → center crop → normalize."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 IN1K linear probes")
    parser.add_argument("--imagenet-val", type=Path, required=True,
                        help="Path to IN1K val/ directory (class subdirectories)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--variant", default="vits16plus", choices=VARIANTS,
                       help="Shorthand for published probes (default: vits16plus)")
    group.add_argument("--probe", help="Full HF repo ID for a custom probe")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches (for quick testing)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true", help="torch.compile the backbone")
    args = parser.parse_args()

    assert args.imagenet_val.is_dir(), f"Not a directory: {args.imagenet_val}"
    device = torch.device(args.device)

    # --- Resolve probe repo and read its config ---
    probe_repo_id = args.probe or probe_repo(args.variant)
    log.info("Probe repo: %s", probe_repo_id)

    cfg_path = hf_hub_download(probe_repo_id, "config.json")
    with open(cfg_path) as f:
        probe_cfg = json.load(f)

    meta = probe_cfg["config_metadata"]
    image_size: int = meta["image_size"]
    backbone_repo = dinov3_repo_from_model_name(meta["model_name"])
    log.info("Backbone: %s (from probe metadata)", backbone_repo)
    log.info("Image size: %dx%d (from probe metadata)", image_size, image_size)
    log.info("Device: %s", device)

    # --- Load backbone ---
    t_load = time.perf_counter()
    log.info("Loading DINOv3 backbone...")
    backbone = AutoModel.from_pretrained(backbone_repo, dtype=torch.float32)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.to(device)
    hidden_size = int(backbone.config.hidden_size)
    log.info("Backbone loaded in %.1fs (hidden_size=%d, patch_size=%d, registers=%d)",
             time.perf_counter() - t_load, hidden_size,
             backbone.config.patch_size, backbone.config.num_register_tokens)
    if args.compile:
        log.info("Compiling backbone...")
        backbone.compile()

    # --- Load probe ---
    t_load = time.perf_counter()
    probe = DINOv3LinearClassificationHead.from_pretrained(probe_repo_id)
    probe.eval()
    probe.to(device)
    assert probe.in_features == hidden_size, (
        f"Dimension mismatch: probe.in_features={probe.in_features} != backbone.hidden_size={hidden_size}"
    )
    assert probe.out_features == NUM_CLASSES
    log.info("Probe loaded in %.1fs (in=%d, out=%d)",
             time.perf_counter() - t_load, probe.in_features, probe.out_features)

    # --- Dataset ---
    t_load = time.perf_counter()
    dataset = datasets.ImageFolder(str(args.imagenet_val), transform=make_val_transform(image_size))
    assert len(dataset.classes) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(dataset.classes)}"
    log.info("Dataset: %d images, %d classes (%.1fs)",
             len(dataset), len(dataset.classes), time.perf_counter() - t_load)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True,
    )

    # Relative filenames for ReAL lookup
    val_root = Path(args.imagenet_val)
    filenames = [str(Path(p).relative_to(val_root)) for p, _ in dataset.imgs]
    real_labels = load_real_labels()

    # --- Evaluate ---
    all_preds: list[torch.Tensor] = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    n_batches = args.max_batches or len(loader)
    batches = enumerate(loader)
    if args.max_batches is not None:
        batches = islice(batches, args.max_batches)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, (images, labels) in tqdm(batches, total=n_batches, desc="Evaluating"):
            t_batch = time.perf_counter()
            images = images.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                cls_tokens = backbone(images).last_hidden_state[:, 0]
            cls_tokens = cls_tokens.float()

            logits = probe(cls_tokens)
            preds = logits.argmax(dim=-1).cpu()
            top5_preds = logits.topk(5, dim=-1).indices.cpu()

            all_preds.append(preds)
            correct_top1 += (preds == labels).sum().item()
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += len(labels)

            batch_time = time.perf_counter() - t_batch
            if batch_idx == 0:
                log.info("First batch: %.1fs (%.0f img/s), estimated total: %.0fs (%.1f min)",
                         batch_time, len(labels) / batch_time,
                         len(dataset) / (len(labels) / batch_time),
                         len(dataset) / (len(labels) / batch_time) / 60)
            elif batch_idx == 2:
                # Steady-state estimate (after warmup)
                elapsed = time.perf_counter() - t0
                img_per_sec = total / elapsed
                remaining = (len(dataset) - total) / img_per_sec
                log.info("Steady-state: %.0f img/s, ~%.0fs remaining", img_per_sec, remaining)

    elapsed = time.perf_counter() - t0
    all_preds_np = torch.cat(all_preds).numpy()

    top1 = correct_top1 / total
    top5 = correct_top5 / total
    real_top1 = real_accuracy(all_preds_np, filenames[:total], real_labels)

    # Published numbers from probe config
    published = probe_cfg.get("val_results", {})

    print(f"\n{'=' * 60}")
    print(f"Probe:   {probe_repo_id}")
    print(f"Images:  {total}")
    print(f"Time:    {elapsed:.1f}s ({total / elapsed:.0f} img/s)")
    print(f"{'=' * 60}")
    print(f"IN1K val top-1:  {top1 * 100:.2f}%", end="")
    if "top1" in published:
        print(f"  (published: {published['top1'] * 100:.2f}%)")
    else:
        print()
    print(f"IN1K val top-5:  {top5 * 100:.2f}%", end="")
    if "top5" in published:
        print(f"  (published: {published['top5'] * 100:.2f}%)")
    else:
        print()
    print(f"IN-ReAL top-1:   {real_top1 * 100:.2f}%", end="")
    if "real_top1" in published:
        print(f"  (published: {published['real_top1'] * 100:.2f}%)")
    else:
        print()
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
