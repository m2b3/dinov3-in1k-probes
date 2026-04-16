# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dinov3-in1k-probes @ git+https://github.com/yberreby/dinov3-in1k-probes.git",
#     "transformers>=4.50",
#     "torchvision>=0.20",
#     "tqdm",
# ]
# ///
"""Standalone IN1K evaluation for DINOv3 linear probes.

Loads DINOv3 backbone + linear probe from HuggingFace Hub,
runs IN1K validation at 512x512 with correct preprocessing,
and reports standard top-1, top-5, and ImageNet-ReAL top-1.

Usage:
    uv run eval_in1k.py --imagenet-val /path/to/ILSVRC2012/val
    uv run eval_in1k.py --imagenet-val /path/to/val --variant vits16plus
"""

import argparse
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import AutoModel

from dinov3_in1k_probes import DINOv3LinearClassificationHead
from dinov3_in1k_probes.data import NUM_CLASSES, load_real_labels, real_accuracy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV3_REPOS = {
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}

PROBE_REPOS = {
    k: f"yberreby/dinov3-{k}-lvd1689m-in1k-512x512-linear-clf-probe"
    for k in DINOV3_REPOS
}


def make_val_transform(image_size: int = 512) -> transforms.Compose:
    """Match the DALI val pipeline: resize shortest side → center crop → normalize."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 IN1K linear probes")
    parser.add_argument("--imagenet-val", type=Path, required=True, help="Path to IN1K val/ directory")
    parser.add_argument("--variant", default="vits16plus", choices=list(DINOV3_REPOS))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches (for quick testing)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert args.imagenet_val.is_dir(), f"Not a directory: {args.imagenet_val}"

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Load models ---
    dinov3_repo = DINOV3_REPOS[args.variant]
    probe_repo = PROBE_REPOS[args.variant]
    print(f"DINOv3: {dinov3_repo}")
    print(f"Probe:  {probe_repo}")

    print("Loading DINOv3 backbone...", end=" ", flush=True)
    backbone = AutoModel.from_pretrained(dinov3_repo, torch_dtype=torch.float32)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.to(device)
    print(f"done (hidden_size={backbone.config.hidden_size})")

    print("Loading probe...", end=" ", flush=True)
    probe = DINOv3LinearClassificationHead.from_pretrained(probe_repo)
    probe.eval()
    probe.to(device)
    print(f"done (in={probe.in_features}, out={probe.out_features})")
    assert probe.out_features == NUM_CLASSES

    # --- Dataset ---
    transform = make_val_transform(args.image_size)
    dataset = datasets.ImageFolder(str(args.imagenet_val), transform=transform)
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    assert len(dataset.classes) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(dataset.classes)}"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # --- Filenames for ReAL ---
    # ImageFolder stores (path, label) pairs — we need the relative path for ReAL lookup
    filenames = [dataset.imgs[i][0] for i in range(len(dataset))]
    # Extract just "nXXXX/ILSVRC2012_val_NNNN.JPEG" from full path
    val_root = Path(args.imagenet_val)
    filenames = [str(Path(f).relative_to(val_root)) for f in filenames]

    real_labels = load_real_labels()

    # --- Evaluate ---
    all_preds = []
    all_labels = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    batches = enumerate(loader)
    if args.max_batches is not None:
        batches = islice(batches, args.max_batches)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for batch_idx, (images, labels) in tqdm(batches, total=args.max_batches or len(loader), desc="Evaluating"):
            images = images.to(device, non_blocking=True)

            # CLS extraction with bf16 autocast (matches training)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                cls_tokens = backbone(images).last_hidden_state[:, 0]
            cls_tokens = cls_tokens.float()

            logits = probe(cls_tokens)
            preds = logits.argmax(dim=-1).cpu()
            top5 = logits.topk(5, dim=-1).indices.cpu()

            all_preds.append(preds)
            all_labels.append(labels)

            correct_top1 += (preds == labels).sum().item()
            correct_top5 += sum(labels[i] in top5[i] for i in range(len(labels)))
            total += len(labels)

            if batch_idx == 0:
                elapsed = time.perf_counter() - t0
                img_per_sec = len(labels) / elapsed
                print(f"\nFirst batch: {elapsed:.1f}s ({img_per_sec:.0f} img/s)")
                est_total = len(dataset) / img_per_sec
                print(f"Estimated total: {est_total:.0f}s ({est_total/60:.1f} min)")

    elapsed = time.perf_counter() - t0

    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()

    top1 = correct_top1 / total
    top5 = correct_top5 / total
    real_top1 = real_accuracy(all_preds_np, filenames[:total], real_labels)

    print(f"\n{'='*60}")
    print(f"Variant: {args.variant} @ {args.image_size}x{args.image_size}")
    print(f"Images:  {total}")
    print(f"Time:    {elapsed:.1f}s ({total/elapsed:.0f} img/s)")
    print(f"{'='*60}")
    print(f"IN1K val top-1:  {top1*100:.2f}%")
    print(f"IN1K val top-5:  {top5*100:.2f}%")
    print(f"IN-ReAL top-1:   {real_top1*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
