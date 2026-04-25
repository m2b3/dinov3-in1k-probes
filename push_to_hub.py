import re
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import torch
import tyro

from dinov3_in1k_probes import DINOv3LinearClassificationHead
from dinov3_in1k_probes.repos import probe_repo

FILENAME_PATTERN = r"dinov3-(?P<slug>[^-]+)-lvd1689m-in1k-(?P<res>\d+)x\d+-linear-clf-probe\.pt"


@dataclass
class Args:
    """Push DINOv3 linear probe to HuggingFace Hub.

    Destination prefix is ``$CANVIT_REPO_ROOT`` (default ``"canvit"``).
    """

    checkpoint: Path


def main() -> None:
    args = tyro.cli(Args)

    print(f"\nCheckpoint: {args.checkpoint}")

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Extract metadata from checkpoint
    metadata = ckpt["config_metadata"]
    model_name = metadata["model_name"]  # e.g., "dinov3_vitb16"
    slug = model_name.replace("dinov3_", "")  # e.g., "vitb16"
    res = metadata["image_size"]
    val_results = ckpt["val_results"]

    print(f"Model: {model_name}")
    print(f"Slug: {slug}")
    print(f"Resolution: {res}x{res}")
    print(f"IN1k val top-1: {val_results['top1'] * 100:.2f}%")
    print(f"IN1k-ReAL top-1: {val_results['real_top1'] * 100:.2f}%")

    # Sanity check: if filename matches pattern, verify it matches checkpoint metadata
    if match := re.match(FILENAME_PATTERN, args.checkpoint.name):
        filename_slug = match.group("slug")
        filename_res = int(match.group("res"))
        if filename_slug != slug or filename_res != res:
            raise ValueError(
                f"Filename metadata mismatch!\n"
                f"  Filename: slug={filename_slug}, res={filename_res}\n"
                f"  Checkpoint: slug={slug}, res={res}"
            )
        print("✓ Filename matches checkpoint metadata")

    # Extract dimensions and create model
    out_features, in_features = ckpt["model_state_dict"]["weight"].shape
    print(f"Dimensions: in_features={in_features}, out_features={out_features}")

    probe = DINOv3LinearClassificationHead(in_features, out_features)
    probe.load_state_dict(ckpt["model_state_dict"])

    # Build config
    config = {
        "in_features": in_features,
        "out_features": out_features,
        **{k: v for k, v in ckpt.items() if k != "model_state_dict"},
    }

    print("\nFull config:")
    pprint(config)

    # Push to hub
    repo_id = probe_repo(slug, image_size=res)
    print(f"\nPushing to {repo_id}...")
    probe.push_to_hub(repo_id, config=config)
    print(f"✓ Successfully pushed to {repo_id}")


if __name__ == "__main__":
    main()
