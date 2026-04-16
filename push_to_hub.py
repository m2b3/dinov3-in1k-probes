import re
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import torch
import tyro

from dinov3_in1k_probes import DINOv3LinearClassificationHead

FILENAME_PATTERN = r"dinov3-(?P<slug>[^-]+)-lvd1689m-in1k-(?P<res>\d+)x\d+-linear-clf-probe\.pt"


@dataclass
class Args:
    """Push DINOv3 linear probe to HuggingFace Hub."""

    checkpoint: Path
    owner: str = "yberreby"


def main() -> None:
    args = tyro.cli(Args)

    print(f"\nCheckpoint: {args.checkpoint}")

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    metadata = ckpt["config_metadata"]
    model_name = metadata["model_name"]
    slug = model_name.replace("dinov3_", "")
    res = metadata["image_size"]
    val_results = ckpt["val_results"]

    print(f"Model: {model_name}")
    print(f"Slug: {slug}")
    print(f"Resolution: {res}x{res}")
    print(f"IN1k val top-1: {val_results['top1'] * 100:.2f}%")
    print(f"IN1k-ReAL top-1: {val_results['real_top1'] * 100:.2f}%")

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

    out_features, in_features = ckpt["model_state_dict"]["weight"].shape
    print(f"Dimensions: in_features={in_features}, out_features={out_features}")

    probe = DINOv3LinearClassificationHead(in_features, out_features)
    probe.load_state_dict(ckpt["model_state_dict"])

    config = {
        "in_features": in_features,
        "out_features": out_features,
        **{k: v for k, v in ckpt.items() if k != "model_state_dict"},
    }

    print("\nFull config:")
    pprint(config)

    repo_id = f"{args.owner}/dinov3-{slug}-lvd1689m-in1k-{res}x{res}-linear-clf-probe"
    print(f"\nPushing to {repo_id}...")
    probe.push_to_hub(repo_id, config=config)
    print(f"✓ Successfully pushed to {repo_id}")


if __name__ == "__main__":
    main()
