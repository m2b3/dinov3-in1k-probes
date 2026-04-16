"""DINOv3 backbone loading via HuggingFace transformers."""

import logging

import torch
from torch import Tensor
from transformers import AutoModel, PreTrainedModel

log = logging.getLogger(__name__)


def load_dinov3(repo_id: str, *, device: torch.device) -> PreTrainedModel:
    """Load a frozen DINOv3 backbone from HuggingFace Hub."""
    log.info("Loading DINOv3: %s → %s", repo_id, device)
    model = AutoModel.from_pretrained(repo_id, dtype=torch.float32)
    assert isinstance(model, PreTrainedModel)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)  # pyright: ignore[reportArgumentType]
    log.info("  hidden_size=%d  num_register_tokens=%d  patch_size=%d",
             model.config.hidden_size, model.config.num_register_tokens, model.config.patch_size)
    return model


def extract_cls(model: PreTrainedModel, images: Tensor) -> Tensor:
    """Forward pass → post-norm CLS token, cast to float32."""
    with torch.no_grad(), torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
        out = model(images).last_hidden_state
    return out[:, 0].float()


def model_name_from_repo(repo: str) -> str:
    """'facebook/dinov3-vitb16-pretrain-lvd1689m' → 'dinov3_vitb16'."""
    slug = repo.split("/")[-1]
    assert "-pretrain" in slug, f"Expected '-pretrain' in repo slug: {repo}"
    name = "_".join(slug.split("-pretrain")[0].split("-"))
    assert name.startswith("dinov3_"), f"Parsed name {name!r} doesn't look like a DINOv3 model (from {repo})"
    return name
