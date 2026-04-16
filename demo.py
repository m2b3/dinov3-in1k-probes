# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dinov3-in1k-probes @ git+https://github.com/yberreby/dinov3-in1k-probes.git",
#     "timm>=1.0",
#     "transformers>=4.50",
# ]
# ///
"""Demo: DINOv3 ImageNet-1k classification with pretrained linear probe."""

print("Importing dependencies...", end=" ", flush=True)
import json

import timm
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

from dinov3_in1k_probes import DINOv3LinearClassificationHead, probe_repo
from dinov3_in1k_probes.repos import dinov3_repo_from_model_name

print("done")

VARIANT = "vitb16"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
TOP_K = 5

# Derive everything from the probe's config metadata.
probe_repo_id = probe_repo(VARIANT)
cfg_path = hf_hub_download(probe_repo_id, "config.json")
with open(cfg_path) as f:
    probe_cfg = json.load(f)
meta = probe_cfg["config_metadata"]
image_size = meta["image_size"]
dinov3_repo = dinov3_repo_from_model_name(meta["model_name"])

print(f"Loading linear probe: {probe_repo_id}...", end=" ", flush=True)
probe = DINOv3LinearClassificationHead.from_pretrained(probe_repo_id)
print("done")

print(f"Loading DINOv3 model: {dinov3_repo}...", end=" ", flush=True)
# Override processor size to match probe training resolution (default is 224px!)
processor = AutoImageProcessor.from_pretrained(dinov3_repo, size={"height": image_size, "width": image_size})
model = AutoModel.from_pretrained(dinov3_repo)
print("done")
print(f"  Patch size: {model.config.patch_size}")
print(f"  Register tokens: {model.config.num_register_tokens}")
print(f"  Image size: {image_size}x{image_size} (from probe config)")

print(f"Processing image: {IMAGE_URL}...", end=" ", flush=True)
image = load_image(IMAGE_URL)
inputs = processor(images=image, return_tensors="pt")
print("done")
print(f"  Original: {image.width}x{image.height}")
print(f"  Preprocessed: {tuple(inputs.pixel_values.shape)}")

print("Running inference...", end=" ", flush=True)
with torch.inference_mode():
    cls = model(**inputs).last_hidden_state[:, 0, :]
    logits = probe(cls.cpu())
    probs = torch.softmax(logits, dim=-1)
print("done")

ini = timm.data.ImageNetInfo()
topk_idx = logits.topk(TOP_K).indices[0]
topk_probs = probs[0, topk_idx]

print(f"\nTop-{TOP_K} predictions:")
for i, (idx, prob) in enumerate(zip(topk_idx, topk_probs), 1):
    label = ini.index_to_description(idx.item())
    print(f"  {i}. {label:40s} {prob * 100:5.2f}%")
