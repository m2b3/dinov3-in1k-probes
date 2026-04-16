"""Fetch and display metrics + hyperparameters for all published DINOv3 IN1K probes."""

import json

from huggingface_hub import hf_hub_download
from tabulate import tabulate

from dinov3_in1k_probes import VARIANTS, probe_repo

configs = {}
for variant in VARIANTS:
    repo = probe_repo(variant)
    cfg_path = hf_hub_download(repo, "config.json")
    with open(cfg_path) as f:
        configs[variant] = json.load(f)

# --- Accuracy table ---
acc_rows = []
for variant, cfg in configs.items():
    vr = cfg["val_results"]
    acc_rows.append([
        variant,
        cfg["in_features"],
        f"{vr['top1'] * 100:.2f}%",
        f"{vr['top5'] * 100:.2f}%",
        f"{vr['real_top1'] * 100:.2f}%",
        f"{vr['loss']:.4f}",
    ])

print("=== Accuracy ===\n")
print(tabulate(acc_rows, headers=["Variant", "dim", "top-1", "top-5", "ReAL top-1", "loss"], tablefmt="github"))

# --- Hyperparameters table ---
hp_rows = []
for variant, cfg in configs.items():
    tp = cfg["trial_params"]
    hp_rows.append([
        variant,
        tp["optimizer"],
        f"{tp['peak_lr']:.2e}",
        f"{tp['weight_decay']:.2e}",
        tp["batch_size"],
        tp["total_steps"],
        f"{tp['beta1']:.3f}",
        f"{tp['beta2']:.4f}",
        tp["use_dinov3_init"],
        tp["n_train_epochs"],
        cfg["config_metadata"]["outer_epochs"],
    ])

print("\n\n=== Hyperparameters (Optuna best trial) ===\n")
print(tabulate(hp_rows, headers=[
    "Variant", "optim", "peak_lr", "wd", "bs", "steps",
    "β1", "β2", "dv3_init", "epochs", "outer_ep",
], tablefmt="github", disable_numparse=True))

# --- Timestamps ---
print("\n\n=== Metadata ===\n")
for variant, cfg in configs.items():
    print(f"  {variant}: trained {cfg['timestamp']}, dim={cfg['in_features']}, "
          f"image_size={cfg['config_metadata']['image_size']}")
