import json

from huggingface_hub import hf_hub_download
from tabulate import tabulate

from dinov3_in1k_probes import VARIANTS, probe_repo

rows = []
for variant in VARIANTS:
    repo = probe_repo(variant)
    cfg_path = hf_hub_download(repo, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    vr = cfg["val_results"]
    rows.append([
        repo,
        f"{vr['top1'] * 100:.2f}%",
        f"{vr['real_top1'] * 100:.2f}%",
    ])

headers = ["HF Hub Repo", "IN1k val top-1", "IN-ReAL top-1"]
print(tabulate(rows, headers=headers, tablefmt="github"))
