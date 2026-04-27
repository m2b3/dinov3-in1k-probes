# DINOv3 ImageNet-1k Linear Classification Probes

Upon its release in August 2025, [DINOv3](https://github.com/facebookresearch/dinov3) marked a milestone in self-supervised representation learning for image processing.
The 7-billion-parameter flagship model was distilled into a family of smaller ViT and ConvNeXT checkpoints, whose sizes make them much more suitable for most CV tasks.

Sadly, only one ImageNet-1k (IN1k) linear classification probe was released: the one for the 7B model.

**Here, we release pretrained linear probes for the smaller DINOv3 ViT models.**
They can be used directly with Meta's official checkpoints.

As in the original DINOv3 paper, we used **512x512 inputs** (1024 input tokens),
and trained the probes on the IN1k training set with Inception-crop augmentation.

**All of our probes match or exceed the best IN1k-ReAL top-1 validation accuracy reported by the DINOv3 authors**, as seen in Table 14 of the original paper.

We note that the raw IN1k top-1 validation accuracy was not reported by the DINOv3 authors, only the [ReAL](https://github.com/google-research/reassessed-imagenet) top-1 accuracy.
Here, we report both.


## Released Probes

- **ViT-S/16** @ 512×512
  - Base: [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
  - Probe: [`canvit/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/canvit/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe)

- **ViT-S+/16** @ 512×512
  - Base: [`facebook/dinov3-vits16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m)
  - Probe: [`canvit/dinov3-vits16plus-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/canvit/dinov3-vits16plus-lvd1689m-in1k-512x512-linear-clf-probe)

- **ViT-B/16** @ 512×512
  - Base: [`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
  - Probe: [`canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe)

- **ViT-L/16** @ 512×512
  - Base: [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
  - Probe: [`canvit/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/canvit/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe)

- **ViT-H+/16** @ 512×512
  - Base: [`facebook/dinov3-vith16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m)
  - Probe: [`canvit/dinov3-vith16plus-lvd1689m-in1k-512x512-linear-clf-probe`](https://huggingface.co/canvit/dinov3-vith16plus-lvd1689m-in1k-512x512-linear-clf-probe)

See [the corresponding HuggingFace Collection](https://huggingface.co/collections/canvit/dinov3-imagenet-1k-probes-pytorch-69ec77ba50ddf36e3cba7346).

## Performance

| Probe | [IN-ReAL](https://github.com/google-research/reassessed-imagenet) val top-1 (official / ours) | IN1k val top-1 (ours) |
|-------|--------------------------------|-------------------|
| [ViT-S/16](https://huggingface.co/canvit/dinov3-vits16-lvd1689m-in1k-512x512-linear-clf-probe) | 87.0% / **87.08%** | 81.40% |
| [ViT-S+/16](https://huggingface.co/canvit/dinov3-vits16plus-lvd1689m-in1k-512x512-linear-clf-probe) | 88.0% / **88.08%** | 82.89% |
| [ViT-B/16](https://huggingface.co/canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe) | 89.3% / **89.54%** | 85.00% |
| [ViT-L/16](https://huggingface.co/canvit/dinov3-vitl16-lvd1689m-in1k-512x512-linear-clf-probe) | 90.2% / **90.42%** | 87.44% |
| [ViT-H+/16](https://huggingface.co/canvit/dinov3-vith16plus-lvd1689m-in1k-512x512-linear-clf-probe) | 90.3% / **90.31%** | 87.65% |

Accuracy and full Optuna hyperparameters can be queried using `uv run print_metrics.py`.

## Usage

We recommend using [`uv`](https://docs.astral.sh/uv/).

### Quick demo

Run the demo directly (no clone needed):

```bash
uv run https://raw.githubusercontent.com/m2b3/dinov3-in1k-probes/main/demo.py
```

### Using `from_pretrained`

```python
from dinov3_in1k_probes import DINOv3LinearClassificationHead

probe = DINOv3LinearClassificationHead.from_pretrained(
    "canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"
)
# DINOv3LinearClassificationHead(in_features=768, out_features=1000, bias=True)
```

See [`demo.py`](demo.py) for a complete example including DINOv3 backbone loading and preprocessing.

To get an interactive shell with the package:

```bash
uvx --with 'git+https://github.com/m2b3/dinov3-in1k-probes.git' ipython
```


## Evaluation

Verify published numbers on your own IN1k val set:

```bash
uv run python eval_in1k.py --imagenet-val /path/to/ILSVRC2012/val
uv run python eval_in1k.py --imagenet-val /path/to/val --variant vitb16
```

Image size and backbone are read from the probe's HuggingFace config — no manual configuration needed.

## Training

The training code is included in this repo under `dinov3_in1k_probes/training/`.
Install the training dependencies with `uv sync --group training`.

### Two-phase pipeline

1. **Extract** CLS tokens from a frozen DINOv3 backbone into `.pt` files on disk (GPU, [DALI](https://github.com/NVIDIA/DALI) — Linux/CUDA only).
2. **Train** a linear probe via Optuna HP search over the cached features (all in GPU memory, fast).

This decouples the expensive backbone forward pass from the probe optimization,
making 100+ Optuna trials practical over the full ImageNet training set.

### Environment variables

| Variable | Required by | Description |
|----------|-------------|-------------|
| `IMAGENET_ROOT` | `extract` | Path to `ILSVRC2012/` (must contain `train/` and `val/` subdirs) |
| `FEATURES_DIR` | `extract`, `train` | Where to read/write cached `.pt` feature files |
| `CHECKPOINTS_DIR` | `train` | Where to save probe checkpoints (default: `checkpoints`) |
| `COMET_API_KEY` | `train` | **Required.** Comet ML experiment tracking |
| `COMET_WORKSPACE` | `train` | Comet workspace name (pass via CLI `--comet-workspace` or env var) |

Extracted features are written to `$FEATURES_DIR/<model_name>/<image_size>/<split>/` (e.g.
`$FEATURES_DIR/dinov3_vitb16/512/val/`). Each extraction run writes `*_cls_tokens.pt`,
`*_labels.pt`, and `*_filenames.pt`. Multiple train extractions (for augmentation epochs)
are concatenated automatically by the training script.

### Running locally

```bash
# Phase 1: extract CLS tokens
uv run --group training python -m dinov3_in1k_probes.training extract --split val
uv run --group training python -m dinov3_in1k_probes.training extract --split train  # run N times for N augmentation epochs

# Phase 2: Optuna HP search
uv run --group training python -m dinov3_in1k_probes.training train
```

### Running on SLURM

```bash
sbatch --account=my_project_name slurm/extract.sbatch                        # val
sbatch --account=my_project_name --array=1-2 slurm/extract.sbatch --split train  # 2 train epochs
sbatch --account=my_project_name slurm/train.sbatch
```

Use `--help` on either subcommand for all options (model repo, image size, batch size, etc.).

## Development

Push probes to HuggingFace Hub:

```bash
uv run push_to_hub.py --checkpoint dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe.pt --owner YOUR_HF_USERNAME
```
