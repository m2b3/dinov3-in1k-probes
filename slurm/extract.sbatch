#!/bin/bash
#SBATCH --job-name=extract-cls
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/extract-%j.out
#SBATCH --error=logs/extract-%j.err

# CLS token extraction for DINOv3 on ImageNet-1K.
#
# Each train invocation creates a uniquely-named file (timestamp+UUID).
# Launch concurrent array jobs for multiple augmentation epochs:
#   sbatch --array=1-N slurm/extract.sbatch --split train
#
# Usage:
#   sbatch slurm/extract.sbatch                              # val (default)
#   sbatch slurm/extract.sbatch --split train                # one train epoch
#   sbatch slurm/extract.sbatch --model-repo facebook/dinov3-vitl16-pretrain-lvd1689m

set -eu
mkdir -p logs

echo "=== DINOv3 CLS Token Extraction ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run --group training python -m dinov3_in1k_probes.training extract "$@"
