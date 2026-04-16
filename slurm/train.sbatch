#!/bin/bash
#SBATCH --job-name=cls-probe
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/cls-probe-%j.out
#SBATCH --error=logs/cls-probe-%j.err

# IN1K classification probe training with Optuna HP search.
# Expects pre-extracted CLS features at $FEATURES_DIR.
#
# Usage:
#   sbatch slurm/train.sbatch
#   sbatch slurm/train.sbatch --model-repo facebook/dinov3-vitb16-pretrain-lvd1689m
#   sbatch slurm/train.sbatch --n-trials 200

set -eu
mkdir -p logs

echo "=== IN1K Classification Probe Training (Optuna) ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run --group training python -m dinov3_in1k_probes.training train "$@"
