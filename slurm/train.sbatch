#!/bin/bash
#SBATCH --job-name=cls-probe
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/cls-probe-%j.out
#SBATCH --error=logs/cls-probe-%j.err
#
# Pass --account on the command line:
#   sbatch --account=my_project_name slurm/train.sbatch
#   sbatch --account=my_project_name slurm/train.sbatch --n-trials 200

set -eu
mkdir -p logs

echo "=== IN1k Classification Probe Training (Optuna) ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run --group training python -m dinov3_in1k_probes.training train "$@"
