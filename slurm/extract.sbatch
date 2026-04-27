#!/bin/bash
#SBATCH --job-name=extract-cls
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/extract-%j.out
#SBATCH --error=logs/extract-%j.err
#
# Pass --account on the command line:
#   sbatch --account=my_project_name slurm/extract.sbatch
#   sbatch --account=my_project_name --array=1-2 slurm/extract.sbatch --split train

set -eu
mkdir -p logs

echo "=== DINOv3 CLS Token Extraction ==="
echo "Job: $SLURM_JOB_ID on $(hostname)"

exec uv run --group training python -m dinov3_in1k_probes.training extract "$@"
