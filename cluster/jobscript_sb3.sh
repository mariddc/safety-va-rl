#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00

set -euo pipefail

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"
echo "Workdir: ${SLURM_SUBMIT_DIR}"

cd "${SLURM_SUBMIT_DIR}"

# Activate uv virtual environment
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export N_ENVS=8

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run experiment
srun python scripts/train.py ppo_5m700k_goal2

echo "Job finished at: $(date)"
