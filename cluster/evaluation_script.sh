#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=30:00:00

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

export MUJOCO_GL=egl

# Run experiment
srun python scripts/evaluate.py ppo_5m600k_goal2_safe_vase_k15

echo "Job finished at: $(date)"
