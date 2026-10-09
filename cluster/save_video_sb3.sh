#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"
cd "${SLURM_SUBMIT_DIR}"

export MUJOCO_GL=egl
export EGL_DEVICE_ID=0


cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

python -m scripts.save_sb3_video ppo_500k

echo "Job finished at: $(date)"
