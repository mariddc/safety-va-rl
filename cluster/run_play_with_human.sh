#!/bin/bash
#SBATCH --job-name=play_human
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --time=02:00:00

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR}/cluster"
cd "${SLURM_SUBMIT_DIR}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
export PYOPENGL_PLATFORM=egl

python play_with_human.py
