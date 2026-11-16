#!/bin/bash
#SBATCH --job-name=rollout_q_stats
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00

# Usage: bash cluster/rollout_q_stats.sh <exp_name> <n> <run_pass>
set -euo pipefail

EXP_NAME="$1"
N="$2"
RUN_PASS="$3"

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

${SLURM_JOB_ID:+srun} python scripts/rollout_q_stats.py "$EXP_NAME" --n "$N" --run-pass "$RUN_PASS"

echo "Job finished at: $(date)"
