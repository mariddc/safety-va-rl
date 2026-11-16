#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=3:00:00

# Usage: bash cluster/finetune.sh <exp_name> <run_pass>
set -euo pipefail

EXP_NAME="$1"
RUN_PASS="$2"

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

${SLURM_JOB_ID:+srun} python scripts/finetune_policy.py "$EXP_NAME" --run-pass "$RUN_PASS"

echo "Job finished at: $(date)"
