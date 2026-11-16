#!/bin/bash
#SBATCH --job-name=evaluate_seeded
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00

# Usage: bash cluster/evaluate_seeded.sh <exp_name> <run_dir> <pool> <n> <run_pass>
set -euo pipefail

EXP_NAME="$1"
RUN_DIR="$2"
POOL="$3"
N="$4"
RUN_PASS="$5"

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

${SLURM_JOB_ID:+srun} python scripts/evaluate_seeded.py "$EXP_NAME" --run-dir "$RUN_DIR" --pool "$POOL" --n "$N" --run-pass "$RUN_PASS"

echo "Job finished at: $(date)"
