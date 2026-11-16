#!/bin/bash
#SBATCH --job-name=train_q
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00

# Usage: bash cluster/train_q.sh <exp_name> <dataset_id> <out q_theta.pt> <run_pass>
set -euo pipefail

EXP_NAME="$1"
DATASET_ID="$2"
OUT="$3"
RUN_PASS="$4"

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

${SLURM_JOB_ID:+srun} python -m src.eil.intervention_learning "$EXP_NAME" "$DATASET_ID" --out "$OUT" --run-pass "$RUN_PASS"

echo "Job finished at: $(date)"
