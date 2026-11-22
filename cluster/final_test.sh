#!/bin/bash
#SBATCH --job-name=final_test
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --no-requeue

# Usage, from the repo to run: sbatch -J test_<base> cluster/final_test.sh <base> <run_pass> <chosen_beta.json> <pool> [n]
#        plain bash: [ALLOW_DIRTY=1] bash cluster/final_test.sh <base> <run_pass> <chosen_beta.json> <pool> [n]
# The code run is that of the dir sbatch is called from (SLURM_SUBMIT_DIR).
# Evaluates the base and its 12 final runs on the first n seeds of the pool (the test: pool test, all seeds).
# ALLOW_DIRTY=1 is for dry runs on another pool only; the script refuses it with the test pool.
set -euo pipefail

BASE="$1"
RUN_PASS="$2"
CHOSEN="$3"
POOL="$4"
N="${5:-}"
DIRTY_FLAG="${ALLOW_DIRTY:+--allow-dirty}"

echo "Starting job on node: $(hostname)"
echo "CPUs available: $(nproc)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/final_test.py "$BASE" --run-pass "$RUN_PASS" --chosen "$CHOSEN" --pool "$POOL" ${N:+--n "$N"} ${DIRTY_FLAG}

echo "Job finished at: $(date)"
