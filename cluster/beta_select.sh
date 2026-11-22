#!/bin/bash
#SBATCH --job-name=beta_select
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --no-requeue

# Usage, from the repo to run: sbatch -J <name> cluster/beta_select.sh fit <base> --journal <journal.log> --out-dir <dir>
#        sbatch -J <name> cluster/beta_select.sh select <base> <run_pass> --top3 <top3.json> --out-dir <dir>
#        (or plain bash with the same arguments)
# The code run is that of the dir sbatch is called from (SLURM_SUBMIT_DIR).
# fit: the arguments go to scripts/beta_select.py fit as they are.
# select: the confirm runs are found as runs/sweeps/<base>_<run_pass>_confirm_b*_s0-2/runs.json (their dir names
# hold beta) and passed as --runs; the other arguments go to scripts/beta_select.py select.
set -euo pipefail

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ "$1" = select ]; then
    BASE="$2"
    RUN_PASS="$3"
    shift 3
    RUNS=(runs/sweeps/"${BASE}_${RUN_PASS}"_confirm_b*_s0-2/runs.json)
    python scripts/beta_select.py select "$BASE" --runs "${RUNS[@]}" "$@"
else
    python scripts/beta_select.py "$@"
fi

echo "Job finished at: $(date)"
