#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=4:00:00
#SBATCH --no-requeue

# Usage, from the repo to run: sbatch -J sweep_<base> cluster/sweep_trial.sh <base> <run_pass> [args]   ([ALLOW_DIRTY=1] for dry runs)
#        plain bash: [ALLOW_DIRTY=1] bash cluster/sweep_trial.sh <base> <run_pass> [args]
# Runs one trial of the base's beta sweep (3 seed fine-tunes in parallel, then their evals) and exits.
# Everything resolves from the submit dir, so a separate copy of the repo can run its own chain.
# Extra args go to scripts/finetune_sweep.py as they are (e.g. --beta-range 0.1 300 --study-tag ext --warm-start <journal>).
set -euo pipefail

BASE="$1"
RUN_PASS="$2"
shift 2
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

# Plain python: the sweep starts its own fine-tune and eval processes inside this job's allocation.
RC=0
python scripts/finetune_sweep.py "$BASE" --run-pass "$RUN_PASS" ${DIRTY_FLAG} "$@" || RC=$?

# Peak memory of the whole job from the cgroup, when readable (also after a failure).
F=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_"${SLURM_JOB_ID:-none}"/memory.peak
if [ -r "$F" ]; then echo "[sweep] job_memory_peak_bytes=$(cat "$F")"; fi

echo "Job finished at: $(date) (exit $RC)"
exit "$RC"
