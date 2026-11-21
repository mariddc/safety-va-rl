#!/bin/bash
#SBATCH --job-name=sweep_ref
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=4:00:00
#SBATCH --no-requeue

# Usage, from the repo to run: sbatch -J sweep_ref_<base> cluster/sweep_ref.sh <base> <run_pass>   ([ALLOW_DIRTY=1] for dry runs)
#        plain bash: [ALLOW_DIRTY=1] bash cluster/sweep_ref.sh <base> <run_pass>
# The code run is that of the dir sbatch is called from (SLURM_SUBMIT_DIR); --chdir does not change it.
# Runs the base's sweep seeds at beta = 0 (3 fine-tunes in parallel, then their evals), with the resources of a
# sweep trial job, so the result pairs with every trial.
set -euo pipefail

BASE="$1"
RUN_PASS="$2"
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

# Plain python: the script starts its own fine-tune and eval processes inside this job's allocation.
RC=0
python scripts/sweep_reference.py "$BASE" --run-pass "$RUN_PASS" ${DIRTY_FLAG} || RC=$?

# Peak memory of the whole job from the cgroup, when readable (also after a failure).
F=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_"${SLURM_JOB_ID:-none}"/memory.peak
if [ -r "$F" ]; then echo "[sweep-ref] job_memory_peak_bytes=$(cat "$F")"; fi

echo "Job finished at: $(date) (exit $RC)"
exit "$RC"
