#!/bin/bash
#SBATCH --job-name=timing
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=0:20

# Timing job: fine-tune one pilot entry for a few rollouts, then run the seeded selection eval on that run.
# Same code path as the real fine-tune; only the timesteps of this one entry are overridden in memory.
# Usage: sbatch cluster/timing.sh <run_pass>
#        plain bash: [TIMESTEPS=16384 EVAL_N=2 ALLOW_DIRTY=1] bash cluster/timing.sh <run_pass>
set -euo pipefail

RUN_PASS="$1"
EXP_NAME="${EXP_NAME:-ppo_5m700k_goal2_v3_pilot_b0.0_s0}"
TIMESTEPS="${TIMESTEPS:-32768}"
EVAL_N="${EVAL_N:-40}"
DIRTY_FLAG="${ALLOW_DIRTY:+--allow-dirty}"

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EXP_NAME TIMESTEPS RUN_PASS DIRTY_FLAG

python -c 'import os; print("cpus available (sched_getaffinity):", len(os.sched_getaffinity(0)))'

FINETUNE='
import os, resource, runpy, sys, time
sys.path.insert(0, "scripts")
from src.experiments.finetune_hparams import FINETUNE_HPARAMS
from src.experiments.registry import get_experiment

if __name__ == "__main__":
    name = os.environ["EXP_NAME"]
    timesteps = int(os.environ["TIMESTEPS"])
    exp = get_experiment(name)
    FINETUNE_HPARAMS[exp["base_exp"]]["timesteps"] = timesteps
    exp["timesteps"] = timesteps
    sys.argv = ["finetune_policy.py", name, "--run-pass", os.environ["RUN_PASS"], *os.environ["DIRTY_FLAG"].split()]
    t0 = time.time()
    runpy.run_path("scripts/finetune_policy.py", run_name="__main__")
    print(f"[timing] finetune_wall_s={time.time() - t0:.1f} timesteps={timesteps}")
    print(f"[timing] max_rss_main_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}"
          f" max_rss_largest_child_mb={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.0f}")
'
T0=$SECONDS
${SLURM_JOB_ID:+srun} python -c "$FINETUNE"
echo "[timing] finetune_total_s=$((SECONDS - T0))"

# The run dir name ends in the job id under Slurm; under plain bash take the newest.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  RUN_DIR="$(ls -d runs/finetune/"$EXP_NAME"/*_"${RUN_PASS}"_"${SLURM_JOB_ID}" | head -1)"
else
  RUN_DIR="$(ls -dt runs/finetune/"$EXP_NAME"/* | head -1)"
fi
[ -d "$RUN_DIR" ] || { echo "No run dir found for $EXP_NAME" >&2; exit 1; }
echo "[timing] run_dir=$RUN_DIR"

T1=$SECONDS
${SLURM_JOB_ID:+srun} python scripts/evaluate_seeded.py "$EXP_NAME" --run-dir "$RUN_DIR" --pool selection --n "$EVAL_N" --run-pass "$RUN_PASS" ${DIRTY_FLAG}
echo "[timing] eval_total_s=$((SECONDS - T1)) n=$EVAL_N"

# Peak memory of the whole job from the cgroup, when readable.
for f in /sys/fs/cgroup/system.slice/slurmstepd.scope/job_"${SLURM_JOB_ID:-none}"/memory.peak; do
  [ -r "$f" ] && echo "[timing] job_memory_peak_bytes=$(cat "$f")"
done

echo "Job finished at: $(date)"
