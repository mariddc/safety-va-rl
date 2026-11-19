#!/bin/bash
#SBATCH --job-name=packing
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=36G
#SBATCH --time=0:30:00

# Packing test: N_PAR fine-tunes run in parallel inside one job for a few rollouts each, no eval.
# Same code path as the real fine-tune; only timesteps (and n_envs, for dry runs only) are overridden in memory.
# Usage: sbatch [--cpus-per-task=16] cluster/packing_test.sh <run_pass>
#        plain bash: [TIMESTEPS=4096 N_ENVS=1 N_PAR=3 ALLOW_DIRTY=1] bash cluster/packing_test.sh <run_pass>
set -euo pipefail

RUN_PASS="$1"
TIMESTEPS="${TIMESTEPS:-32768}"
N_PAR="${N_PAR:-3}"
N_ENVS="${N_ENVS:-}"
DIRTY_FLAG="${ALLOW_DIRTY:+--allow-dirty}"
EXPS=(ppo_5m700k_goal2_v3_pilot_b0.0_s0 ppo_5m700k_goal2_v3_pilot_b0.0_s1 ppo_5m700k_goal2_v3_pilot_b0.0_s2)
JOB="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
CG="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID:-none}"

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
export TIMESTEPS RUN_PASS DIRTY_FLAG N_ENVS

CPUS="$(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
echo "cpus available (sched_getaffinity): $CPUS"

FINETUNE='
import os, resource, runpy, sys, time
sys.path.insert(0, "scripts")
from src.experiments.finetune_hparams import FINETUNE_HPARAMS
from src.experiments.registry import get_experiment

if __name__ == "__main__":
    name = os.environ["EXP_NAME"]
    exp = get_experiment(name)
    over = {"timesteps": int(os.environ["TIMESTEPS"])}
    if os.environ["N_ENVS"]:
        over["n_envs"] = int(os.environ["N_ENVS"])
    FINETUNE_HPARAMS[exp["base_exp"]].update(over)
    exp.update(over)
    sys.argv = ["finetune_policy.py", name, "--run-pass", os.environ["RUN_PASS"], *os.environ["DIRTY_FLAG"].split()]
    t0 = time.time()
    runpy.run_path("scripts/finetune_policy.py", run_name="__main__")
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    ts, ne = exp["timesteps"], exp["n_envs"]
    print(f"[packing] finetune_wall_s={time.time() - t0:.1f} timesteps={ts} n_envs={ne}")
    print(f"[packing] cpu_s_main={s.ru_utime + s.ru_stime:.0f} cpu_s_children={c.ru_utime + c.ru_stime:.0f}")
    print(f"[packing] max_rss_main_mb={s.ru_maxrss / 1024:.0f} max_rss_largest_child_mb={c.ru_maxrss / 1024:.0f}")
'

usage_usec() { if [ -r "$CG/cpu.stat" ]; then awk '$1 == "usage_usec" {print $2}' "$CG/cpu.stat"; else echo NA; fi; }

U0="$(usage_usec)"; W0="$(date +%s.%N)"
PIDS=()
for ((i = 0; i < N_PAR; i++)); do
  LOG="cluster/packing_${JOB}_${i}.out"
  EXP_NAME="${EXPS[$i]}" python -c "$FINETUNE" > "$LOG" 2>&1 &
  PIDS+=($!)
  echo "[packing] ft$i exp=${EXPS[$i]} pid=$! log=$LOG"
done
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "[packing] ft$i failed, see cluster/packing_${JOB}_${i}.out" >&2; FAIL=1; }
done
U1="$(usage_usec)"; W1="$(date +%s.%N)"

# Mean time per rollout of one fine-tune = SB3 time_elapsed at iteration 2 / 2.
ROLLS=()
for ((i = 0; i < N_PAR; i++)); do
  LOG="cluster/packing_${JOB}_${i}.out"
  TE=($(grep -oE 'time_elapsed +\| +[0-9]+' "$LOG" | awk '{print $NF}' || true))
  ROLL="$(awk -v t="${TE[1]:-}" 'BEGIN {print ((t == "") ? "NA" : t / 2)}')"
  ROLLS+=("$ROLL")
  echo "[packing] ft$i time_elapsed_it1=${TE[0]:-NA} time_elapsed_it2=${TE[1]:-NA} s_per_rollout=$ROLL"
  grep '^\[packing\]' "$LOG" | sed "s/^\[packing\]/[packing] ft$i/" || true
done

WALL="$(awk -v a="$W0" -v b="$W1" 'BEGIN {printf "%.1f", b - a}')"
CORES="$(awk -v u0="$U0" -v u1="$U1" -v w="$WALL" 'BEGIN {print ((u0 == "NA" || u1 == "NA") ? "NA" : sprintf("%.2f", (u1 - u0) / (w * 1e6)))}')"
MEAN_ROLL="$(printf '%s\n' "${ROLLS[@]}" | awk '$1 == "NA" {bad = 1} {s += $1; n++} END {print bad ? "NA" : sprintf("%.1f", s / n)}')"
PEAK="$( [ -r "$CG/memory.peak" ] && cat "$CG/memory.peak" || echo NA)"
PEAK_GB="$(awk -v p="$PEAK" 'BEGIN {print ((p == "NA") ? "NA" : sprintf("%.1f", p / 1e9))}')"
echo "[packing] cgroup usage_usec_start=$U0 usage_usec_end=$U1 parallel_wall_s=$WALL mean_cores_used=$CORES memory_peak_bytes=$PEAK"
echo "[packing] SUMMARY job=$JOB node=$(hostname) cpus=$CPUS n_par=$N_PAR n_envs=${N_ENVS:-8} timesteps=$TIMESTEPS" \
  "mean_s_per_rollout=$MEAN_ROLL per_ft=$(IFS=,; echo "${ROLLS[*]}") parallel_wall_s=$WALL mean_cores=$CORES" \
  "memory_peak_gb=$PEAK_GB status=$([ "$FAIL" = 0 ] && echo ok || echo failed)"

echo "Job finished at: $(date)"
exit "$FAIL"
