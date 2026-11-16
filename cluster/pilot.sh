#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --output=cluster/%x_%A_%a.out
#SBATCH --error=cluster/%x_%A_%a.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-11

# Usage: sbatch cluster/pilot.sh <run_pass> (whole array); plain bash: bash cluster/pilot.sh <run_pass> <index>
# Fine-tunes one pilot run, then evaluates it on the selection pool.
set -euo pipefail

RUN_PASS="$1"
INDEX="${2:-${SLURM_ARRAY_TASK_ID}}"

EXPS=(
  ppo_5m_goal2_safe_v3_pilot_b0.0_s0
  ppo_5m_goal2_safe_v3_pilot_b0.0_s1
  ppo_5m_goal2_safe_v3_pilot_b0.0_s2
  ppo_5m_goal2_safe_v3_pilot_b1.0_s0
  ppo_5m_goal2_safe_v3_pilot_b1.0_s1
  ppo_5m_goal2_safe_v3_pilot_b1.0_s2
  ppo_5m700k_goal2_v3_pilot_b0.0_s0
  ppo_5m700k_goal2_v3_pilot_b0.0_s1
  ppo_5m700k_goal2_v3_pilot_b0.0_s2
  ppo_5m700k_goal2_v3_pilot_b1.0_s0
  ppo_5m700k_goal2_v3_pilot_b1.0_s1
  ppo_5m700k_goal2_v3_pilot_b1.0_s2
)
EXP_NAME="${EXPS[$INDEX]}"

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
# The run dir name ends in the job id under Slurm; under plain bash take the newest.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  RUN_DIR="$(ls -d runs/finetune/"$EXP_NAME"/*_"${RUN_PASS}"_"${SLURM_JOB_ID}" | head -1)"
else
  RUN_DIR="$(ls -dt runs/finetune/"$EXP_NAME"/* | head -1)"
fi
[ -d "$RUN_DIR" ] || { echo "No run dir found for $EXP_NAME" >&2; exit 1; }

${SLURM_JOB_ID:+srun} python scripts/evaluate_seeded.py "$EXP_NAME" --run-dir "$RUN_DIR" --pool selection --n 500 --run-pass "$RUN_PASS"

echo "Job finished at: $(date)"
