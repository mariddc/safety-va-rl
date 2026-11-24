#!/bin/bash
#SBATCH --job-name=render_videos
#SBATCH --output=cluster/%x_%j.out
#SBATCH --error=cluster/%x_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --no-requeue

# Usage: sbatch --nodelist=<test host> cluster/render_episode_videos.sh <test index json> <train seed> <episode seed> [<train seed> <episode seed> ...]
#        plain bash: [OUT_DIR=...] [MUJOCO_GL=osmesa] bash cluster/render_episode_videos.sh ...
# Replays must run on the host that ran the test (recorded in the index): on other hardware the same
# seed can give a different episode. Each replay is checked against the recorded episodes.csv row.
set -euo pipefail

INDEX="$1"
shift

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

# Same thread settings as the test job
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

while [ "$#" -ge 2 ]; do
    python scripts/render_episode_videos.py "$INDEX" --train-seed "$1" --episode-seed "$2" ${OUT_DIR:+--out-dir "$OUT_DIR"}
    shift 2
done

echo "Job finished at: $(date)"
