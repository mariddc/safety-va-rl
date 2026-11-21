"""beta = 0 reference on the sweep seeds: the same fine-tunes and evals as one sweep trial, without the study.

Results go to runs/sweeps/<base>_<run_pass>_ref_b0/runs.json, outside every study, so they never reach the sampler.
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sweeps.finetune_sweep import N_SEEDS, OBJECTIVE_KEY, _run_seeds
from src.utils.run_meta import CODE_ROOT, git_info
from src.utils.seed_pools import load_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate the sweep seeds of a base at beta = 0.")
    parser.add_argument("base", choices=list(OBJECTIVE_KEY))
    parser.add_argument("--run-pass", required=True, help="Round tag in run dir names and meta.json")
    parser.add_argument("--allow-dirty", action="store_true", help="Run with uncommitted code changes (not for reported runs)")
    parser.add_argument("--n-eval", type=int, default=500, help="Evaluate on the first n selection seeds")
    args = parser.parse_args()

    git = git_info()
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(f"uncommitted code changes {git['dirty_paths']}; commit them or pass --allow-dirty")
    out_dir = CODE_ROOT / "runs" / "sweeps" / f"{args.base}_{args.run_pass}_ref_b0"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    runs = _run_seeds(args.base, 0.0, run_pass=args.run_pass, n_eval=args.n_eval, allow_dirty=args.allow_dirty,
                      serial=False, commit=git["commit"], log_prefix=out_dir / "ref")
    key = OBJECTIVE_KEY[args.base]
    value = sum(r[key] for r in runs) / len(runs)
    (out_dir / "runs.json").write_text(json.dumps({
        "beta": 0.0, "seeds": load_pool("sweep")[:N_SEEDS], "code_commit": git["commit"], "dirty": git["dirty"],
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "wall_time_s": time.monotonic() - start, "objective_key": key, "objective": value, "runs": runs,
    }, indent=2))
    print(f"[sweep-ref] {args.base}: beta=0.0 {key}={value:.4f}")


if __name__ == "__main__":
    main()
