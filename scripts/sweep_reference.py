"""One beta on given seeds of a training pool: the same fine-tunes and evals as one sweep trial, without the study.

Results go to runs/sweeps/<base>_<run_pass>_<pool>_b<beta>_s<first>-<last>/runs.json (<base>_<run_pass>_ref_b0 for
beta 0 on sweep seeds 0,1,2), outside every study, so they never reach the sampler. Runs of two betas on the same
pool and seeds differ only in beta.
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

from src.sweeps.finetune_sweep import OBJECTIVE_KEY, _run_seeds
from src.utils.run_meta import CODE_ROOT, git_info, repo_relative, sha256_file
from src.utils.seed_pools import load_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate one beta on given seeds of a training pool.")
    parser.add_argument("base", choices=list(OBJECTIVE_KEY))
    parser.add_argument("--run-pass", required=True, help="Round tag in run dir names and meta.json")
    parser.add_argument("--allow-dirty", action="store_true", help="Run with uncommitted code changes (not for reported runs)")
    parser.add_argument("--n-eval", type=int, default=500, help="Evaluate on the first n selection seeds")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--beta", type=float, default=0.0)
    source.add_argument("--beta-file", type=Path, help='JSON with "beta", or with "betas" (pick one with --rank)')
    parser.add_argument("--rank", type=int, help='Index into the "betas" of --beta-file')
    parser.add_argument("--pool", choices=("sweep", "confirm", "final"), default="sweep", help="Training seed pool")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated indices into the pool")
    args = parser.parse_args()

    beta, beta_file = args.beta, None
    if args.beta_file:
        data = json.loads(args.beta_file.read_text())
        beta = float(data["betas"][args.rank] if "betas" in data else data["beta"])
        beta_file = {"path": repo_relative(args.beta_file), "sha256": sha256_file(args.beta_file), "rank": args.rank}
    seed_indices = [int(i) for i in args.seeds.split(",")]

    git = git_info()
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(f"uncommitted code changes {git['dirty_paths']}; commit them or pass --allow-dirty")
    if (beta, args.pool, seed_indices) == (0.0, "sweep", [0, 1, 2]):
        out_name = f"{args.base}_{args.run_pass}_ref_b0"
    else:
        out_name = f"{args.base}_{args.run_pass}_{args.pool}_b{beta!r}_s{seed_indices[0]}-{seed_indices[-1]}"
    out_dir = CODE_ROOT / "runs" / "sweeps" / out_name
    if (out_dir / "runs.json").exists():
        raise FileExistsError(f"{out_dir / 'runs.json'} already exists")
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    runs = _run_seeds(args.base, beta, run_pass=args.run_pass, n_eval=args.n_eval, allow_dirty=args.allow_dirty,
                      serial=False, commit=git["commit"], log_prefix=out_dir / "ref", pool=args.pool,
                      seed_indices=seed_indices)
    key = OBJECTIVE_KEY[args.base]
    value = sum(r[key] for r in runs) / len(runs)
    pool = load_pool(args.pool)
    (out_dir / "runs.json").write_text(json.dumps({
        "beta": beta, "beta_file": beta_file, "pool": args.pool, "seed_indices": seed_indices,
        "seeds": [pool[i] for i in seed_indices], "code_commit": git["commit"], "dirty": git["dirty"],
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "wall_time_s": time.monotonic() - start, "objective_key": key, "objective": value, "runs": runs,
    }, indent=2))
    print(f"[sweep-ref] {args.base}: beta={beta!r} {args.pool}{seed_indices} {key}={value:.4f}")


if __name__ == "__main__":
    main()
