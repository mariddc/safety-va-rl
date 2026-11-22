"""The one test of a base: the base policy and its 6 ft + 6 ref final runs on the first n seeds of a pool.

Each eval goes to <run_dir>/final_eval/<pool>_n<n>/ (as scripts/evaluate_seeded.py), and an index of the 13 eval
dirs to runs/final_test/<base>_<run_pass>_<pool>_n<n>.json. Every check runs before anything is written.
A re-run after a job died part-way keeps the finished evals, redoes an unfinished one and evaluates the rest.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_seeded import evaluate_seeded
from src.experiments.registry import get_experiment, get_run_dir
from src.sweeps.finetune_sweep import OBJECTIVE_KEY
from src.utils.run_meta import CODE_ROOT, git_info, repo_relative, sha256_file
from src.utils.seed_pools import SEED_POOLS_PATH, load_pool

N_FINAL = 6
TEST_POOL_SIZE = 1000  # the test is always the whole test pool


def _newest_finished(name: str, run_pass: str) -> tuple[Path, dict]:
    for run_dir in sorted((CODE_ROOT / "runs" / "finetune" / name).glob(f"*_{run_pass}_*"), reverse=True):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        # The glob also matches longer tags that start with run_pass
        if meta["status"] == "finished" and meta["pass"] == run_pass:
            return run_dir, meta
    raise FileNotFoundError(f"{name}: no finished run at {run_pass}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a base and its final ft and ref runs, once, on the test pool.")
    parser.add_argument("base", choices=list(OBJECTIVE_KEY))
    parser.add_argument("--run-pass", required=True,
                        help="Round tag of the final runs, also stored in the eval meta.json")
    parser.add_argument("--chosen", type=Path, required=True, help="chosen_beta.json of this base")
    parser.add_argument("--pool", required=True, choices=("test", "selection", "diagnostic"))
    parser.add_argument("--n", type=int, default=TEST_POOL_SIZE, help="Use the first n seeds of the pool")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run with uncommitted code changes (not with the test pool)")
    args = parser.parse_args()

    if args.pool == "test" and args.allow_dirty:
        raise ValueError("--allow-dirty is refused with the test pool")
    if args.pool == "test" and args.n != TEST_POOL_SIZE:
        raise ValueError(f"--n {args.n}: the test uses all {TEST_POOL_SIZE} seeds of the test pool")
    git = git_info()
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(f"uncommitted code changes {git['dirty_paths']}; commit them first")
    git["dirty_allowed"] = args.allow_dirty

    # ft and ref differ only in beta: same commit, timesteps and Q, seeds final[0..5]
    chosen = json.loads(args.chosen.read_text())
    if chosen["base"] != args.base:
        raise ValueError(f"{args.chosen} is for base {chosen['base']}, not {args.base}")
    beta = chosen["beta"]
    final_seeds = load_pool("final")
    frozen_q = json.loads((CODE_ROOT / "configs" / "frozen.json").read_text())["q"][args.base]["sha256"]
    runs = {}
    for arm_beta in (beta, 0.0):
        for i in range(N_FINAL):
            name = f"{args.base}_v3_final_b{arm_beta!r}_s{i}"
            run_dir, meta = _newest_finished(name, args.run_pass)
            if meta["git"]["dirty"] and not args.allow_dirty:
                raise ValueError(f"{name}: {run_dir.name} ran with uncommitted code")
            if meta["beta"] != arm_beta or meta["seed"] != final_seeds[i]:
                raise ValueError(f"{name}: beta {meta['beta']} seed {meta['seed']}, "
                                 f"expected {arm_beta} {final_seeds[i]}")
            if meta["q_sha256"] != frozen_q or meta["q_hash_check"] != "matched":
                raise ValueError(f"{name}: Q {meta['q_sha256']} ({meta['q_hash_check']}) "
                                 f"is not the frozen {frozen_q}")
            runs[name] = (run_dir, meta)
    commits = {m["git"]["commit"] for _, m in runs.values()}
    timesteps = {m["timesteps"] for _, m in runs.values()}
    if len(commits) != 1 or len(timesteps) != 1:
        raise ValueError(f"{args.base}: final runs differ: commits {sorted(commits)}, "
                         f"timesteps {sorted(timesteps)}")

    seeds = load_pool(args.pool, allow_test=args.pool == "test")[: args.n]
    out_name = f"{args.pool}_n{len(seeds)}"
    evals = [(args.base, CODE_ROOT / get_run_dir(get_experiment(args.base)))]
    evals += [(name, run_dir) for name, (run_dir, _) in runs.items()]
    # The test runs once: only its own output, left by a job that died part-way, may exist
    other = [repo_relative(p) for _, d in evals for p in (d / "final_eval").glob("test_*") if p.name != out_name]
    if other:
        raise RuntimeError(f"other test output exists: {other}")
    index_path = CODE_ROOT / "runs" / "final_test" / f"{args.base}_{args.run_pass}_{out_name}.json"
    if index_path.exists():
        raise FileExistsError(f"{index_path} exists: this test already ran to the end")

    index = {"base": args.base, "pool": args.pool, "n": len(seeds), "beta": beta,
             "chosen": {"path": str(args.chosen), "sha256": sha256_file(args.chosen)},
             "seed_pools_file": repo_relative(SEED_POOLS_PATH), "seed_pools_sha256": sha256_file(SEED_POOLS_PATH),
             "evals": {}}
    for name, run_dir in evals:
        out_dir = run_dir / "final_eval" / out_name
        eval_meta = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").exists() else {}
        if eval_meta.get("status") == "finished" and (out_dir / "summary.json").exists():
            results = json.loads((out_dir / "summary.json").read_text())
            evaluated = "earlier job"
        else:
            evaluated = "this job"
            if out_dir.exists():
                shutil.rmtree(out_dir)
                evaluated = "this job, unfinished earlier dir removed"
            results, out_dir = evaluate_seeded(name, get_experiment(name), run_dir.resolve(), args.pool, seeds,
                                               args.run_pass, git)
            eval_meta = json.loads((out_dir / "meta.json").read_text())
        index["evals"][name] = {"eval_dir": repo_relative(out_dir), "evaluated": evaluated,
                                "job_id": eval_meta["slurm_job_id"], "hostname": eval_meta["hostname"]}
        print(f"[final_test] {name} ({evaluated}): "
              f"{OBJECTIVE_KEY[args.base]}={results[OBJECTIVE_KEY[args.base]]:.4f}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2))
    print(f"[final_test] index={index_path}")


if __name__ == "__main__":
    main()
