import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.registry import get_experiment, get_run_dir
from src.evaluation.final_eval.sb3_eval import run_sb3_seeded_eval
from src.utils.run_meta import git_info, recorded_run, repo_relative, sha256_file
from src.utils.seed_pools import SEED_POOLS_PATH, load_pool
from src.wrappers.reward import reward_config


def evaluate_seeded(exp_name: str, exp: dict, run_dir: Path, pool: str, seeds: list[int], run_pass: str,
                    git: dict) -> tuple[dict, Path]:
    """Evaluate run_dir on seeds into run_dir/final_eval/<pool>_n<len(seeds)>/, with its meta.json."""
    model_path = run_dir / "model" / "model.zip"
    fields = dict(
        run_dir=repo_relative(run_dir), model_path=repo_relative(model_path), model_sha256=sha256_file(model_path),
        env_id=exp["env_id"], **reward_config(exp),
        seed_pool={"name": pool, "n": len(seeds), "file": repo_relative(SEED_POOLS_PATH),
                   "sha256": sha256_file(SEED_POOLS_PATH)},
        deterministic=True,
    )
    out_dir = run_dir / "final_eval" / f"{pool}_n{len(seeds)}"
    out_dir.mkdir(parents=True, exist_ok=False)
    with recorded_run(out_dir / "meta.json", exp_name, run_pass, git, **fields):
        results = run_sb3_seeded_eval(exp, run_dir, seeds, out_dir)
    return results, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic eval of one run on the first n seeds of a seed pool.")
    parser.add_argument("exp_name", help="Experiment name in the registry (env, reward, algorithm)")
    parser.add_argument("--run-dir", required=True, help="Run folder holding model/model.zip")
    parser.add_argument("--pool", required=True, choices=("selection", "diagnostic"),
                        help="Episode seed pool in configs/seed_pools.json")
    parser.add_argument("--n", type=int, required=True, help="Use the first n seeds of the pool")
    parser.add_argument("--run-pass", required=True, help="Round tag stored in meta.json")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run with uncommitted code changes (recorded as dirty_allowed=true; not for reported runs)")
    args = parser.parse_args()

    pool = load_pool(args.pool)
    if not 1 <= args.n <= len(pool):
        raise ValueError(f"--n {args.n} must be between 1 and {len(pool)}, the size of seed pool {args.pool}")
    seeds = pool[: args.n]
    exp = get_experiment(args.exp_name)

    # Checks before anything is written: a refused eval leaves nothing on disk.
    git = git_info()
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(f"{args.exp_name}: uncommitted code changes {git['dirty_paths']}; commit them or pass --allow-dirty")
    git["dirty_allowed"] = args.allow_dirty

    run_dir = Path(args.run_dir).resolve()
    # The env and reward come from exp_name, so the run dir must belong to that experiment.
    if "base_exp" in exp:
        meta_path = run_dir / "meta.json"
        ok = meta_path.exists() and json.loads(meta_path.read_text()).get("exp_name") == args.exp_name
    else:
        ok = run_dir == (PROJECT_ROOT / get_run_dir(exp)).resolve()
    if not ok:
        raise ValueError(f"{args.exp_name}: {args.run_dir} is not a run of this experiment")
    results, out_dir = evaluate_seeded(args.exp_name, exp, run_dir, args.pool, seeds, args.run_pass, git)
    print(f"[eval] success_rate={results['success_rate']:.3f} "
          f"safe_success_rate_hazards_only={results['safe_success_rate_hazards_only']:.3f} "
          f"safe_success_rate_strict={results['safe_success_rate_strict']:.3f}")
    print(f"[eval] out_dir={out_dir}")


if __name__ == "__main__":
    main()
