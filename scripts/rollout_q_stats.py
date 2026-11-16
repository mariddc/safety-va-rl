from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import safety_gymnasium
from stable_baselines3 import PPO, SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.registry import get_experiment, get_run_dir
from src.utils.run_meta import git_info, recorded_run, repo_relative, sha256_file
from src.utils.seed_pools import SEED_POOLS_PATH, load_pool
from src.wrappers.reward import wrap_reward, reward_config
from src.wrappers.InterventionRewardWrapper import InterventionRewardWrapper


def _resolve_model_path(run_dir: Path) -> Path:
    p1 = run_dir / "model" / "model.zip"
    if p1.exists():
        return p1
    p2 = run_dir / "model" / "model"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Model not found in {run_dir / 'model'}")


def _load_model(algo: str, model_path: Path):
    algo = algo.lower()
    if algo == "ppo":
        return PPO.load(str(model_path), device="cpu")
    if algo == "sac":
        return SAC.load(str(model_path), device="cpu")
    raise ValueError(f"Unsupported algorithm: {algo}")


def _stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()), "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Q statistics of the base policy on the first n diagnostic seeds.")
    parser.add_argument("exp_name", help="Fine-tune entry with q_path and base_exp")
    parser.add_argument("--n", type=int, required=True, help="Use the first n seeds of the diagnostic pool")
    parser.add_argument("--run-pass", required=True, help="Round tag stored in meta.json")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run with uncommitted code changes (recorded as dirty_allowed=true; not for reported runs)")
    args = parser.parse_args()

    exp = get_experiment(args.exp_name)
    pool = load_pool("diagnostic")
    if not 1 <= args.n <= len(pool):
        raise ValueError(f"--n {args.n} must be between 1 and {len(pool)}, the size of seed pool diagnostic")
    seeds = pool[: args.n]

    git = git_info()
    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(f"{args.exp_name}: uncommitted code changes {git['dirty_paths']}; commit them or pass --allow-dirty")
    git["dirty_allowed"] = args.allow_dirty

    # Always the base policy, never a fine-tuned run of this entry.
    base = get_experiment(exp["base_exp"])
    model_path = _resolve_model_path(ROOT / get_run_dir(base))
    q_path = ROOT / exp["q_path"]

    # B is not set here: the wrapper reads the Q-training B from the checkpoint.
    env = wrap_reward(safety_gymnasium.make(base["env_id"]), base)
    env = InterventionRewardWrapper(env, q_path=str(q_path), device=str(exp.get("q_device", "cpu")))

    model = _load_model(base["algorithm"], model_path)

    fields = dict(
        base_exp=exp["base_exp"], base_model_path=repo_relative(model_path), base_model_sha256=sha256_file(model_path),
        q_path=repo_relative(q_path), q_sha256=sha256_file(q_path), B=env.B,
        env_id=base["env_id"], **reward_config(base),
        seed_pool={"name": "diagnostic", "n": args.n, "file": repo_relative(SEED_POOLS_PATH),
                   "sha256": sha256_file(SEED_POOLS_PATH)},
        deterministic=True,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "runs" / "q_diagnostics" / exp["base_exp"] / f"{stamp}_{args.run_pass}_diagnostic_n{args.n}"
    out_dir.mkdir(parents=True, exist_ok=False)

    with recorded_run(out_dir / "meta.json", args.exp_name, args.run_pass, git, **fields):
        q_vals = []
        episodes = []
        for s in seeds:
            obs, _ = env.reset(seed=s)
            ep_q = []
            haz_steps, haz_cost, vase_cost = 0, 0.0, 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                ep_q.append(info["q_value"])
                # Same info keys and fallbacks as the final eval
                hz = float(info.get("hazard_cost_step", info.get("cost_hazards_used", info.get("cost_hazards", 0.0))))
                haz_cost += hz
                haz_steps += int(bool(info.get("hazard_in_contact", False)) or (hz > 0.0))
                vase_cost += float(info.get("cost_vases_contact", info.get("cost_vases_contact_used", 0.0)))
                vase_cost += float(info.get("cost_vases_velocity", info.get("cost_vases_velocity_used", 0.0)))
                done = terminated or truncated
            q_vals += ep_q
            episodes.append({"seed": s, "length": len(ep_q),
                             "active_frac": float((np.asarray(ep_q) > env.B).mean()),
                             "hazard_steps": haz_steps, "hazard_cost": haz_cost, "vase_cost": vase_cost})

        q_vals = np.asarray(q_vals)
        stats = {
            "steps": int(q_vals.size),
            "frac_q_gt_B": float((q_vals > env.B).mean()),
            "q": _stats(q_vals),
            "episode_active_frac": _stats([e["active_frac"] for e in episodes]),
            "episodes": episodes,
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"[rollout_q_stats] EXP={args.exp_name}  base={exp['base_exp']}  model={model_path}")
    print(f"[rollout_q_stats] B={env.B:+.6f} (from checkpoint)  reward={reward_config(base)}")
    print(f"[rollout_q_stats] Q: {stats['q']}")
    print(f"[rollout_q_stats] episode active fraction: {stats['episode_active_frac']}")
    print(f"[rollout_q_stats] frac Q>B = {stats['frac_q_gt_B']:.3f}  (steps={stats['steps']}, episodes={len(episodes)})")
    print(f"[rollout_q_stats] out_dir={out_dir}")


if __name__ == "__main__":
    main()
