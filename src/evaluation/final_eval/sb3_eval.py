from pathlib import Path
import csv
import json
import multiprocessing as mp
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import safety_gymnasium
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.wrappers.reward import wrap_reward, reward_config


def make_env(exp: dict):
    return wrap_reward(safety_gymnasium.make(exp["env_id"]), exp)


def _eval_episodes(exp: dict, model_path: Path, n_episodes: int, seeds=None):
    """Play episodes; returns one tuple of per-episode metrics each."""
    algo = exp["algorithm"]
    env = DummyVecEnv([lambda: make_env(exp)])

    if algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported SB3 algorithm: {algo}")

    records = []
    for ep in range(n_episodes):
        if seeds is not None:
            # DummyVecEnv passes this seed to the next reset() only
            env.seed(seeds[ep])
        obs = env.reset()
        done = False

        ep_reward = 0.0
        ep_length = 0
        ep_cost = 0.0
        ep_success = 0.0

        haz_cost = 0.0
        haz_steps = 0
        haz_entries = 0

        vase_contact_cost = 0.0
        vase_velocity_cost = 0.0
        vase_contact_steps = 0
        vase_velocity_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            info = infos[0]
            done_flag = done[0]

            ep_reward += float(reward[0])
            ep_length += 1

            if "cost" in info:
                ep_cost += float(info["cost"])

            if info.get("goal_met_used", False) or info.get("goal_met", False) or info.get("is_success", False) or info.get("success", False):
                ep_success = 1.0

            hz = float(info.get("hazard_cost_step", info.get("cost_hazards_used", info.get("cost_hazards", 0.0))))
            haz_cost += hz
            haz_steps += int(bool(info.get("hazard_in_contact", False)) or (hz > 0.0))
            haz_entries += int(bool(info.get("hazard_entry", False)))

            vc = float(info.get("cost_vases_contact", info.get("cost_vases_contact_used", 0.0)))
            vv = float(info.get("cost_vases_velocity", info.get("cost_vases_velocity_used", 0.0)))
            vase_contact_cost += vc
            vase_velocity_cost += vv
            vase_contact_steps += int(vc > 0.0)
            vase_velocity_steps += int(vv > 0.0)

            if done_flag:
                break

        records.append((
            ep_reward, ep_length, ep_cost, ep_success,
            haz_cost, haz_steps, haz_entries,
            vase_contact_cost, vase_velocity_cost, vase_contact_steps, vase_velocity_steps,
        ))

    env.close()
    return records


def _init_worker():
    torch.set_num_threads(1)


def _eval_chunk(args):
    return _eval_episodes(*args)


def run_sb3_final_eval(exp: dict, run_dir: Path, n_episodes: int = 100, *, seeds=None, out_dir=None):
    if seeds is not None:
        n_episodes = len(seeds)
    env_id = exp["env_id"]
    algo = exp["algorithm"]

    model_path = run_dir / "model" / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Seeded episodes are independent, so chunks of seeds run in separate processes.
    n_workers = min(len(os.sched_getaffinity(0)), n_episodes) if seeds is not None else 1
    if n_workers > 1:
        size = -(-n_episodes // n_workers)
        chunks = [(exp, model_path, len(seeds[i:i + size]), seeds[i:i + size]) for i in range(0, n_episodes, size)]
        with mp.get_context("spawn").Pool(n_workers, initializer=_init_worker) as pool:
            records = [r for chunk in pool.map(_eval_chunk, chunks) for r in chunk]
    else:
        records = _eval_episodes(exp, model_path, n_episodes, seeds)

    (episode_rewards, episode_lengths, episode_costs, episode_successes,
     episode_hazard_costs, episode_hazard_steps, episode_hazard_entries,
     episode_vase_contact_costs, episode_vase_velocity_costs,
     episode_vase_contact_steps, episode_vase_velocity_steps) = map(list, zip(*records))

    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    costs = np.array(episode_costs)
    successes = np.array(episode_successes)

    haz_costs = np.array(episode_hazard_costs)
    haz_steps = np.array(episode_hazard_steps)
    haz_entries = np.array(episode_hazard_entries)

    vase_contact_costs = np.array(episode_vase_contact_costs)
    vase_velocity_costs = np.array(episode_vase_velocity_costs)
    vase_contact_steps = np.array(episode_vase_contact_steps)
    vase_velocity_steps = np.array(episode_vase_velocity_steps)

    success_mask = successes.astype(bool)

    safe_haz_mask = (haz_steps == 0)
    safe_vase_touch_mask = (vase_contact_steps == 0)
    strict_safe_mask = safe_haz_mask & safe_vase_touch_mask

    results = {
        "environment": env_id,
        "algorithm": algo,
        "n_episodes": n_episodes,
        **reward_config(exp),
        "reward": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
        },
        "episode_length": {
            "mean": float(lengths.mean()),
            "time_to_goal_success_only": float(lengths[success_mask].mean()) if success_mask.any() else None,
        },
        "cost": {
            "mean": float(costs.mean()),
            "max": float(costs.max()),
            "unsafe_rate": float((costs > 0).mean()),
        },
        "success_rate": float(successes.mean()),
        "safety_rate_hazards_only": float(safe_haz_mask.mean()),
        "safety_rate_vases_touch_only": float(safe_vase_touch_mask.mean()),
        "strict_safety_rate": float(strict_safe_mask.mean()),
        "safe_success_rate_hazards_only": float((success_mask & safe_haz_mask).mean()),
        "safe_success_rate_strict": float((success_mask & strict_safe_mask).mean()),
        "safety_breakdown": {
            "hazards": {
                "any_contact_rate": float((haz_steps > 0).mean()),
                "entries_mean": float(haz_entries.mean()),
                "contact_steps_mean": float(haz_steps.mean()),
                "cost_sum_mean": float(haz_costs.mean()),
                "entries_max": int(haz_entries.max()) if haz_entries.size else 0,
                "contact_steps_max": int(haz_steps.max()) if haz_steps.size else 0,
                "cost_sum_max": float(haz_costs.max()) if haz_costs.size else 0.0,
            },
            "vases": {
                "any_touch_rate": float((vase_contact_steps > 0).mean()),
                "touch_steps_mean": float(vase_contact_steps.mean()),
                "touch_cost_sum_mean": float(vase_contact_costs.mean()),
                "touch_steps_max": int(vase_contact_steps.max()) if vase_contact_steps.size else 0,
                "touch_cost_sum_max": float(vase_contact_costs.max()) if vase_contact_costs.size else 0.0,
                "any_velocity_cost_rate": float((vase_velocity_steps > 0).mean()),
                "velocity_steps_mean": float(vase_velocity_steps.mean()),
                "velocity_cost_sum_mean": float(vase_velocity_costs.mean()),
                "velocity_steps_max": int(vase_velocity_steps.max()) if vase_velocity_steps.size else 0,
                "velocity_cost_sum_max": float(vase_velocity_costs.max()) if vase_velocity_costs.size else 0.0,
            },
        },
    }

    if seeds is not None:
        results["seeds"] = seeds

    out_dir = Path(out_dir) if out_dir is not None else run_dir / "final_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if seeds is not None:
        with open(out_dir / "episodes.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed", "success", "return", "length", "cost",
                "hazard_steps", "hazard_entries", "hazard_cost",
                "vase_contact_steps", "vase_contact_cost", "vase_velocity_steps", "vase_velocity_cost",
                "safe_success_hazards_only", "safe_success_strict",
            ])
            writer.writerows(zip(
                seeds, success_mask.astype(int), episode_rewards, episode_lengths, episode_costs,
                episode_hazard_steps, episode_hazard_entries, episode_hazard_costs,
                episode_vase_contact_steps, episode_vase_contact_costs, episode_vase_velocity_steps, episode_vase_velocity_costs,
                (success_mask & safe_haz_mask).astype(int), (success_mask & strict_safe_mask).astype(int),
            ))

    plt.figure(figsize=(8, 6))
    plt.hist(rewards, bins=30, edgecolor="black")
    plt.xlabel("Episode return")
    plt.ylabel("Number of episodes")
    plt.title("Distribution of episode returns")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_dir / "reward_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(costs, bins=30, edgecolor="black")
    plt.xlabel("Episode safety cost")
    plt.ylabel("Number of episodes")
    plt.title("Distribution of episode safety costs")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_dir / "cost_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()

    return results


def run_sb3_seeded_eval(exp: dict, run_dir: Path, seeds: list[int], out_dir: Path):
    """One deterministic episode per reset seed; seeds come from src.utils.seed_pools."""
    if not seeds:
        raise ValueError("seeded eval needs a non-empty seed list")
    return run_sb3_final_eval(exp, run_dir, seeds=seeds, out_dir=out_dir)
