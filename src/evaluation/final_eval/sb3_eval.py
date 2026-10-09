from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

import safety_gymnasium
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.wrappers.FastSafeCompleteRewardWrapper import FastSafeCompleteRewardWrapper


def make_env(env_id: str):
    env = safety_gymnasium.make(env_id)
    env = FastSafeCompleteRewardWrapper(env)
    return env


def run_sb3_final_eval(exp: dict, run_dir: Path, n_episodes: int = 100):
    env_id = exp["env_id"]
    algo = exp["algorithm"]

    model_path = run_dir / "model" / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = DummyVecEnv([lambda: make_env(env_id)])

    if algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported SB3 algorithm: {algo}")

    episode_rewards = []
    episode_lengths = []
    episode_costs = []
    episode_successes = []

    episode_hazard_costs = []
    episode_hazard_steps = []
    episode_hazard_entries = []

    episode_vase_contact_costs = []
    episode_vase_velocity_costs = []
    episode_vase_contact_steps = []
    episode_vase_velocity_steps = []

    for _ in range(n_episodes):
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

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_costs.append(ep_cost)
        episode_successes.append(ep_success)

        episode_hazard_costs.append(haz_cost)
        episode_hazard_steps.append(haz_steps)
        episode_hazard_entries.append(haz_entries)

        episode_vase_contact_costs.append(vase_contact_cost)
        episode_vase_velocity_costs.append(vase_velocity_cost)
        episode_vase_contact_steps.append(vase_contact_steps)
        episode_vase_velocity_steps.append(vase_velocity_steps)

    env.close()

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

    out_dir = run_dir / "final_eval"
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

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
