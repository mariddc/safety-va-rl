from pathlib import Path
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
import safety_gymnasium

from src.trainers.omnisafe.compat import patch_linear_lr
patch_linear_lr()

import omnisafe

ALGO_MAP = {"ppolag": "PPOLag"}


def _max_epoch_in_seed_dir(seed_dir: Path) -> int:

    torch_dir = seed_dir / "torch_save"
    if not torch_dir.is_dir():
        return -1

    epochs = []
    for p in torch_dir.glob("epoch-*.pt"):
        m = re.search(r"epoch-(\d+)\.pt", p.name)
        if m:
            epochs.append(int(m.group(1)))

    return max(epochs) if epochs else -1


def run_omnisafe_final_eval(exp: dict, run_dir: Path, n_episodes: int = 50):
    env_id = exp["env_id"]
    algo = ALGO_MAP[exp["algorithm"].lower()]

    seed_root = run_dir / "logs"

    seed_dirs = [
        p for p in seed_root.glob("**/seed-*")
        if (p / "torch_save").is_dir()
    ]

    if not seed_dirs:
        raise RuntimeError(
            f"No valid OmniSafe seed directory with torch_save found in {seed_root}"
        )

    seed_dir = max(seed_dirs, key=_max_epoch_in_seed_dir)
    max_epoch = _max_epoch_in_seed_dir(seed_dir)


    if exp.get("timesteps", 0) > 50_000 and max_epoch == 0:
        raise RuntimeError(
            f"Refusing to evaluate: only epoch-0.pt found for "
            f"timesteps={exp['timesteps']} in {seed_dir}"
        )

    torch_save_dir = seed_dir / "torch_save"
    ckpts = list(torch_save_dir.glob("epoch-*.pt"))

    def epoch_num(p: Path):
        m = re.search(r"epoch-(\d+)\.pt", p.name)
        return int(m.group(1)) if m else -1

    ckpt = max(ckpts, key=epoch_num)
    print(f"[FINAL EVAL] Reporting checkpoint: {ckpt.name}")


    agent = omnisafe.Agent(
        algo=algo,
        env_id=env_id,
        custom_cfgs={"logger_cfgs": {"log_dir": str(seed_dir.parent)}}
    )

    state = torch.load(ckpt, map_location="cpu")
    agent.agent._actor_critic.actor.load_state_dict(state["pi"])
    agent.agent._actor_critic.actor.eval()

    p = next(agent.agent._actor_critic.actor.parameters())
    print("[DEBUG] Actor weight mean:", p.mean().item())


    # Evaluation
    EVAL_SEED = 12
    env = safety_gymnasium.make(env_id)

    rewards, lengths, costs = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        done = False
        ep_r, ep_l, ep_c = 0.0, 0, 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                dist = agent.agent._actor_critic.actor(obs_t)
                action = dist.mean.cpu().numpy()[0]  # deterministic

            obs, reward, cost, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_r += float(reward)
            ep_c += float(cost)
            ep_l += 1

        rewards.append(ep_r)
        lengths.append(ep_l)
        costs.append(ep_c)

    env.close()

    rewards = np.asarray(rewards)
    lengths = np.asarray(lengths)
    costs = np.asarray(costs)


    out_dir = run_dir / "final_eval"
    out_dir.mkdir(exist_ok=True)

    summary = {
        "environment": env_id,
        "algorithm": algo,
        "checkpoint": ckpt.name,
        "max_epoch": max_epoch,
        "n_episodes": n_episodes,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "episode_length_mean": float(lengths.mean()),
        "cost_mean": float(costs.mean()),
        "cost_max": float(costs.max()),
        "unsafe_rate": float((costs > 0).mean()),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 6))
    plt.hist(rewards, bins=30, edgecolor="black")
    plt.xlabel("Episode return")
    plt.ylabel("Number of episodes")
    plt.title("Distribution of episode returns")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_dir / "reward_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(costs, bins=30, edgecolor="black")
    plt.xlabel("Episode safety cost")
    plt.ylabel("Number of episodes")
    plt.title("Distribution of episode safety costs")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_dir / "cost_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()
