from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.registry import get_experiment, get_run_dir
from src.eil.q_penalty_callback import QPenaltyStatsCallback
from src.wrappers.FastSafeRewardWrapper import FastSafeRewardWrapper
from src.wrappers.InterventionRewardWrapper import InterventionRewardWrapper


def _resolve_model_path(run_dir: Path) -> Path:
    p1 = run_dir / "model" / "model.zip"
    if p1.exists():
        return p1
    p2 = run_dir / "model" / "model"
    if p2.exists():
        return p2 
    raise FileNotFoundError(f"Model not found in {run_dir / 'model'}")


def make_env(env_id: str, rank: int, seed: int, *, q_path: Path, beta: float, B: float, q_device: str):
    def _init():
        env = safety_gymnasium.make(env_id)
        env = FastSafeRewardWrapper(env)
        env = InterventionRewardWrapper(env, q_path=str(q_path), beta=beta, B=B, device=q_device)
        env.reset(seed=seed + rank)
        return env
    return _init


def finetune_from_registry(exp_name: str) -> Path:
    exp = get_experiment(exp_name)

    base_exp_name = exp["base_exp"]
    q_path = (ROOT / exp["q_path"]).resolve()
    beta = float(exp["beta"])
    B = float(exp["B"])
    q_device = str(exp["q_device"])


    env_id = exp["env_id"]
    timesteps = int(exp["timesteps"])
    seed = int(exp.get("seed", 0))
    n_envs = int(exp.get("n_envs", int(os.environ.get("N_ENVS", "8"))))

    set_random_seed(seed)

    # Load baseline model path
    base_exp = get_experiment(base_exp_name)
    base_run_dir = get_run_dir(base_exp)
    base_model_path = _resolve_model_path(base_run_dir)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "finetune" / exp_name / ts
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)

    # Env
    env = SubprocVecEnv(
        [make_env(env_id, i, seed, q_path=q_path, beta=beta, B=B, q_device=q_device) for i in range(n_envs)],
        start_method="spawn",
    )
    env = VecMonitor(env)

    # Load model + continue learning
    model_kwargs = {"device": exp.get("device", "cuda")}

    # PPO hyperparams for fine-tune
    for k in ["learning_rate", "n_steps", "batch_size", "gamma", "clip_range"]:
        if k in exp:
            model_kwargs[k] = exp[k]

    model = PPO.load(str(base_model_path), env=env, **model_kwargs, verbose=1)

    cb = QPenaltyStatsCallback(print_every_rollout=True, verbose=0)
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False, callback=cb)

    #model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)

    model.save(run_dir / "model" / "model.zip")
    env.close()

    meta = {
        "finetune_exp": exp_name,
        "base_exp": base_exp_name,
        "base_run_dir": str(base_run_dir),
        "base_model_path": str(base_model_path),
        "env_id": env_id,
        "timesteps": timesteps,
        "seed": seed,
        "n_envs": n_envs,
        "q_path": str(q_path),
        "beta": beta,
        "B": B,
        "q_device": q_device,
    }
    (run_dir / "meta.json").write_text(__import__("json").dumps(meta, indent=2))

    print("Saved fine-tuned model to:", run_dir / "model" / "model.zip")
    return run_dir


if __name__ == "__main__":
    finetune_from_registry(sys.argv[1])
