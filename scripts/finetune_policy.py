from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import safety_gymnasium
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.registry import get_experiment, get_run_dir
from src.experiments.finetune_hparams import FINETUNE_HPARAMS
from src.eil.q_penalty_callback import QPenaltyStatsCallback
from src.wrappers.reward import wrap_reward, reward_config
from src.wrappers.InterventionRewardWrapper import InterventionRewardWrapper
from src.utils.run_meta import git_info, recorded_run, repo_relative, sha256_file


def _resolve_model_path(run_dir: Path) -> Path:
    p1 = run_dir / "model" / "model.zip"
    if p1.exists():
        return p1
    p2 = run_dir / "model" / "model"
    if p2.exists():
        return p2 
    raise FileNotFoundError(f"Model not found in {run_dir / 'model'}")


def _check_reward_matches_base(exp_name: str, exp: dict, base_exp_name: str, base_exp: dict) -> None:
    # A different reward would make the fine-tune result not comparable to its base.
    ft_reward = reward_config(exp)
    base_reward = reward_config(base_exp)
    if ft_reward != base_reward:
        raise ValueError(
            f"{exp_name}: reward {ft_reward} differs from base {base_exp_name} reward {base_reward}. "
            f"Set 'reward_wrapper' and 'reward_kwargs' in {exp_name} equal to those of {base_exp_name}."
        )


def make_env(exp: dict, rank: int, seed: int, *, q_path: Path, beta: float, q_device: str):
    def _init():
        env = wrap_reward(safety_gymnasium.make(exp["env_id"]), exp)
        env = InterventionRewardWrapper(env, q_path=str(q_path), beta=beta, device=q_device)
        env.reset(seed=seed + rank)
        return env
    return _init


def finetune_from_registry(exp_name: str, *, run_pass: str, allow_dirty: bool = False) -> Path:
    exp = get_experiment(exp_name)

    base_exp_name = exp["base_exp"]
    q_path = (ROOT / exp["q_path"]).resolve()
    beta = float(exp["beta"])
    q_device = str(exp["q_device"])

    # Load baseline model path
    base_exp = get_experiment(base_exp_name)
    _check_reward_matches_base(exp_name, exp, base_exp_name, base_exp)

    if exp["env_id"] != base_exp["env_id"]:
        raise ValueError(
            f"{exp_name}: env_id {exp['env_id']!r} differs from base {base_exp_name} env_id {base_exp['env_id']!r}. "
            f"Set 'env_id' in {exp_name} equal to that of {base_exp_name}."
        )

    # ft and ref share these values, so a pair differs only in beta (and the per-run seed)
    frozen = FINETUNE_HPARAMS[base_exp_name]
    changed = [k for k in frozen if exp[k] != frozen[k]]
    if changed:
        raise ValueError(f"{exp_name}: {changed} differ from FINETUNE_HPARAMS[{base_exp_name!r}]")
    model_kwargs = {k: v for k, v in frozen.items() if k not in ("n_envs", "timesteps")}

    env_id = exp["env_id"]
    timesteps = int(exp["timesteps"])
    seed = exp["seed"]
    n_envs = int(exp["n_envs"])

    set_random_seed(seed)

    base_run_dir = get_run_dir(base_exp)
    base_model_path = _resolve_model_path(base_run_dir)

    # Checks before anything is written: a refused run leaves nothing on disk.
    git = git_info()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError(f"{exp_name}: uncommitted code changes {git['dirty_paths']}; commit them or pass --allow-dirty")
    git["dirty_allowed"] = allow_dirty

    q_sha256 = sha256_file(q_path)
    # configs/frozen.json is written when Q is frozen; until then the hash is only recorded.
    frozen_file = ROOT / "configs" / "frozen.json"
    q_hash_check = "no_frozen_file"
    if frozen_file.exists():
        frozen_sha256 = json.loads(frozen_file.read_text())["q"][base_exp_name]["sha256"]
        if q_sha256 != frozen_sha256:
            raise ValueError(f"{exp_name}: Q {q_path.name} sha256 {q_sha256} differs from frozen {frozen_sha256}")
        q_hash_check = "matched"
    q_ckpt = torch.load(q_path, map_location="cpu")

    job = os.environ.get("SLURM_JOB_ID") or os.getpid()
    run_dir = ROOT / "runs" / "finetune" / exp_name / f"{datetime.now():%Y%m%d_%H%M%S}_{run_pass}_{job}"
    run_dir.mkdir(parents=True, exist_ok=False)

    requested = {**model_kwargs, "n_envs": n_envs, "seed": seed}
    with recorded_run(
        run_dir / "meta.json", exp_name, run_pass, git,
        seed=seed, beta=beta, timesteps=timesteps, n_envs=n_envs, env_id=env_id, **reward_config(exp),
        hyperparameters={"requested": requested, "effective": None},
        q_path=repo_relative(q_path), q_sha256=q_sha256, q_hash_check=q_hash_check,
        B=q_ckpt.get("B"), lambda_I=q_ckpt.get("lambda_I"), dataset_id=q_path.parent.name,
        base_exp=base_exp_name, base_model_path=repo_relative(base_model_path),
        base_model_sha256=sha256_file(base_model_path),
    ) as meta:
        (run_dir / "logs").mkdir()
        (run_dir / "model").mkdir()

        # Env
        env = SubprocVecEnv(
            [make_env(exp, i, seed, q_path=q_path, beta=beta, q_device=q_device) for i in range(n_envs)],
            start_method="spawn",
        )
        env = VecMonitor(env)

        # Load model + continue learning
        # seed: otherwise PPO.load re-seeds with the seed saved in the base zip (0)
        # tensorboard_log: otherwise the curves go to the base run's log dir
        model = PPO.load(str(base_model_path), env=env, **model_kwargs, seed=seed,
                         tensorboard_log=str(run_dir / "logs"), verbose=1)

        # Read back after load, so an override that did not take stops the run before training
        effective = {k: getattr(model, k) for k in requested}
        effective.update(learning_rate=model.lr_schedule(1.0), clip_range=model.clip_range(1.0), device=str(model.device))
        meta["hyperparameters"]["effective"] = effective
        if effective != requested:
            raise RuntimeError(f"{exp_name}: loaded model hyperparameters {effective} differ from requested {requested}")

        cb = QPenaltyStatsCallback(print_every_rollout=True, verbose=0)
        meta["timesteps_at_start"] = model.num_timesteps
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False, callback=cb)
        meta["timesteps_trained"] = model.num_timesteps - meta["timesteps_at_start"]

        #model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)

        model.save(run_dir / "model" / "model.zip")
        env.close()

    print("Saved fine-tuned model to:", run_dir / "model" / "model.zip")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a base PPO policy with the Q-intervention penalty.")
    parser.add_argument("exp_name", help="Fine-tune experiment name in the registry")
    parser.add_argument("--run-pass", required=True,
                        help="Round tag stored in the run dir name and meta.json")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run with uncommitted code changes (recorded as dirty_allowed=true; not for reported runs)")
    args = parser.parse_args()
    finetune_from_registry(args.exp_name, run_pass=args.run_pass, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
