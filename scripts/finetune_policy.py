from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.registry import get_experiment, get_run_dir
from src.eil.q_penalty_callback import QPenaltyStatsCallback
from src.wrappers.reward import wrap_reward, reward_config
from src.wrappers.InterventionRewardWrapper import InterventionRewardWrapper
from src.utils.frozen import check_frozen_q
from src.utils.run_meta import (
    build_run_meta, finetune_run_dir_name, git_provenance, q_info, recorded_run, repo_relative,
    require_clean, sb3_effective_hyperparameters, sha256_file, validate_run_pass,
)


def _resolve_model_path(run_dir: Path) -> Path:
    p1 = run_dir / "model" / "model.zip"
    if p1.exists():
        return p1
    p2 = run_dir / "model" / "model"
    if p2.exists():
        return p2 
    raise FileNotFoundError(f"Model not found in {run_dir / 'model'}")


def _check_reward_matches_base(exp_name: str, exp: dict, base_exp_name: str, base_exp: dict) -> None:
    # A different reward would make the fine-tune result not comparable to its base (CLAUDE.md research rule).
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
    validate_run_pass(run_pass)
    exp = get_experiment(exp_name)

    base_exp_name = exp["base_exp"]
    q_path = (ROOT / exp["q_path"]).resolve()
    beta = float(exp["beta"])
    q_device = str(exp["q_device"])

    if "B" in exp:
        raise ValueError(
            f"{exp_name}: 'B' is no longer a fine-tune setting. "
            "The penalty threshold is the Q-training B stored in q_theta.pt; remove 'B' from the definition."
        )

    env_id = exp["env_id"]
    timesteps = int(exp["timesteps"])
    seed = int(exp.get("seed", 0))
    n_envs = int(exp.get("n_envs", int(os.environ.get("N_ENVS", "8"))))

    set_random_seed(seed)

    # Load baseline model path
    base_exp = get_experiment(base_exp_name)
    _check_reward_matches_base(exp_name, exp, base_exp_name, base_exp)
    base_run_dir = get_run_dir(base_exp)
    base_model_path = _resolve_model_path(base_run_dir)

    # Provenance checks before anything is written: a refused run leaves nothing on disk.
    git = require_clean(git_provenance(), what=exp_name, allow_dirty=allow_dirty)
    q = q_info(q_path)
    q_hash_check = check_frozen_q(base_exp_name, q["q_sha256"])

    model_kwargs = {"device": exp.get("device", "cuda")}

    # PPO hyperparams for fine-tune
    for k in ["learning_rate", "n_steps", "batch_size", "gamma", "clip_range"]:
        if k in exp:
            model_kwargs[k] = exp[k]

    meta = build_run_meta(
        run_kind="finetune",
        run_pass=run_pass,
        exp_name=exp_name,
        git=git,
        device=model_kwargs["device"],
        seed=seed,
        beta=beta,
        timesteps=timesteps,
        n_envs=n_envs,
        env_id=env_id,
        **reward_config(exp),
        hyperparameters={"requested": dict(model_kwargs), "effective": None},
        **q,
        q_hash_check=q_hash_check,
        base_exp=base_exp_name,
        base_model_path=repo_relative(base_model_path),
        base_model_sha256=sha256_file(base_model_path),
        # Keys of the pre-schema fine-tune meta, kept for older readers
        extra={
            "finetune_exp": exp_name,
            "base_run_dir": repo_relative(base_run_dir),
            "q_penalty": "indicator",
            "q_device": q_device,
        },
    )

    # exist_ok=False: two runs must never share a directory (D032)
    run_dir = ROOT / "runs" / "finetune" / exp_name / finetune_run_dir_name(run_pass)
    run_dir.mkdir(parents=True, exist_ok=False)

    with recorded_run(run_dir / "meta.json", meta):
        (run_dir / "logs").mkdir()
        (run_dir / "model").mkdir()

        # Env
        env = SubprocVecEnv(
            [make_env(exp, i, seed, q_path=q_path, beta=beta, q_device=q_device) for i in range(n_envs)],
            start_method="spawn",
        )
        env = VecMonitor(env)

        # Load model + continue learning
        model = PPO.load(str(base_model_path), env=env, **model_kwargs, verbose=1)

        # Read back after load: PPO.load rebuilds schedules from the overrides (D024)
        meta["hyperparameters"]["effective"] = sb3_effective_hyperparameters(model)
        meta["hardware"]["device"] = str(model.device)

        cb = QPenaltyStatsCallback(print_every_rollout=True, verbose=0)
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False, callback=cb)

        #model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=False)

        model.save(run_dir / "model" / "model.zip")
        env.close()

    print("Saved fine-tuned model to:", run_dir / "model" / "model.zip")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a base PPO policy with the Q-intervention penalty.")
    parser.add_argument("exp_name", help="Fine-tune experiment name in the registry")
    parser.add_argument("--run-pass", required=True,
                        help="Implementation round tag stored in the run dir name and meta.json, e.g. pass1 (D064)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run with uncommitted code changes (recorded as dirty_allowed=true; not for reported runs)")
    args = parser.parse_args()
    finetune_from_registry(args.exp_name, run_pass=args.run_pass, allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
