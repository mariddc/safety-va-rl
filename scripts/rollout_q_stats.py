from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import safety_gymnasium
from stable_baselines3 import PPO, SAC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.registry import get_experiment, get_run_dir
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


def _load_model(algo: str, model_path: Path):
    algo = algo.lower()
    if algo == "ppo":
        return PPO.load(str(model_path), device="cpu")
    if algo == "sac":
        return SAC.load(str(model_path), device="cpu")
    raise ValueError(f"Unsupported algorithm: {algo}")


def _stats(x: np.ndarray) -> str:
    if x.size == 0:
        return "n=0"
    return (
        f"n={x.size} mean={x.mean():+.6f} "
        f"p10={np.percentile(x,10):+.6f} p50={np.percentile(x,50):+.6f} p90={np.percentile(x,90):+.6f} "
        f"min={x.min():+.6f} max={x.max():+.6f}"
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/rollout_q_stats.py <exp_name> [n_steps]")
        sys.exit(1)

    exp_name = sys.argv[1]
    n_steps = int(sys.argv[2]) if len(sys.argv) >= 3 else 20_000

    exp = get_experiment(exp_name)
    try:
        run_dir = get_run_dir(exp)
    except FileNotFoundError:
        # Finetune not run yet: roll out its base policy instead
        if "base_exp" not in exp:
            raise
        run_dir = get_run_dir(get_experiment(exp["base_exp"]))
        print(f"No finetune run for {exp_name}; using base model {exp['base_exp']}")
    model_path = _resolve_model_path(run_dir)

    # Pull q params from exp if present (works for FINETUNES too).
    # B is not set here: the wrapper reads the Q-training B from the checkpoint.
    q_path = exp.get("q_path", None)
    beta = float(exp.get("beta", 1.0))
    q_device = str(exp.get("q_device", "cpu"))

    env = safety_gymnasium.make(exp["env_id"])
    env = FastSafeRewardWrapper(env)

    if q_path is None:
        raise KeyError(
            "exp does not contain q_path. Add it to your FINETUNES entry (or pass a finetune exp name)."
        )

    qwrap = InterventionRewardWrapper(env, q_path=str(q_path), beta=beta, device=q_device)
    env = qwrap

    model = _load_model(exp["algorithm"], model_path)

    obs, _ = env.reset()

    q_vals = []
    penalties = [] 
    costs = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if "q_value" in info:
            q_vals.append(float(info["q_value"]))
        if "q_penalty" in info:
            penalties.append(float(info["q_penalty"]))
        if "cost" in info:
            costs.append(float(info["cost"]))

        if terminated or truncated:
            obs, _ = env.reset()

    q_vals = np.asarray(q_vals, dtype=np.float32)
    penalties = np.asarray(penalties, dtype=np.float32)
    costs = np.asarray(costs, dtype=np.float32)

    print(f"EXP={exp_name}  model={model_path}")
    print(f"B={qwrap.B:+.6f} (from checkpoint)  beta={qwrap.beta}")
    print(f"Q:       {_stats(q_vals)}")
    print(f"Penalty: {_stats(penalties)}")
    if costs.size:
        print(f"Cost:    {_stats(costs)}")
    print(f"Penalty values = {np.unique(penalties).tolist() if penalties.size else []}")
    print(f"Frac penalty>0 = {float((penalties>0).mean()) if penalties.size else 0.0:.3f}")
    # Expected active fraction under this policy; mean penalty = beta * this
    print(f"Frac Q>B       = {float((q_vals>qwrap.B).mean()) if q_vals.size else 0.0:.3f}")


if __name__ == "__main__":
    main()
