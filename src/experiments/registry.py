import re
from pathlib import Path
from .definitions import EXPERIMENTS
from .sweep_definitions import SWEEPS
from .finetune_definitions import FINETUNES
from src.wrappers.reward import REWARD_WRAPPERS
from src.utils.seed_pools import load_pool


for _name, _exp in {**EXPERIMENTS, **SWEEPS, **FINETUNES}.items():
    if _exp.get("reward_wrapper") not in REWARD_WRAPPERS or "reward_kwargs" not in _exp:
        raise ValueError(f"{_name}: definition needs 'reward_wrapper' (one of {list(REWARD_WRAPPERS)}) and 'reward_kwargs'")


def get_experiment(name: str):
    if name in EXPERIMENTS:
        return EXPERIMENTS[name]
    if name in SWEEPS:
        return SWEEPS[name]
    if name in FINETUNES:
        return FINETUNES[name]
    # Sweep runs <base>_v3_sweep_b<beta>_s<i>: the base's v3 entry with that beta and sweep seed i
    m = re.fullmatch(r"(.+)_v3_sweep_b(.+)_s(\d+)", name)
    if m and f"{m[1]}_v3_b0.0" in FINETUNES:
        return {**FINETUNES[f"{m[1]}_v3_b0.0"], "name": name, "beta": float(m[2]), "seed": load_pool("sweep")[int(m[3])]}
    
    raise ValueError(f"Unknown experiment or sweep: {name}")


def get_run_dir(exp):
    if "base_exp" in exp:
        exp_name = exp.get("name")
        if exp_name is None:
            raise KeyError("Finetune exp dict must include exp['name'] (e.g. 'ppo_5m_goal2_safe_qft_v1').")

        root = Path("runs") / "finetune" / exp_name
        if not root.exists():
            raise FileNotFoundError(f"No finetune folder found at: {root}")

        # Most recent
        candidates = [p for p in root.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No timestamp runs found under: {root}")

        latest = sorted(candidates)[-1]
        return latest
    
    algo = exp["algorithm"]
    env = exp["env_id"]
    steps = exp["timesteps"]

    return (Path("experiments")/ algo/ env/ f"n_timesteps_{steps}")
