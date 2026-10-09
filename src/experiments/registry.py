from pathlib import Path
from .definitions import EXPERIMENTS
from .sweep_definitions import SWEEPS
from .finetune_definitions import FINETUNES


def get_experiment(name: str):
    if name in EXPERIMENTS:
        return EXPERIMENTS[name]
    if name in SWEEPS:
        return SWEEPS[name]
    if name in FINETUNES:
        return FINETUNES[name]
    
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
