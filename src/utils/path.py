from __future__ import annotations
from pathlib import Path
from typing import Mapping


def get_intervention_root(exp: Mapping, project_root: Path, exp_name: str) -> Path:
    """
    logs/interventions/{algo}/{env_id}/{exp_name}/n_timesteps_{timesteps}
    """
    algo = exp["algorithm"]
    env_id = exp["env_id"]
    timesteps = exp["timesteps"]

    return (
        project_root
        / "logs"
        / "interventions"
        / str(algo)
        / str(env_id)
        / str(exp_name)
        / f"n_timesteps_{int(timesteps)}"
    )


def get_eil_dataset_root(exp: Mapping, project_root: Path, exp_name: str, dataset_id: str) -> Path:
    """
    logs/datasets/eil/{algo}/{env_id}/{exp_name}/{dataset_id}
    """
    algo = exp["algorithm"]
    env_id = exp["env_id"]

    return (
        project_root
        / "logs"
        / "datasets"
        / "eil"
        / str(algo)
        / str(env_id)
        / str(exp_name)
        / str(dataset_id)
    )