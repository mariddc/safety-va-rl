import json
from pathlib import Path

from src.experiments.finetune_hparams import FINETUNE_HPARAMS

SEED_POOLS_PATH = Path(__file__).resolve().parents[2] / "configs" / "seed_pools.json"
TRAINING_POOLS = ("pilot", "sweep", "confirm", "final")
EPISODE_POOLS = ("selection", "test", "diagnostic")


def _load_checked() -> dict:
    data = json.loads(SEED_POOLS_PATH.read_text())
    unknown = set(data) - set(TRAINING_POOLS) - set(EPISODE_POOLS) - {"sampler_seed"}
    if unknown:
        raise ValueError(f"unknown seed pools {sorted(unknown)}: add them to TRAINING_POOLS or EPISODE_POOLS")

    # A training seed s seeds its n_envs envs with s..s+n_envs-1; episode seeds are used as is.
    n_envs = max(h["n_envs"] for h in FINETUNE_HPARAMS.values())
    owner = {}
    for name, seeds in data.items():
        if name == "sampler_seed":
            continue
        width = n_envs if name in TRAINING_POOLS else 1
        for s in seeds:
            # Seeds below 1M are reserved: the base policies trained on env seeds 0..7.
            if s < 1_000_000:
                raise ValueError(f"seed pool {name} has a seed below 1000000")
            for env_seed in range(s, s + width):
                if env_seed in owner:
                    raise ValueError(f"seed pools {owner[env_seed]} and {name} overlap")
                owner[env_seed] = name
    return data


def load_pool(name: str, *, allow_test: bool = False) -> list[int]:
    if name == "test" and not allow_test:
        raise ValueError("the test pool is only loaded by the final test script (allow_test=True)")
    return _load_checked()[name]


def load_sampler_seed() -> int:
    return _load_checked()["sampler_seed"]
