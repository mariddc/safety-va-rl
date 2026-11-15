"""
Frozen PPO hyperparameters for fine-tunes, one dict per base (D022), shared by policy_ft and policy_ref.

A fine-tune definition is `{name, backend, ..., **FINETUNE_HPARAMS[base], "beta": b, "seed": s}`, so the
two arms of a pair can differ only in beta (and the per-run seed). `resolve_finetune_hparams` refuses a
definition that is missing a key, carries an unknown key, or changes a frozen value; `check_effective_hparams`
refuses a model whose values read back after `PPO.load` differ from the requested ones (D023, D067).
"""
from __future__ import annotations

from typing import Any, Mapping

# FROZEN (FROZEN.md row 0.6, D026). Values from D017-D021 and D008; do not edit without a new decision in
# DECISIONS.md. "timesteps" is a placeholder pending the timing trial 1.5 (D021).
# The dicts are per base (D022) even though the values are identical today.
FINETUNE_HPARAMS: dict[str, dict[str, Any]] = {
    "ppo_5m_goal2_safe": {
        "learning_rate": 1e-4,
        "n_steps": 2048,        
        "batch_size": 64,       
        "n_epochs": 10,         
        "gamma": 0.99,
        "gae_lambda": 0.95,     
        "clip_range": 0.2,
        "ent_coef": 0.0,        
        "vf_coef": 0.5,         
        "max_grad_norm": 0.5,   
        "n_envs": 8,            
        "timesteps": 500_000,
        "device": "cpu",        
    },
    "ppo_5m700k_goal2": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "n_envs": 8,
        "timesteps": 500_000,
        "device": "cpu",
    },
}

HPARAM_KEYS = (
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
    "ent_coef", "vf_coef", "max_grad_norm", "n_envs", "timesteps", "device",
)

# Used by the script for the envs and for learn(), not passed to PPO.load.
_SCRIPT_KEYS = ("n_envs", "timesteps")

# Every other key a fine-tune definition may carry (CONVENTIONS "Experiment and config definitions").
# The registry only checks reward_wrapper / reward_kwargs and injects nothing.
_RUN_KEYS = (
    "name", "backend", "algorithm", "env_id", "base_exp", "seed",
    "q_path", "beta", "q_device", "reward_wrapper", "reward_kwargs",
)

ALLOWED_DEFINITION_KEYS = frozenset(_RUN_KEYS) | frozenset(HPARAM_KEYS)


def _same_value(a: Any, b: Any) -> bool:
    # Same type as well as equal: 64.0 for batch_size (or True for 1) is not the frozen value, and a float
    # n_steps/batch_size would break SB3's buffer indexing. 1e-4 and 0.0001 are the same float.
    return type(a) is type(b) and a == b


def resolve_finetune_hparams(exp_name: str, exp: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate a fine-tune definition against FINETUNE_HPARAMS[exp["base_exp"]] and return the kwargs for
    `PPO.load` (all frozen keys except n_envs and timesteps). Pure: run it before anything is written.
    """
    base = exp["base_exp"]
    if base not in FINETUNE_HPARAMS:
        raise KeyError(
            f"{exp_name}: base {base!r} has no frozen fine-tune hyperparameters; "
            f"FINETUNE_HPARAMS in src/experiments/finetune_hparams.py has {sorted(FINETUNE_HPARAMS)}."
        )
    frozen = FINETUNE_HPARAMS[base]

    missing = [k for k in HPARAM_KEYS if k not in exp]
    if missing:
        raise KeyError(
            f"{exp_name}: definition lacks fine-tune hyperparameters {missing}. "
            f"Spread **FINETUNE_HPARAMS[{base!r}] into the definition."
        )

    unknown = sorted(set(exp) - ALLOWED_DEFINITION_KEYS)
    if unknown:
        raise ValueError(
            f"{exp_name}: unknown definition keys {unknown}; allowed keys are {sorted(ALLOWED_DEFINITION_KEYS)}. "
            "Fix the spelling or remove them."
        )

    changed = {k: (exp[k], frozen[k]) for k in HPARAM_KEYS if not _same_value(exp[k], frozen[k])}
    if changed:
        shown = ", ".join(f"{k}={v!r} (frozen {f!r})" for k, (v, f) in changed.items())
        raise ValueError(
            f"{exp_name}: hyperparameters differ from FINETUNE_HPARAMS[{base!r}]: {shown}. "
            f"Use **FINETUNE_HPARAMS[{base!r}] unchanged; the values are frozen (FROZEN.md 0.6)."
        )

    # Per-run, not shared, but it must be set: PPO.load would otherwise re-seed with the base zip's seed (D070)
    if "seed" not in exp:
        raise KeyError(f"{exp_name}: definition lacks 'seed'; set the per-run training seed (an int).")
    if type(exp["seed"]) is not int:
        raise ValueError(f"{exp_name}: seed {exp['seed']!r} is not an int; set 'seed' to an integer.")

    rollout = exp["n_steps"] * exp["n_envs"]
    if rollout % exp["batch_size"] != 0:
        # SB3 only warns about this, and only when the model is built, not on PPO.load.
        raise ValueError(
            f"{exp_name}: batch_size {exp['batch_size']} does not divide n_steps * n_envs = {rollout}."
        )

    return {k: exp[k] for k in HPARAM_KEYS if k not in _SCRIPT_KEYS}


def check_effective_hparams(exp_name: str, requested: Mapping[str, Any], effective: Mapping[str, Any]) -> None:
    """RuntimeError unless the values read back from the loaded model equal the requested ones, key for key."""
    keys = sorted(set(requested) | set(effective))
    mismatched = {k: (requested.get(k), effective.get(k)) for k in keys if requested.get(k) != effective.get(k)}
    if mismatched:
        shown = ", ".join(f"{k}: requested {r!r}, model has {e!r}" for k, (r, e) in mismatched.items())
        raise RuntimeError(
            f"{exp_name}: loaded model does not use the requested hyperparameters ({shown}). "
            "PPO.load did not apply an override; do not train with this model."
        )
