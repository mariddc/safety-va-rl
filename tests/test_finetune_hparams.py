from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

import scripts.finetune_policy as finetune_policy
from src.experiments.finetune_hparams import (
    ALLOWED_DEFINITION_KEYS, FINETUNE_HPARAMS, HPARAM_KEYS, check_effective_hparams, resolve_finetune_hparams,
)
from src.utils.run_meta import sb3_effective_hyperparameters

# Decided values (D017-D021, D008). Pinned here so an edit to a frozen value fails a test (FROZEN.md 0.6).
DECIDED = {
    "learning_rate": 1e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gamma": 0.99,
    "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "n_envs": 8, "timesteps": 500_000, "device": "cpu",
}
BASES = ("ppo_5m_goal2_safe", "ppo_5m700k_goal2")


def _definition(base: str = "ppo_5m_goal2_safe", *, beta: float = 0.0, seed: int = 0, **overrides) -> dict:
    exp = {
        "name": f"ft_b{beta}", "backend": "sb3", "algorithm": "ppo", "env_id": "SafetyPointGoal2-v0",
        "base_exp": base, "q_path": "logs/q_theta.pt", "q_device": "cpu",
        "reward_wrapper": "fastsafe", "reward_kwargs": {"goal_bonus": 50.0, "shaping_k": 1.0},
        **FINETUNE_HPARAMS[base], "beta": beta, "seed": seed,
    }
    exp.update(overrides)
    return exp


# ---------------------------------------------------------------- frozen values

@pytest.mark.parametrize("base", BASES)
def test_frozen_values_are_the_decided_ones(base: str) -> None:
    assert set(FINETUNE_HPARAMS) == set(BASES)
    hp = FINETUNE_HPARAMS[base]
    assert hp == DECIDED
    # Types too: an int where a float was decided (or the reverse) is a different definition.
    assert {k: type(v) for k, v in hp.items()} == {k: type(v) for k, v in DECIDED.items()}
    assert hp["device"] == "cpu"


def test_bases_have_identical_key_sets() -> None:
    a, b = (set(FINETUNE_HPARAMS[x]) for x in BASES)
    assert a == b == set(HPARAM_KEYS)


def test_seed_and_beta_are_not_shared_hparams() -> None:
    for base in BASES:
        assert "beta" not in FINETUNE_HPARAMS[base] and "seed" not in FINETUNE_HPARAMS[base]


# ---------------------------------------------------------------- pairing: ft and ref differ only in beta

@pytest.mark.parametrize("base", BASES)
def test_ft_and_ref_resolve_to_identical_ppo_kwargs(base: str) -> None:
    ref = resolve_finetune_hparams("ref", _definition(base, beta=0.0, seed=3))
    ft = resolve_finetune_hparams("ft", _definition(base, beta=1.0, seed=3))
    assert ft == ref
    assert set(ref) == set(HPARAM_KEYS) - {"n_envs", "timesteps"}
    assert ref["device"] == "cpu" and ref["learning_rate"] == 1e-4 and ref["batch_size"] == 64


def test_resolved_kwargs_have_no_run_fields() -> None:
    kw = resolve_finetune_hparams("x", _definition(beta=1.0))
    assert not {"beta", "seed", "n_envs", "timesteps", "q_path"} & set(kw)


# ---------------------------------------------------------------- loud failures

def test_missing_key_raises_keyerror_naming_exp_and_key() -> None:
    exp = _definition()
    del exp["n_epochs"]
    with pytest.raises(KeyError, match=r"my_ft.*n_epochs.*FINETUNE_HPARAMS"):
        resolve_finetune_hparams("my_ft", exp)


def test_typo_key_raises_valueerror() -> None:
    exp = _definition(learnig_rate=1e-4)
    with pytest.raises(ValueError, match=r"my_ft.*learnig_rate"):
        resolve_finetune_hparams("my_ft", exp)


@pytest.mark.parametrize("key, value", [
    ("learning_rate", 3e-4), ("batch_size", 2048), ("timesteps", 200_000), ("n_envs", 4),
    ("device", "cuda"), ("clip_range", 0.1), ("gamma", 0.995),
])
def test_changed_value_raises_valueerror(key: str, value) -> None:
    with pytest.raises(ValueError, match=rf"my_ft.*{key}"):
        resolve_finetune_hparams("my_ft", _definition(**{key: value}))


def test_same_value_other_type_raises() -> None:
    # Documented choice (D069): 64.0 is not the frozen 64.
    with pytest.raises(ValueError, match="batch_size"):
        resolve_finetune_hparams("my_ft", _definition(batch_size=64.0))
    resolve_finetune_hparams("my_ft", _definition(learning_rate=0.0001))  # same float as 1e-4


def test_missing_seed_raises_keyerror() -> None:
    exp = _definition()
    del exp["seed"]
    with pytest.raises(KeyError, match=r"my_ft.*seed"):
        resolve_finetune_hparams("my_ft", exp)


@pytest.mark.parametrize("seed", [3.0, True, "3", None])
def test_non_int_seed_raises_valueerror(seed) -> None:
    with pytest.raises(ValueError, match=r"my_ft.*seed"):
        resolve_finetune_hparams("my_ft", _definition(seed=seed))


def test_unknown_base_raises_keyerror() -> None:
    exp = _definition()
    exp["base_exp"] = "ppo_6m_goal2"
    with pytest.raises(KeyError, match=r"my_ft.*ppo_6m_goal2"):
        resolve_finetune_hparams("my_ft", exp)


def test_batch_size_must_divide_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {**FINETUNE_HPARAMS["ppo_5m_goal2_safe"], "batch_size": 100}
    monkeypatch.setitem(FINETUNE_HPARAMS, "ppo_5m_goal2_safe", bad)
    with pytest.raises(ValueError, match="does not divide"):
        resolve_finetune_hparams("my_ft", _definition())


def test_allowed_keys_cover_the_conventions_example() -> None:
    # CONVENTIONS "Fine-tune entries": shared required keys plus the fine-tune ones.
    required = {"backend", "algorithm", "env_id", "timesteps", "seed", "n_envs", "reward_wrapper",
                "reward_kwargs", "name", "base_exp", "q_path", "beta", "q_device"}
    assert required <= ALLOWED_DEFINITION_KEYS


def test_old_definitions_no_longer_resolve() -> None:
    # They predate 0.6 (finished runs, left unchanged); none may run with inherited hyperparameters.
    from src.experiments.finetune_definitions import FINETUNES
    for name, exp in FINETUNES.items():
        with pytest.raises((KeyError, ValueError)):
            resolve_finetune_hparams(name, exp)


def _run_script_until_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exp: dict, base_env_id: str):
    base = {"env_id": base_env_id, "timesteps": 1, "seed": 0,
            "reward_wrapper": "fastsafe", "reward_kwargs": {"goal_bonus": 50.0, "shaping_k": 1.0}}
    exps = {"ft": exp, "ppo_5m_goal2_safe": base}
    repo = tmp_path / "repo"
    repo.mkdir()

    def _no_git():
        raise AssertionError("git checked before the config checks")

    monkeypatch.setattr(finetune_policy, "ROOT", repo)
    monkeypatch.setattr(finetune_policy, "get_experiment", lambda name: exps[name])
    monkeypatch.setattr(finetune_policy, "get_run_dir", lambda e: (_ for _ in ()).throw(AssertionError("run dir")))
    monkeypatch.setattr(finetune_policy, "git_provenance", _no_git)
    try:
        finetune_policy.finetune_from_registry("ft", run_pass="pass1")
    finally:
        assert list(repo.iterdir()) == []


def test_script_validates_before_git_or_any_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = _definition()
    del exp["gae_lambda"]
    with pytest.raises(KeyError, match="gae_lambda"):
        _run_script_until_config_error(tmp_path, monkeypatch, exp, "SafetyPointGoal2-v0")


def test_script_rejects_env_id_different_from_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match=r"ft: env_id 'SafetyPointGoal2-v0' differs from base ppo_5m_goal2_safe"):
        _run_script_until_config_error(tmp_path, monkeypatch, _definition(), "SafetyPointGoal1-v0")


# ---------------------------------------------------------------- effective vs requested (D067)

def _requested() -> dict:
    return {**resolve_finetune_hparams("x", _definition()), "n_envs": 8, "seed": 0}


def test_effective_check_passes_on_equal() -> None:
    check_effective_hparams("x", _requested(), dict(_requested()))


@pytest.mark.parametrize("key, value", [("learning_rate", 3e-4), ("batch_size", 2048), ("device", "cuda"),
                                        ("n_envs", 1), ("seed", 3)])
def test_effective_check_raises_on_mismatch(key: str, value) -> None:
    eff = {**_requested(), key: value}
    with pytest.raises(RuntimeError, match=rf"x.*{key}"):
        check_effective_hparams("x", _requested(), eff)


def test_effective_check_raises_on_missing_key() -> None:
    eff = _requested()
    del eff["ent_coef"]
    with pytest.raises(RuntimeError, match="ent_coef"):
        check_effective_hparams("x", _requested(), eff)


def test_ppo_load_without_seed_keeps_zip_seed(tmp_path: Path) -> None:
    # The bug D070 fixes: without seed=..., the loaded model re-seeds the env with the zip's seed.
    PPO("MlpPolicy", DummyVecEnv([lambda: gym.make("CartPole-v1")]), n_steps=64, batch_size=32, seed=0,
        device="cpu").save(tmp_path / "base.zip")
    env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])
    model = PPO.load(str(tmp_path / "base.zip"), env=env, device="cpu")
    assert model.seed == 0 and env._seeds == [0, 1]


def test_ppo_load_applies_every_override(tmp_path: Path) -> None:
    # No training: build a model with other values, save, reload with the frozen kwargs, read back.
    def venv(n: int) -> DummyVecEnv:
        return DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(n)])

    # Saved with seed 0, as both real bases are.
    PPO("MlpPolicy", venv(2), learning_rate=3e-4, n_steps=64, batch_size=32, n_epochs=3, gamma=0.9,
        gae_lambda=0.9, clip_range=0.3, ent_coef=0.01, vf_coef=0.4, max_grad_norm=0.9, seed=0,
        device="cpu").save(tmp_path / "base.zip")

    kwargs = resolve_finetune_hparams("x", _definition(seed=3))
    env = venv(8)
    model = PPO.load(str(tmp_path / "base.zip"), env=env, **kwargs, seed=3, tensorboard_log=str(tmp_path / "logs"))
    requested = {**kwargs, "n_envs": 8, "seed": 3}
    check_effective_hparams("x", requested, sb3_effective_hyperparameters(model))
    # The per-run seed reached _setup_model: env resets in learn() use seed + idx, not the zip's 0 + idx (D070).
    assert model.seed == 3
    assert env._seeds == list(range(3, 11))
    assert model.tensorboard_log == str(tmp_path / "logs")
    assert model.rollout_buffer.buffer_size == 2048 and model.rollout_buffer.gae_lambda == 0.95
    # The optimizer state comes from the zip with the old lr; PPO.train sets it from lr_schedule
    # (_update_learning_rate) before the first gradient step.
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    model.set_logger(configure(None, []))  # learn() would configure it; only the call below needs one
    model._update_learning_rate(model.policy.optimizer)
    assert model.policy.optimizer.param_groups[0]["lr"] == 1e-4
