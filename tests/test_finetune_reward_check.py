from __future__ import annotations

import pytest

from scripts.finetune_policy import _check_reward_matches_base
from src.experiments.registry import get_experiment


def _exp(reward_wrapper: str, reward_kwargs: dict) -> dict:
    return {"reward_wrapper": reward_wrapper, "reward_kwargs": reward_kwargs}


def test_matching_reward_passes() -> None:
    ft = _exp("fastsafe", {"goal_bonus": 50.0, "shaping_k": 1.0})
    base = _exp("fastsafe", {"goal_bonus": 50.0, "shaping_k": 1.0})
    _check_reward_matches_base("ft_exp", ft, "base_exp", base)


def test_different_wrapper_raises_with_names() -> None:
    ft = _exp("fastsafe", {"goal_bonus": 50.0, "shaping_k": 1.0})
    base = _exp("complete", {"goal_bonus": 50.0, "shaping_k": 1.0})
    with pytest.raises(ValueError) as e:
        _check_reward_matches_base("ft_exp", ft, "base_exp", base)
    msg = str(e.value)
    assert "ft_exp" in msg and "base_exp" in msg
    assert "fastsafe" in msg and "complete" in msg


def test_different_kwargs_raises() -> None:
    # Same wrapper, different shaping: still a different reward.
    ft = _exp("fastsafe", {"goal_bonus": 50.0, "shaping_k": 1.0})
    base = _exp("fastsafe", {"goal_bonus": 50.0, "shaping_k": 1.5})
    with pytest.raises(ValueError, match="ft_exp"):
        _check_reward_matches_base("ft_exp", ft, "base_exp", base)


def test_registry_pair_matching_passes() -> None:
    name = "ppo_5m_goal2_safe_qind_b0.0"
    exp = get_experiment(name)
    _check_reward_matches_base(name, exp, exp["base_exp"], get_experiment(exp["base_exp"]))


def test_registry_pair_mismatched_raises() -> None:
    # Old definition: fastsafe fine-tune on a base trained with the complete reward.
    name = "ppo_5m700k_goal2_inverted"
    exp = get_experiment(name)
    with pytest.raises(ValueError, match="ppo_5m700k_goal2_inverted.*ppo_5m700k_goal2"):
        _check_reward_matches_base(name, exp, exp["base_exp"], get_experiment(exp["base_exp"]))
