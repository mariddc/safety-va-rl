from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.frozen import FROZEN_FILE, check_frozen_q


def _write(path: Path, q: dict) -> Path:
    path.write_text(json.dumps({"q": q, "base_models": {}}, indent=2))
    return path


def test_absent_file_is_record_only(tmp_path: Path) -> None:
    assert check_frozen_q("ppo_5m_goal2_safe", "aa", path=tmp_path / "frozen.json") == "no_frozen_file"


def test_matching_hash(tmp_path: Path) -> None:
    f = _write(tmp_path / "frozen.json", {"ppo_5m_goal2_safe": {"path": "logs/q_theta.pt", "sha256": "aa"}})
    assert check_frozen_q("ppo_5m_goal2_safe", "aa", path=f) == "matched"


def test_mismatched_hash_raises_naming_file_and_base(tmp_path: Path) -> None:
    f = _write(tmp_path / "frozen.json", {"ppo_5m_goal2_safe": {"path": "logs/q_theta.pt", "sha256": "aa"}})
    with pytest.raises(ValueError) as e:
        check_frozen_q("ppo_5m_goal2_safe", "bb", path=f)
    assert "frozen.json" in str(e.value) and "ppo_5m_goal2_safe" in str(e.value)


def test_missing_base_entry_raises(tmp_path: Path) -> None:
    f = _write(tmp_path / "frozen.json", {"ppo_5m_goal2_safe": {"path": "logs/q_theta.pt", "sha256": "aa"}})
    with pytest.raises(ValueError, match="ppo_5m700k_goal2"):
        check_frozen_q("ppo_5m700k_goal2", "aa", path=f)


def test_malformed_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "frozen.json"
    f.write_text(json.dumps({"base_models": {}}))
    with pytest.raises(ValueError, match="'q'"):
        check_frozen_q("ppo_5m_goal2_safe", "aa", path=f)


def test_default_location_is_committed_configs_dir() -> None:
    assert FROZEN_FILE.parts[-2:] == ("configs", "frozen.json")
    assert FROZEN_FILE.parent.parent == Path(__file__).resolve().parents[1]
