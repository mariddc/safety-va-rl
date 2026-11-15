from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import torch

import scripts.finetune_policy as finetune_policy
from src.utils import run_meta
from src.utils.run_meta import (
    build_run_meta, dirty_paths_from_status, finetune_run_dir_name, git_provenance, q_info,
    recorded_run, require_clean, sha256_file, write_meta,
)


def _git(dirty: bool = False) -> dict:
    return {"commit": "abc123", "dirty": dirty, "dirty_allowed": None,
            "dirty_paths": ["src/x.py"] if dirty else [], "diff_sha256": None, "source": "git"}


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


# ---------------------------------------------------------------- dirty rule (D065)

def test_clean_status_is_not_dirty() -> None:
    assert dirty_paths_from_status("") == []


def test_tracked_changes_count_anywhere() -> None:
    status = "\n".join([
        " M src/eil/intervention_learning.py",
        "M  README.md",
        " D runs/finetune/a/meta.json",  # tracked file under runs/ still counts
        "R  old.py -> scripts/new.py",
    ])
    assert dirty_paths_from_status(status) == [
        "src/eil/intervention_learning.py", "README.md", "runs/finetune/a/meta.json", "scripts/new.py",
    ]


@pytest.mark.parametrize("path", [
    "src/utils/new.py", "scripts/new_script.py", "configs/frozen.json", "tests/test_new.py", "cluster/run_ft.sh",
])
def test_untracked_code_counts(path: str) -> None:
    assert dirty_paths_from_status(f"?? {path}") == [path]


@pytest.mark.parametrize("path", [
    "runs/finetune/x/20261115_120000_pass1_1/meta.json",
    "logs/datasets/eil/ppo/q_meta.json",
    "experiments/ppo/SafetyPointGoal2-v0/n_timesteps_1/meta.json",
    "cluster/local_ft_20261115_120000.log",
])
def test_untracked_run_outputs_ignored(path: str) -> None:
    assert dirty_paths_from_status(f"?? {path}") == []


def test_require_clean_raises_unless_allowed() -> None:
    with pytest.raises(RuntimeError, match="ft_exp.*--allow-dirty"):
        require_clean(_git(dirty=True), what="ft_exp", allow_dirty=False)
    assert require_clean(_git(dirty=True), what="ft_exp", allow_dirty=True)["dirty_allowed"] is True
    assert require_clean(_git(dirty=False), what="ft_exp", allow_dirty=False)["dirty_allowed"] is False


# ---------------------------------------------------------------- git fallback (D031)

def _git_fails(*args, **kwargs):
    raise subprocess.CalledProcessError(128, "git")


def test_git_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_meta, "_git", _git_fails)
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("GIT_DIRTY", "0")
    git = git_provenance()
    assert git["source"] == "env" and git["commit"] == "deadbeef" and git["dirty"] is False

    monkeypatch.setenv("GIT_DIRTY", "1")
    assert git_provenance()["dirty"] is True


def test_git_missing_binary_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")
    monkeypatch.setattr(run_meta, "_git", no_git)
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("GIT_DIRTY", "true")
    assert git_provenance()["source"] == "env"


def test_git_and_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_meta, "_git", _git_fails)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.delenv("GIT_DIRTY", raising=False)
    with pytest.raises(RuntimeError, match="GIT_COMMIT"):
        git_provenance()


def test_bad_git_dirty_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_meta, "_git", _git_fails)
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("GIT_DIRTY", "maybe")
    with pytest.raises(ValueError, match="GIT_DIRTY"):
        git_provenance()


# ---------------------------------------------------------------- hashes, paths, schema

def test_sha256_changes_with_one_byte(tmp_path: Path) -> None:
    f = tmp_path / "q_theta.pt"
    f.write_bytes(b"\x00" * 1000)
    h1 = sha256_file(f)
    f.write_bytes(b"\x00" * 999 + b"\x01")
    assert sha256_file(f) != h1


def test_paths_are_repo_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_meta, "CODE_ROOT", tmp_path)
    ds = tmp_path / "logs" / "datasets" / "eil" / "ds_v3"
    ds.mkdir(parents=True)
    (ds / "data.npz").write_bytes(b"data")
    torch.save({"state_dict": {}, "obs_dim": 1, "act_dim": 1, "B": 0.0, "lambda_I": 1.0}, ds / "q_theta.pt")
    model = tmp_path / "experiments" / "ppo" / "model" / "model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"zip")

    q = q_info(ds / "q_theta.pt")
    assert q["q_path"] == "logs/datasets/eil/ds_v3/q_theta.pt"
    assert q["dataset_id"] == "ds_v3" and q["data_npz_sha256"] == sha256_file(ds / "data.npz")
    meta = build_run_meta(
        run_kind="finetune", run_pass="pass1", exp_name="ft", git=_git(), device="cpu",
        **q, base_model_path=run_meta.repo_relative(model),
        extra={"base_run_dir": run_meta.repo_relative(model.parent.parent)},
    )
    path = tmp_path / "meta.json"
    write_meta(path, meta)
    text = path.read_text()
    assert str(tmp_path) not in text
    for s in _walk_strings(json.loads(text)):
        assert not s.startswith("/"), s


def test_repo_relative_rejects_outside_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_meta, "CODE_ROOT", tmp_path / "repo")
    with pytest.raises(ValueError, match="outside the code repo"):
        run_meta.repo_relative(tmp_path / "elsewhere" / "q_theta.pt")


def test_schema_has_every_field_and_rejects_typos() -> None:
    meta = build_run_meta(run_kind="q_train", run_pass="pass1", exp_name=None, git=_git())
    assert set(run_meta.SCHEMA_KEYS) <= set(meta)
    assert meta["schema_version"] == 1 and meta["status"] == "running" and meta["pass"] == "pass1"
    assert meta["seed_pool"] is None and meta["optuna"] is None and meta["env_config_path"] is None
    assert set(meta["slurm"]) == {"job_id", "array_job_id", "array_task_id", "node"}
    with pytest.raises(KeyError, match="betta"):
        build_run_meta(run_kind="finetune", run_pass="pass1", exp_name="x", git=_git(), betta=0.1)
    with pytest.raises(ValueError, match="run_kind"):
        build_run_meta(run_kind="eval", run_pass=None, exp_name="x", git=_git())


def test_beta_zero_recorded_as_float(tmp_path: Path) -> None:
    meta = build_run_meta(run_kind="finetune", run_pass="pass1", exp_name="x", git=_git(), beta=0)
    path = tmp_path / "meta.json"
    write_meta(path, meta)
    loaded = json.loads(path.read_text())
    assert loaded["beta"] == 0.0 and isinstance(loaded["beta"], float)
    assert '"beta": 0.0' in path.read_text()


def test_exec_mode_from_slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert build_run_meta(run_kind="q_train", run_pass="p", exp_name=None, git=_git())["exec_mode"] == "local"
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    meta = build_run_meta(run_kind="q_train", run_pass="p", exp_name=None, git=_git())
    assert meta["exec_mode"] == "slurm" and meta["slurm"]["job_id"] == "4242"


# ---------------------------------------------------------------- lifecycle (D027, D037)

def test_finished_run(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    meta = build_run_meta(run_kind="q_train", run_pass="pass1", exp_name=None, git=_git())
    with recorded_run(path, meta):
        assert json.loads(path.read_text())["status"] == "running"
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "finished" and loaded["end_time"] and loaded["error"] is None


def test_failed_run_records_error_and_reraises(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    meta = build_run_meta(run_kind="q_train", run_pass="pass1", exp_name=None, git=_git())
    with pytest.raises(ZeroDivisionError):
        with recorded_run(path, meta):
            1 / 0
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "failed" and loaded["end_time"]
    assert loaded["error"]["type"] == "ZeroDivisionError" and "division" in loaded["error"]["message"]


def test_atomic_write_leaves_no_tmp(tmp_path: Path) -> None:
    write_meta(tmp_path / "meta.json", {"a": 1})
    write_meta(tmp_path / "meta.json", {"a": 2})
    assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]
    assert json.loads((tmp_path / "meta.json").read_text()) == {"a": 2}


def test_failed_serialization_leaves_old_file_and_no_tmp(tmp_path: Path) -> None:
    write_meta(tmp_path / "meta.json", {"a": 1})
    with pytest.raises(TypeError):
        write_meta(tmp_path / "meta.json", {"a": object()})
    assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]
    assert json.loads((tmp_path / "meta.json").read_text()) == {"a": 1}


# ---------------------------------------------------------------- run dir name (D032, D064)

def test_run_dir_name_has_pass_and_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    name = finetune_run_dir_name("pass1", now=datetime(2026, 11, 15, 12, 30, 5), pid=4321)
    assert name == "20261115_123005_pass1_4321"


def test_run_dir_name_prefers_slurm_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "998877")
    name = finetune_run_dir_name("pass2", now=datetime(2026, 11, 15, 12, 30, 5), pid=4321)
    assert name == "20261115_123005_pass2_998877"


@pytest.mark.parametrize("bad", ["", "pass 1", "pass/1", "pass_1"])
def test_run_dir_name_rejects_bad_pass(bad: str) -> None:
    with pytest.raises(ValueError):
        finetune_run_dir_name(bad, pid=1)


# ---------------------------------------------------------------- fine-tune refuses a dirty tree

def test_finetune_dirty_raises_before_any_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    q_path = tmp_path / "q" / "q_theta.pt"
    q_path.parent.mkdir()
    torch.save({"state_dict": {}, "obs_dim": 1, "act_dim": 1, "B": 0.0}, q_path)
    base_dir = tmp_path / "base"
    (base_dir / "model").mkdir(parents=True)
    (base_dir / "model" / "model.zip").write_bytes(b"zip")

    reward = {"reward_wrapper": "fastsafe", "reward_kwargs": {"goal_bonus": 50.0, "shaping_k": 1.0}}
    exps = {
        "ft": {"name": "ft", "base_exp": "base", "env_id": "SafetyPointGoal2-v0", "timesteps": 1, "seed": 0,
               "n_envs": 1, "q_path": str(q_path), "beta": 0.0, "q_device": "cpu", **reward},
        "base": {"env_id": "SafetyPointGoal2-v0", "timesteps": 1, "seed": 0, **reward},
    }
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(finetune_policy, "ROOT", repo)
    monkeypatch.setattr(finetune_policy, "get_experiment", lambda name: exps[name])
    monkeypatch.setattr(finetune_policy, "get_run_dir", lambda exp: base_dir)
    monkeypatch.setattr(finetune_policy, "git_provenance", lambda: _git(dirty=True))

    with pytest.raises(RuntimeError, match="ft.*uncommitted"):
        finetune_policy.finetune_from_registry("ft", run_pass="pass1")
    assert list(repo.iterdir()) == []


def test_git_provenance_on_this_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Read-only git on the real code repo, run from a different cwd: the root comes from the module path.
    monkeypatch.chdir(tmp_path)
    git = git_provenance()
    assert git["source"] == "git" and len(git["commit"]) == 40
    assert isinstance(git["dirty"], bool) and isinstance(git["dirty_paths"], list)
