"""
One run record (meta.json) shared by fine-tune, base training and Q training.

Lifecycle: `start_run_meta` writes status="running" with the provenance captured at launch,
`finish_run_meta` sets end_time and "finished" or "failed". `recorded_run` wraps both and marks the
run failed (with the error) on any exception, then re-raises. A SIGKILLed job stays "running".

Every field of the schema is always present; a field that does not apply to this run kind, or that a
later task fills (seed_pool: 0.2, optuna: 3.2, env_config_path: 0.8), is null. Paths are repo-relative (D035).
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA_VERSION = 1
RUN_KINDS = ("finetune", "base_train", "q_train")

# Code repo root from this file's location, never the cwd (D031).
CODE_ROOT = Path(__file__).resolve().parents[2]

# Untracked files here are code and make the tree dirty (D065). Everything else untracked
# (runs/, logs/, experiments/, cluster/*.log) is run output.
_CODE_DIRS = ("src/", "scripts/", "configs/", "tests/")

_PASS_RE = re.compile(r"[A-Za-z0-9.-]+")

_PACKAGES = ("torch", "stable-baselines3", "safety-gymnasium", "gymnasium", "numpy", "optuna", "scipy")

# Floats even when the definition says 0 or 1: "beta": 0.0 must not read as an int (or as a missing value).
_FLOAT_FIELDS = ("beta", "B", "lambda_I")


def _schema_template() -> dict[str, Any]:
    return {
        # identity / status
        "schema_version": SCHEMA_VERSION,
        "run_kind": None,
        "status": None,
        "pass": None,
        "exp_name": None,
        "start_time": None,
        "end_time": None,
        "error": None,
        # execution
        "exec_mode": None,
        "hostname": None,
        "slurm": None,
        "hardware": None,
        # code
        "git": None,
        "packages": None,
        # run params
        "seed": None,
        "beta": None,
        "timesteps": None,
        "n_envs": None,
        "env_id": None,
        "reward_wrapper": None,
        "reward_kwargs": None,
        # hyperparameters (D024)
        "hyperparameters": {"requested": None, "effective": None},
        # Q
        "q_path": None,
        "q_sha256": None,
        "B": None,
        "lambda_I": None,
        "dataset_id": None,
        "data_npz_sha256": None,
        "q_hash_check": None,
        # base model (0.8, D039/D040)
        "base_exp": None,
        "base_model_path": None,
        "base_model_sha256": None,
        # filled by later tasks
        "seed_pool": None,   # 0.2: {name, file, sha256, n}
        "optuna": None,      # 3.2: {study, sampler_seed, storage}
        "env_config_path": None,  # 0.8
    }


SCHEMA_KEYS = tuple(_schema_template())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_relative(path: str | Path | None) -> str | None:
    """Path relative to the code repo root. Raises if it lies outside, since an absolute path would be machine-specific."""
    if path is None:
        return None
    p = Path(path).resolve()
    try:
        return p.relative_to(CODE_ROOT).as_posix()
    except ValueError:
        raise ValueError(
            f"{p} is outside the code repo {CODE_ROOT}; meta.json stores repo-relative paths only. "
            "Move the file into the repo."
        ) from None


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_run_pass(run_pass: str) -> str:
    # The pass goes into a directory name, so no separators or spaces.
    if not isinstance(run_pass, str) or not _PASS_RE.fullmatch(run_pass):
        raise ValueError(f"run pass {run_pass!r} is not a valid tag; use letters, digits, '.' or '-', e.g. 'pass1'.")
    return run_pass


# ---------------------------------------------------------------- git 

def dirty_paths_from_status(porcelain: str) -> list[str]:
    """
    Paths that make the tree dirty, from `git status --porcelain=v1 --untracked-files=all` output.

    Any tracked change counts. Untracked files count only under src/, scripts/, configs/, tests/
    or as a cluster/*.sh script; untracked run outputs (runs/, logs/, experiments/, cluster/*.log) do not.
    """
    dirty = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:  # rename: "R  old -> new"
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if code == "!!":
            continue
        if code == "??":
            is_code = path.startswith(_CODE_DIRS) or (path.startswith("cluster/") and path.endswith(".sh"))
            if not is_code:
                continue
        dirty.append(path)
    return dirty


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(CODE_ROOT), *args], check=True, capture_output=True, text=True,
    ).stdout


def _git_from_repo() -> dict[str, Any]:
    toplevel = Path(_git("rev-parse", "--show-toplevel").strip()).resolve()
    # Without its own .git, git would report an enclosing repo's commit.
    if toplevel != CODE_ROOT:
        raise RuntimeError(f"git toplevel {toplevel} is not the code repo {CODE_ROOT}")
    commit = _git("rev-parse", "HEAD").strip()
    dirty_paths = dirty_paths_from_status(_git("status", "--porcelain=v1", "--untracked-files=all"))
    diff = _git("diff", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(dirty_paths),
        "dirty_allowed": None,
        "dirty_paths": dirty_paths,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "source": "git",
    }


def _parse_git_dirty(value: str) -> bool:
    v = value.strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    raise ValueError(f"GIT_DIRTY={value!r} is not a boolean; the jobscript must export 0/1 or true/false.")


def git_provenance() -> dict[str, Any]:
    """Commit and dirty state of the code repo; GIT_COMMIT/GIT_DIRTY env vars if git cannot run (D031)."""
    try:
        return _git_from_repo()
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        git_error = exc

    commit = os.environ.get("GIT_COMMIT")
    dirty = os.environ.get("GIT_DIRTY")
    if not commit or dirty is None:
        raise RuntimeError(
            f"Cannot read git state of {CODE_ROOT} ({git_error}) and GIT_COMMIT/GIT_DIRTY are not set. "
            "Run where git works, or export GIT_COMMIT and GIT_DIRTY from the jobscript."
        )
    return {
        "commit": commit,
        "dirty": _parse_git_dirty(dirty),
        "dirty_allowed": None,
        "dirty_paths": None,
        "diff_sha256": None,
        "source": "env",
    }


def require_clean(git: Mapping[str, Any], *, what: str, allow_dirty: bool) -> dict[str, Any]:
    """For runs that may be reported (fine-tunes): a dirty tree is an error unless explicitly allowed (D029)."""
    if git["dirty"] and not allow_dirty:
        paths = git.get("dirty_paths")
        shown = f": {', '.join(paths[:10])}{' ...' if len(paths) > 10 else ''}" if paths else ""
        raise RuntimeError(
            f"{what}: code repo {CODE_ROOT} has uncommitted changes{shown}. "
            "Commit them first, or pass --allow-dirty for a run that will not be reported."
        )
    return {**git, "dirty_allowed": bool(allow_dirty)}


# ---------------------------------------------------------------- environment

def execution_info(device: str | None = None) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    return {
        "exec_mode": "slurm" if job_id else "local",
        "hostname": socket.gethostname(),
        "slurm": {
            "job_id": job_id,
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
        "hardware": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "device": None if device is None else str(device),
        },
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# ---------------------------------------------------------------- run-specific fields

def q_info(q_path: str | Path) -> dict[str, Any]:
    """Q file fields: path, file hash, B and lambda_I as stored (null if absent), dataset id and data.npz hash."""
    import torch

    q_path = Path(q_path)
    ckpt = torch.load(q_path, map_location="cpu")
    data_npz = q_path.parent / "data.npz"
    return {
        "q_path": repo_relative(q_path),
        "q_sha256": sha256_file(q_path),
        "B": ckpt.get("B"),
        "lambda_I": ckpt.get("lambda_I"),
        "dataset_id": q_path.parent.name,
        "data_npz_sha256": sha256_file(data_npz) if data_npz.exists() else None,
    }


def _schedule_value(v: Any) -> Any:
    # SB3 turns learning_rate and clip_range into schedules; the value at progress 1.0 is the start value.
    return float(v(1.0)) if callable(v) else v


def _plain(v: Any) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return int(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def sb3_effective_hyperparameters(model: Any) -> dict[str, Any]:
    """Values read back from the loaded/built model, so a check can compare ft and ref (D024). Keys an algorithm lacks are null."""
    lr = getattr(model, "lr_schedule", None)
    clip = getattr(model, "clip_range", None)
    out = {
        "learning_rate": _plain(_schedule_value(lr)) if lr is not None else None,
        "clip_range": _plain(_schedule_value(clip)) if clip is not None else None,
    }
    for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "n_envs"):
        out[k] = _plain(getattr(model, k, None))
    out["device"] = str(model.device)
    return out


def finetune_run_dir_name(run_pass: str, *, now: datetime | None = None,
                          job_id: str | None = None, pid: int | None = None) -> str:
    """`{YYYYmmdd_HHMMSS}_{pass}_{SLURM_JOB_ID or pid}` (D032, D064): sorts by time, unique per job."""
    validate_run_pass(run_pass)
    now = now or datetime.now()
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    suffix = job_id if job_id else str(pid if pid is not None else os.getpid())
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{run_pass}_{suffix}"


# ---------------------------------------------------------------- build / write

def build_run_meta(*, run_kind: str, run_pass: str | None, exp_name: str | None, git: Mapping[str, Any],
                   device: str | None = None, extra: Mapping[str, Any] | None = None,
                   **fields: Any) -> dict[str, Any]:
    """
    Full schema with provenance filled in. `fields` must be schema keys (a typo raises instead of
    vanishing); `extra` holds non-schema top-level keys kept for older readers.
    """
    if run_kind not in RUN_KINDS:
        raise ValueError(f"run_kind {run_kind!r} is not one of {RUN_KINDS}")
    if run_pass is not None:
        validate_run_pass(run_pass)
    reserved = {"schema_version", "run_kind", "status", "pass", "exp_name", "start_time", "end_time", "error",
                "exec_mode", "hostname", "slurm", "hardware", "git", "packages"}
    unknown = sorted(set(fields) - set(SCHEMA_KEYS))
    if unknown:
        raise KeyError(f"meta fields {unknown} are not in the run meta schema (src/utils/run_meta.py)")
    clash = sorted(set(fields) & reserved)
    if clash:
        raise KeyError(f"meta fields {clash} are set by build_run_meta itself")
    extra = dict(extra or {})
    clash = sorted(set(extra) & set(SCHEMA_KEYS))
    if clash:
        raise KeyError(f"extra keys {clash} are schema fields; pass them as fields")

    meta = _schema_template()
    meta.update(
        run_kind=run_kind,
        status="running",
        exp_name=exp_name,
        start_time=utc_now(),
        git=dict(git),
        packages=package_versions(),
        **execution_info(device),
    )
    meta["pass"] = run_pass
    meta.update(fields)
    for k in _FLOAT_FIELDS:
        if meta[k] is not None:
            meta[k] = float(meta[k])
    meta.update(extra)
    return meta


def write_meta(path: str | Path, meta: Mapping[str, Any]) -> None:
    """Atomic: a reader sees the old or the new file, never half of one."""
    path = Path(path)
    text = json.dumps(meta, indent=2)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def start_run_meta(path: str | Path, meta: dict[str, Any]) -> dict[str, Any]:
    meta["status"] = "running"
    write_meta(path, meta)
    return meta


def finish_run_meta(path: str | Path, meta: dict[str, Any], *, error: BaseException | None = None) -> dict[str, Any]:
    meta["end_time"] = utc_now()
    if error is None:
        meta["status"] = "finished"
        meta["error"] = None
    else:
        meta["status"] = "failed"
        meta["error"] = {"type": type(error).__name__, "message": str(error)}
    write_meta(path, meta)
    return meta


@contextmanager
def recorded_run(path: str | Path, meta: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """
    Write the running meta, yield it for in-run updates (effective hyperparameters, Q hash),
    then finish it. Any exception, KeyboardInterrupt included, marks the run failed and is re-raised (D037).
    """
    start_run_meta(path, meta)
    try:
        yield meta
    except BaseException as exc:
        finish_run_meta(path, meta, error=exc)
        raise
    finish_run_meta(path, meta)
