from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_relative(path) -> str:
    return Path(path).resolve().relative_to(CODE_ROOT).as_posix()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(CODE_ROOT), *args], check=True, capture_output=True, text=True).stdout

    # Only code counts as dirty; run outputs (runs/, logs/, experiments/, cluster/*.log) do not.
    status = git("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "configs", "cluster/*.sh")
    dirty_paths = [line[3:] for line in status.splitlines()]
    return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(dirty_paths), "dirty_paths": dirty_paths}


@contextmanager
def recorded_run(path: Path, exp_name, run_pass, git: dict, **fields):
    """Write meta.json with status "running"; on exit set "finished", or "failed" with the error."""
    job_id = os.environ.get("SLURM_JOB_ID")
    meta = {
        "exp_name": exp_name, "status": "running", "error": None, "pass": run_pass,
        "start_time": utc_now(), "end_time": None,
        "exec_mode": "slurm" if job_id else "local", "hostname": socket.gethostname(), "slurm_job_id": job_id,
        "git": git, **fields,
    }
    path.write_text(json.dumps(meta, indent=2))
    try:
        yield meta
        meta["status"] = "finished"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta["end_time"] = utc_now()
        path.write_text(json.dumps(meta, indent=2))
