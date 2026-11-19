"""One beta-sweep trial per call: ask the study for beta, fine-tune and evaluate each sweep seed, tell the mean.

The study of a base lives in runs/sweeps/<base>_<run_pass>/journal.log, next to one log file per seed-run.
Only one trial runs at a time per study, so a job can open the journal, run one trial and exit.
"""
from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from src.evaluation.training_metrics.sb3_metrics import load_tb_scalars
from src.experiments.registry import get_experiment, get_run_dir
from src.utils.run_meta import CODE_ROOT, git_info, repo_relative, sha256_file
from src.utils.seed_pools import SEED_POOLS_PATH, load_pool, load_sampler_seed

# Safe-success key per base, fixed before the sweep
OBJECTIVE_KEY = {
    "ppo_5m_goal2_safe": "safe_success_rate_hazards_only",
    "ppo_5m700k_goal2": "safe_success_rate_strict",
}
# Left out of the per-seed results: per-episode seeds and fields fixed by the base
SUMMARY_DROP = ("seeds", "environment", "algorithm", "reward_wrapper", "reward_kwargs")
TB_KEYS = ("q/active_frac", "q/penalty_ratio", "train/value_loss", "train/approx_kl", "train/clip_fraction")
BETA_RANGE = (1.0, 300.0)
STARTUP_BETAS = [300 ** (i / 4) for i in range(5)]  # log-spaced over BETA_RANGE, both ends included
N_SEEDS = 3


def _meta(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _trial_beta(t) -> float | None:
    # A job killed between ask and suggest leaves no params; an enqueued beta is still in fixed_params
    return t.params.get("beta", t.system_attrs.get("fixed_params", {}).get("beta"))


def _reusable_run(name: str, run_pass: str, n_eval: int, commit: str, allow_dirty: bool) -> tuple[Path, bool] | None:
    """Newest run dir of name with a finished fine-tune at this commit, and whether its selection eval is done."""
    def ok(meta):
        return (meta.get("status") == "finished" and meta["git"]["commit"] == commit
                and (allow_dirty or not meta["git"]["dirty"]))

    for run_dir in sorted((CODE_ROOT / "runs" / "finetune" / name).glob(f"*_{run_pass}_*"), reverse=True):
        eval_dir = run_dir / "final_eval" / f"selection_n{n_eval}"
        if not ok(_meta(run_dir / "meta.json")):
            continue
        if not eval_dir.exists():
            return run_dir, False
        # A half-written eval dir is skipped: the eval refuses an existing out dir
        if ok(_meta(eval_dir / "meta.json")) and (eval_dir / "summary.json").exists():
            return run_dir, True
    return None


def _run_seeds(base: str, beta: float, *, run_pass: str, n_eval: int, allow_dirty: bool, serial: bool,
               commit: str, log_prefix: Path) -> list[dict]:
    dirty_flag = ["--allow-dirty"] if allow_dirty else []
    # repr keeps every digit, so the beta parsed back from the name is this exact float
    names = [f"{base}_v3_sweep_b{beta!r}_s{i}" for i in range(N_SEEDS)]
    found = [_reusable_run(name, run_pass, n_eval, commit, allow_dirty) for name in names]
    run_dirs = [f[0] if f else None for f in found]
    eval_done = [bool(f and f[1]) for f in found]

    procs = {}
    for i, name in enumerate(names):
        if run_dirs[i] is None:
            log = open(f"{log_prefix}_s{i}.log", "a")
            procs[i] = (subprocess.Popen(
                [sys.executable, str(CODE_ROOT / "scripts" / "finetune_policy.py"), name, "--run-pass", run_pass,
                 *dirty_flag], cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT), log)
            if serial:
                procs[i][0].wait()
    for i, (p, log) in procs.items():
        p.wait()
        log.close()
        if p.returncode == 0:
            # finetune_policy.py names its run dir <timestamp>_<pass>_<SLURM_JOB_ID or its pid>
            job = os.environ.get("SLURM_JOB_ID") or p.pid
            run_dirs[i] = sorted((CODE_ROOT / "runs" / "finetune" / names[i]).glob(f"*_{run_pass}_{job}"))[-1]

    # One eval at a time: each eval starts one worker per CPU of the job.
    # Seeds whose fine-tune finished are evaluated even if another failed, so a retry reuses them.
    for i, run_dir in enumerate(run_dirs):
        if run_dir is not None and not eval_done[i]:
            with open(f"{log_prefix}_s{i}.log", "a") as log:
                subprocess.run(
                    [sys.executable, str(CODE_ROOT / "scripts" / "evaluate_seeded.py"), names[i],
                     "--run-dir", str(run_dir), "--pool", "selection", "--n", str(n_eval), "--run-pass", run_pass,
                     *dirty_flag], cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    failed = [names[i] for i, (p, _) in procs.items() if p.returncode != 0]
    if failed:
        raise RuntimeError(f"fine-tune failed for {failed}; see {log_prefix}_s*.log")

    results = []
    for i, run_dir in enumerate(run_dirs):
        eval_dir = run_dir / "final_eval" / f"selection_n{n_eval}"
        summary = json.loads((eval_dir / "summary.json").read_text())
        tb = load_tb_scalars(run_dir / "logs" / "PPO_0")
        results.append({
            "seed_index": i, "run_dir": repo_relative(run_dir), "eval_dir": repo_relative(eval_dir),
            "reused_finetune": found[i] is not None, "reused_eval": eval_done[i],
            **{k: v for k, v in summary.items() if k not in SUMMARY_DROP},
            "tb": {k: {"last": float(tb[k][1][-1]), "mean": float(tb[k][1].mean())} for k in TB_KEYS},
        })
    return results


def run_trial(base: str, *, run_pass: str, allow_dirty: bool = False, n_trials: int = 12, n_eval: int = 500,
              serial: bool = False) -> None:
    sweep_dir = CODE_ROOT / "runs" / "sweeps" / f"{base}_{run_pass}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sampler_seed = load_sampler_seed()
    study = optuna.create_study(
        study_name=f"beta_{base}", direction="maximize", load_if_exists=True,
        storage=JournalStorage(JournalFileBackend(str(sweep_dir / "journal.log"))),
        sampler=optuna.samplers.GPSampler(seed=sampler_seed, n_startup_trials=len(STARTUP_BETAS),
                                          deterministic_objective=False),
    )
    n_complete = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    if n_complete >= n_trials:
        print(f"[sweep] beta_{base}: {n_complete} trials complete, nothing to do")
        return
    # Set when the sweep must wait for the human; setting it to "" resumes it
    if study.user_attrs.get("stopped"):
        print(f"[sweep] beta_{base}: stopped: {study.user_attrs['stopped']}")
        return

    git = git_info()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError(f"beta_{base}: uncommitted code changes {git['dirty_paths']}; commit them or pass --allow-dirty")
    seeds = load_pool("sweep")[:N_SEEDS]
    base_model = CODE_ROOT / get_run_dir(get_experiment(base)) / "model" / "model.zip"
    setup = {
        "code_commit": git["commit"], "run_pass": run_pass, "beta_range": list(BETA_RANGE),
        "startup_betas": STARTUP_BETAS, "sweep_seeds": seeds, "sampler_seed": sampler_seed,
        "n_trials": n_trials, "selection_n": n_eval, "objective_key": OBJECTIVE_KEY[base],
        "seed_pools_file": repo_relative(SEED_POOLS_PATH), "seed_pools_sha256": sha256_file(SEED_POOLS_PATH),
        "q_sha256": sha256_file(CODE_ROOT / get_experiment(f"{base}_v3_b0.0")["q_path"]),
        "base_model_sha256": sha256_file(base_model), "optuna_version": optuna.__version__,
        "uv_lock_sha256": sha256_file(CODE_ROOT / "uv.lock"),
    }
    if "code_commit" not in study.user_attrs:
        for beta in STARTUP_BETAS:
            study.enqueue_trial({"beta": beta})
        for k, v in setup.items():
            study.set_user_attr(k, v)
    # Trials are comparable only with the same code, seeds, episodes and inputs; the budget may be raised.
    recorded = study.user_attrs
    changed = {k: (recorded[k], v) for k, v in setup.items() if k != "n_trials" and recorded[k] != v}
    if changed:
        raise ValueError(f"beta_{base}: (recorded, current) differ: {changed}")

    # Only one trial runs at a time, so a RUNNING trial here was left by a killed or failed job.
    # Waiting trials are handed out before the sampler is asked
    failed_betas = [_trial_beta(t) for t in study.get_trials(deepcopy=False, states=(TrialState.FAIL,))]
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        beta = _trial_beta(t)
        study.tell(t.number, state=TrialState.FAIL)
        if beta is None:
            print(f"[sweep] trial {t.number} was left running without a beta; failed")
        elif beta in failed_betas:
            reason = f"beta {beta!r} failed twice; see trial logs"
            study.set_user_attr("stopped", reason)
            raise RuntimeError(f"beta_{base}: {reason}")
        else:
            study.enqueue_trial({"beta": beta})
            print(f"[sweep] trial {t.number} (beta={beta!r}) was left running; failed and re-enqueued")

    trial = study.ask()
    beta = trial.suggest_float("beta", *BETA_RANGE, log=True)
    # A re-proposed measured beta adds no new point; the human decides how to go on
    for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        if abs(math.log(beta) - math.log(t.params["beta"])) < 1e-6:
            reason = f"GP re-proposed beta {beta!r} (trial {t.number} measured {t.params['beta']!r})"
            study.tell(trial, state=TrialState.FAIL)
            study.set_user_attr("stopped", reason)
            raise RuntimeError(f"beta_{base}: {reason}")
    trial.set_user_attr("beta", beta)
    trial.set_user_attr("seeds", seeds)
    trial.set_user_attr("dirty", git["dirty"])
    trial.set_user_attr("job_id", os.environ.get("SLURM_JOB_ID"))
    trial.set_user_attr("hostname", socket.gethostname())
    print(f"[sweep] beta_{base}: trial {trial.number} beta={beta!r}")

    start = time.monotonic()
    runs = _run_seeds(base, beta, run_pass=run_pass, n_eval=n_eval, allow_dirty=allow_dirty, serial=serial,
                      commit=git["commit"], log_prefix=sweep_dir / f"trial{trial.number}")
    value = sum(r[OBJECTIVE_KEY[base]] for r in runs) / len(runs)
    trial.set_user_attr("runs", runs)
    trial.set_user_attr("wall_time_s", time.monotonic() - start)
    study.tell(trial, value)
    print(f"[sweep] beta_{base}: trial {trial.number} beta={beta!r} {OBJECTIVE_KEY[base]}={value:.4f}")
