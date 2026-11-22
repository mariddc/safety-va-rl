"""Choose beta after the sweep: GP fit to a study's completed trials (top 3), then the best confirm mean."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna._gp import gp, prior
from optuna._gp.search_space import SearchSpace
from optuna.samplers._gp.sampler import _standardize_values
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from src.sweeps.finetune_sweep import OBJECTIVE_KEY
from src.utils.run_meta import sha256_file

EARLIER_RULE_MARGIN = 0.01


def fit_top3(base: str, journal: Path, out_dir: Path) -> list[float]:
    """Write top3.json: the 3 measured beta with the highest GP posterior mean."""
    # load_study only reads the journal
    study = optuna.load_study(study_name=f"beta_{base}",
                              storage=JournalStorage(JournalFileBackend(str(journal))))
    # A re-proposal is the normal end of the sweep; any other stop means something broke
    stopped = study.user_attrs.get("stopped")
    if stopped and not stopped.startswith("GP re-proposed"):
        raise RuntimeError(f"{journal}: sweep stopped: {stopped}; a human decides how to go on")
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

    # The model GPSampler(deterministic_objective=False) fits: beta normalized on log scale to [0, 1],
    # values standardized, Matern 5/2 kernel + noise with Optuna's priors, fitted from its default start.
    dists = [t.distributions["beta"] for t in trials]
    if any(d != dists[0] for d in dists):
        raise ValueError(f"{journal}: completed trials have different beta distributions "
                         f"{sorted(set(map(str, dists)))}")
    low, high = dists[0].low, dists[0].high
    space = SearchSpace({"beta": dists[0]})
    X = space.get_normalized_params(trials)
    y, y_mean, y_std = _standardize_values(np.array([t.values for t in trials]))
    gpr = gp.fit_kernel_params(X=X, Y=y[:, 0], is_categorical=space.is_categorical,
                               log_prior=prior.default_log_prior, minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
                               deterministic_objective=False)

    def posterior(x: np.ndarray) -> tuple[list[float], list[float]]:
        # sd is that of the mean objective (the latent function), without the noise
        with torch.no_grad():
            mean, var = gpr.posterior(torch.from_numpy(x))
        return ((mean.numpy() * y_std[0] + y_mean[0]).tolist(), (np.sqrt(var.numpy()) * y_std[0]).tolist())

    trial_mean, trial_sd = posterior(X)
    rows = [{"number": t.number, "beta": t.params["beta"], "value": t.value, "gp_mean": m, "gp_sd": s}
            for t, m, s in zip(trials, trial_mean, trial_sd)]
    top3 = [r["beta"] for r in sorted(rows, key=lambda r: -r["gp_mean"])[:3]]
    u = np.linspace(0.0, 1.0, 101)[:, None]
    grid_mean, grid_sd = posterior(u)

    noise_var = gpr.noise_var.item()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "top3.json").write_text(json.dumps({
        "betas": top3, "base": base, "rule": "the 3 measured beta with the highest GP posterior mean",
        "objective_key": OBJECTIVE_KEY[base], "journal": str(journal), "journal_sha256": sha256_file(journal),
        "study_name": study.study_name, "n_trials": len(trials), "beta_range": [low, high],
        "gp": {
            "kernel": "Matern 5/2 on log(beta) scaled to [0, 1], plus noise; values standardized",
            "inverse_squared_lengthscale": gpr.inverse_squared_lengthscales.tolist(),
            "kernel_scale": gpr.kernel_scale.item(), "noise_var": noise_var,
            "value_mean": float(y_mean[0]), "value_std": float(y_std[0]),
            "noise_sd": math.sqrt(noise_var) * float(y_std[0]), "optuna_version": optuna.__version__,
        },
        "trials": rows,
        "grid": {"beta": np.exp(math.log(low) + u[:, 0] * (math.log(high) - math.log(low))).tolist(),
                 "gp_mean": grid_mean, "gp_sd": grid_sd},
    }, indent=2))
    for r in rows:
        print(f"[beta_select] trial {r['number']} beta={r['beta']:.4g} value={r['value']:.4f} "
              f"gp_mean={r['gp_mean']:.4f} gp_sd={r['gp_sd']:.4f}")
    print(f"[beta_select] {base}: noise_sd={math.sqrt(noise_var) * y_std[0]:.4f} top3={top3}")
    return top3


def select_beta(base: str, top3_path: Path, runs_paths: list[Path], out_dir: Path) -> float:
    """Write chosen_beta.json: the top-3 beta with the highest mean over the confirm seeds."""
    key = OBJECTIVE_KEY[base]
    top3 = json.loads(top3_path.read_text())["betas"]
    values, commits = {}, set()
    for path in runs_paths:
        r = json.loads(path.read_text())
        if r["pool"] != "confirm" or r["seed_indices"] != [0, 1, 2]:
            raise ValueError(f"{path}: pool {r['pool']} seeds {r['seed_indices']}, expected confirm seeds 0, 1, 2")
        if r["dirty"]:
            raise ValueError(f"{path}: ran with uncommitted code")
        commits.add(r["code_commit"])
        values[r["beta"]] = [run[key] for run in r["runs"]]
    if sorted(values) != sorted(top3):
        raise ValueError(f"confirm runs have beta {sorted(values)}, {top3_path} has {top3}")
    if len(commits) != 1:
        raise ValueError(f"confirm runs differ in code commit: {sorted(commits)}")

    means = {b: statistics.mean(v) for b, v in values.items()}
    chosen = max(means, key=lambda b: (means[b], -b))
    best = means[chosen]
    earlier = min(b for b in means if means[b] >= best - EARLIER_RULE_MARGIN)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chosen_beta.json").write_text(json.dumps({
        "beta": chosen, "base": base, "objective_key": key,
        "rule": f"highest mean of {key} over the confirm seeds; exact tie: the smaller beta",
        "confirm": [{"beta": b, "mean": means[b], "sd": statistics.stdev(values[b]), "values": values[b]}
                    for b in sorted(means)],
        "earlier_rule": {"beta": earlier, "margin": EARLIER_RULE_MARGIN,
                         "rule": "smallest beta whose confirm mean >= best mean - margin (recorded, not used)"},
        "inputs": {"top3": {"path": str(top3_path), "sha256": sha256_file(top3_path)},
                   "runs": [{"path": str(p), "sha256": sha256_file(p)} for p in runs_paths]},
    }, indent=2))
    for b in sorted(means):
        print(f"[beta_select] beta={b:.4g} confirm mean {key}={means[b]:.4f} values={values[b]}")
    print(f"[beta_select] {base}: chosen beta={chosen!r} (earlier rule, margin {EARLIER_RULE_MARGIN}: {earlier!r})")
    return chosen
