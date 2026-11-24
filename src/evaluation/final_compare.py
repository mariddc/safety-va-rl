"""Paired comparison of the fine-tuned (ft) and beta = 0 (ref) policies on the test episodes.

d_i = metric(ft_i) - metric(ref_i) for training seed i. Across seeds: mean d with a 95% t-interval
mean +- t(0.975, n - 1) * SD / sqrt(n); "improved" only if its lower end is > 0. Within each seed: a paired
bootstrap over episodes (the same resampled indices for ft_i and ref_i). The base policy is a third point.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.sweeps.finetune_sweep import OBJECTIVE_KEY
from src.utils.run_meta import CODE_ROOT, sha256_file

N_BOOT = 10_000
BOOT_SEED = 0
SAFE_KEYS = ("safe_success_rate_hazards_only", "safe_success_rate_strict")

# metric -> (value from episodes.csv, path of the same value in summary.json)
METRICS = {
    "safe_success_rate_hazards_only": (lambda e: e["safe_success_hazards_only"].mean(),
                                       ("safe_success_rate_hazards_only",)),
    "safe_success_rate_strict": (lambda e: e["safe_success_strict"].mean(), ("safe_success_rate_strict",)),
    "success_rate": (lambda e: e["success"].mean(), ("success_rate",)),
    "strict_safety_rate": (lambda e: ((e["hazard_steps"] == 0) & (e["vase_contact_steps"] == 0)).mean(),
                           ("strict_safety_rate",)),
    "cost_mean": (lambda e: e["cost"].mean(), ("cost", "mean")),
    "unsafe_rate": (lambda e: (e["cost"] > 0).mean(), ("cost", "unsafe_rate")),
    "hazard_contact_rate": (lambda e: (e["hazard_steps"] > 0).mean(),
                            ("safety_breakdown", "hazards", "any_contact_rate")),
    "vase_touch_rate": (lambda e: (e["vase_contact_steps"] > 0).mean(),
                        ("safety_breakdown", "vases", "any_touch_rate")),
    "length_mean": (lambda e: e["length"].mean(), ("episode_length", "mean")),
    "time_to_goal_mean": (lambda e: e.loc[e["success"] == 1, "length"].mean(),
                          ("episode_length", "time_to_goal_success_only")),
    "return_mean": (lambda e: e["return"].mean(), ("reward", "mean")),
}
PRIMARY_COLUMN = {"safe_success_rate_hazards_only": "safe_success_hazards_only",
                  "safe_success_rate_strict": "safe_success_strict"}


def _metric_order(key: str) -> list[str]:
    other = [k for k in SAFE_KEYS if k != key]
    return [key] + other + [m for m in METRICS if m not in SAFE_KEYS]


def _load_eval(eval_dir: Path) -> tuple[pd.DataFrame, dict]:
    episodes = pd.read_csv(eval_dir / "episodes.csv")
    summary = json.loads((eval_dir / "summary.json").read_text())
    values = {}
    for name, (fn, path) in METRICS.items():
        values[name] = float(fn(episodes))
        ref = summary
        for k in path:
            ref = ref[k]
        if not math.isclose(values[name], ref, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{eval_dir}: {name} from episodes.csv {values[name]!r} != summary.json {ref!r}")
    return episodes, values


def _t_interval(d: np.ndarray) -> dict:
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return {"mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}


def compare(base: str, index_path: Path, out_dir: Path) -> dict:
    """Write compare.{json,csv,md,png} for one base to out_dir and return the JSON record."""
    key = OBJECTIVE_KEY[base]
    index = json.loads(index_path.read_text())
    if index["base"] != base:
        raise ValueError(f"{index_path}: base {index['base']}, expected {base}")
    beta, evals = index["beta"], index["evals"]
    n_seeds = sum(name.startswith(f"{base}_v3_final_b0.0_s") for name in evals)
    names = {"base": base,
             **{f"ft_{i}": f"{base}_v3_final_b{beta!r}_s{i}" for i in range(n_seeds)},
             **{f"ref_{i}": f"{base}_v3_final_b0.0_s{i}" for i in range(n_seeds)}}
    if sorted(names.values()) != sorted(evals):
        raise ValueError(f"{index_path}: evals {sorted(evals)}, expected {sorted(names.values())}")

    episodes, values, inputs = {}, {}, {}
    for arm, name in names.items():
        eval_dir = CODE_ROOT / evals[name]["eval_dir"]
        episodes[arm], values[arm] = _load_eval(eval_dir)
        inputs[name] = {f: {"path": f"{evals[name]['eval_dir']}/{f}", "sha256": sha256_file(eval_dir / f)}
                        for f in ("episodes.csv", "summary.json")}

    seeds = episodes["base"]["seed"].to_numpy()
    if len(seeds) != index["n"]:
        raise ValueError(f"{names['base']}: {len(seeds)} episodes, index says {index['n']}")
    for arm, e in episodes.items():
        if not np.array_equal(e["seed"].to_numpy(), seeds):
            raise ValueError(f"{names[arm]}: episode seeds differ from {base}'s (set or order)")

    # ft_i and ref_i must share the training seed and differ in beta only as intended
    train_seed = []
    for i in range(n_seeds):
        ft_meta, ref_meta = (json.loads((CODE_ROOT / evals[names[f"{a}_{i}"]]["eval_dir"]).parents[1]
                                        .joinpath("meta.json").read_text()) for a in ("ft", "ref"))
        if ft_meta["seed"] != ref_meta["seed"] or ft_meta["beta"] != beta or ref_meta["beta"] != 0.0:
            raise ValueError(f"{names[f'ft_{i}']} / {names[f'ref_{i}']}: seed {ft_meta['seed']} / "
                             f"{ref_meta['seed']}, beta {ft_meta['beta']} / {ref_meta['beta']}")
        train_seed.append(ft_meta["seed"])

    metrics = _metric_order(key)
    per_seed, summary = [], {}
    for m in metrics:
        ft = np.array([values[f"ft_{i}"][m] for i in range(n_seeds)])
        ref = np.array([values[f"ref_{i}"][m] for i in range(n_seeds)])
        b = values["base"][m]
        summary[m] = {"base": b, "ft_mean": float(ft.mean()), "ref_mean": float(ref.mean()),
                      "d": _t_interval(ft - ref)}
    for i in range(n_seeds):
        row = {"seed_index": i, "train_seed": train_seed[i], "metrics": {}}
        for m in metrics:
            ft, ref, b = values[f"ft_{i}"][m], values[f"ref_{i}"][m], values["base"][m]
            row["metrics"][m] = {"ft": ft, "ref": ref, "d": ft - ref, "ft_minus_base": ft - b,
                                 "ref_minus_base": ref - b}
        per_seed.append(row)

    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))
    col = PRIMARY_COLUMN[key]
    for i, row in enumerate(per_seed):
        d_boot = (episodes[f"ft_{i}"][col].to_numpy()[idx].mean(axis=1)
                  - episodes[f"ref_{i}"][col].to_numpy()[idx].mean(axis=1))
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        row["bootstrap"] = {"ci_low": float(lo), "ci_high": float(hi)}

    d = summary[key]["d"]
    label = "improved" if d["ci_low"] > 0 else "no detectable difference"
    record = {
        "base": base, "primary_key": key, "beta": beta, "n_episodes": len(seeds), "n_seeds": n_seeds,
        "label": label, "t_crit": float(stats.t.ppf(0.975, n_seeds - 1)),
        "rule": "improved only if the lower end of the 95% t-interval of mean d over seeds is > 0",
        "bootstrap": {"B": N_BOOT, "rng": "numpy default_rng", "rng_seed": BOOT_SEED,
                      "interval": "95% percentile", "metric": key},
        "summary": summary, "per_seed": per_seed,
        "inputs": {"index": {"path": str(index_path), "sha256": sha256_file(index_path)}, "evals": inputs},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(record, indent=2))
    _write_csv(record, metrics, out_dir / "compare.csv")
    _write_md(record, metrics, out_dir / "compare.md")
    _plot(record, out_dir / "compare.png")
    print(f"[final_compare] {base}: {key} mean d={d['mean']:+.4f} sd={d['sd']:.4f} "
          f"95% t-interval [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] -> {label}")
    return record


def _write_csv(record: dict, metrics: list[str], path: Path) -> None:
    rows = []
    for row in record["per_seed"]:
        for kind in ("ft", "ref", "d", "ft_minus_base", "ref_minus_base"):
            r = {"row": kind, "seed_index": row["seed_index"], "train_seed": row["train_seed"]}
            r.update({m: row["metrics"][m][kind] for m in metrics})
            if kind == "d":
                r["boot_low"], r["boot_high"] = row["bootstrap"]["ci_low"], row["bootstrap"]["ci_high"]
            rows.append(r)
    s = record["summary"]
    rows.append({"row": "base", **{m: s[m]["base"] for m in metrics}})
    rows.append({"row": "ft_mean", **{m: s[m]["ft_mean"] for m in metrics}})
    rows.append({"row": "ref_mean", **{m: s[m]["ref_mean"] for m in metrics}})
    for stat in ("mean", "sd", "ci_low", "ci_high"):
        rows.append({"row": f"d_{stat}", **{m: s[m]["d"][stat] for m in metrics}})
    fields = ["row", "seed_index", "train_seed"] + metrics + ["boot_low", "boot_high"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_md(record: dict, metrics: list[str], path: Path) -> None:
    key, s = record["primary_key"], record["summary"]
    d = s[key]["d"]
    lines = [
        f"# {record['base']}: ft (beta = {record['beta']:.4g}) vs ref (beta = 0), test episodes",
        "",
        f"n = {record['n_episodes']} episodes, {record['n_seeds']} training seeds, deterministic. "
        f"d = ft - ref on the same seed and episodes.",
        "",
        f"**{key}: mean d = {d['mean']:+.4f}, SD = {d['sd']:.4f}, 95% t-interval "
        f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] (t = {record['t_crit']:.4f}, df = {record['n_seeds'] - 1}): "
        f"{record['label']}.**",
        "",
        f"## Per seed, {key}",
        f"Bootstrap: paired over episodes, B = {record['bootstrap']['B']}, rng seed "
        f"{record['bootstrap']['rng_seed']}, 95% percentile interval of d_i. Base = {s[key]['base']:.3f}.",
        "",
        "| seed | train seed | ft | ref | d | bootstrap 95% | ft - base | ref - base |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in record["per_seed"]:
        r, bs = row["metrics"][key], row["bootstrap"]
        lines.append(f"| {row['seed_index']} | {row['train_seed']} | {r['ft']:.3f} | {r['ref']:.3f} | "
                     f"{r['d']:+.3f} | [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}] | "
                     f"{r['ft_minus_base']:+.3f} | {r['ref_minus_base']:+.3f} |")
    lines += [
        "",
        "## All metrics (context, not deciding)",
        "Rates are shares of episodes; unsafe = cost > 0; hazard contact = hazard_steps > 0; vase touch = "
        "vase_contact_steps > 0; time to goal = mean length over successful episodes.",
        "",
        "| metric | base | ft mean | ref mean | mean d | SD of d | 95% t-interval | d per seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        dm = s[m]["d"]
        per = " / ".join(f"{row['metrics'][m]['d']:+.3g}" for row in record["per_seed"])
        lines.append(f"| {m} | {s[m]['base']:.4g} | {s[m]['ft_mean']:.4g} | {s[m]['ref_mean']:.4g} | "
                     f"{dm['mean']:+.4g} | {dm['sd']:.3g} | [{dm['ci_low']:+.4g}, {dm['ci_high']:+.4g}] | {per} |")
    path.write_text("\n".join(lines) + "\n")


def _plot(record: dict, path: Path) -> None:
    key, s = record["primary_key"], record["summary"]
    x = [row["seed_index"] for row in record["per_seed"]]
    ft = [row["metrics"][key]["ft"] for row in record["per_seed"]]
    ref = [row["metrics"][key]["ref"] for row in record["per_seed"]]
    d = [row["metrics"][key]["d"] for row in record["per_seed"]]
    lo = [row["metrics"][key]["d"] - row["bootstrap"]["ci_low"] for row in record["per_seed"]]
    hi = [row["bootstrap"]["ci_high"] - row["metrics"][key]["d"] for row in record["per_seed"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    for xi, a, b in zip(x, ft, ref):
        ax1.plot([xi, xi], [b, a], color="0.6", lw=1, zorder=1)
    ax1.scatter(x, ref, color="#2a78d6", label="ref (β = 0)", zorder=2)
    ax1.scatter(x, ft, color="#eb6834", label=f"ft (β = {record['beta']:.4g})", zorder=2)
    ax1.axhline(s[key]["base"], color="0.3", ls="--", lw=1, label="base policy")
    ax1.set_xticks(x)
    ax1.set_xlabel("training seed index")
    ax1.set_ylabel(key)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(f"{record['base']}, test episodes (n = {record['n_episodes']})", fontsize=10)

    dm = s[key]["d"]
    ax2.errorbar(x, d, yerr=[lo, hi], fmt="o", color="0.2", ms=4, capsize=3, lw=1,
                 label="d_i, bootstrap 95%")
    ax2.axhspan(dm["ci_low"], dm["ci_high"], color="#eb6834", alpha=0.2, label="mean d, 95% t-interval")
    ax2.axhline(dm["mean"], color="#eb6834", lw=1.5)
    ax2.axhline(0, color="0.3", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xlabel("training seed index")
    ax2.set_ylabel("d = ft − ref")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title(f"mean d = {dm['mean']:+.4f}: {record['label']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
