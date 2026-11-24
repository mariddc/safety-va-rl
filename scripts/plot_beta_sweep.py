import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sweeps.finetune_sweep import OBJECTIVE_KEY
from src.utils.run_meta import sha256_file

REF, TRIAL, CONFIRM, GP = "#2a78d6", "#eb6834", "#1baf7a", "0.45"
MARKER = {"startup": "s", "GP": "o", "copied": "s"}
CONFIRM_SHIFT = 1.07  # confirm runs reuse trial betas; shift right so the points don't overlap


def load_inputs(base: str, sweeps: Path) -> dict:
    top3_path = sweeps / f"{base}_pass1_select" / "top3.json"
    top3 = json.loads(top3_path.read_text())
    journal = Path(top3["journal"])
    if sha256_file(journal) != top3["journal_sha256"]:
        raise ValueError(f"{journal} changed since {top3_path} was fitted")
    # load_study only reads the journal; create_study would append to it
    study = optuna.load_study(study_name=top3["study_name"], storage=JournalStorage(JournalFileBackend(str(journal))))
    gp_at = {t["number"]: t for t in top3["trials"]}
    trials = []
    for t in study.get_trials(states=[TrialState.COMPLETE]):
        source = ("copied" if t.user_attrs.get("copied_from_trial") is not None
                  else "startup" if "fixed_params" in t.system_attrs else "GP")
        trials.append({"number": t.number, "source": source, "beta": t.params["beta"], "value": t.value,
                       "seeds": t.user_attrs["seeds"], "runs": t.user_attrs["runs"], **gp_at[t.number]})
    confirm = [json.loads((sweeps / f"{base}_pass1_confirm_b{b!r}_s0-2" / "runs.json").read_text())
               for b in sorted(top3["betas"])]
    ref = json.loads((sweeps / f"{base}_pass1_ref_b0" / "runs.json").read_text())
    chosen = json.loads((sweeps / f"{base}_pass1_select" / "chosen_beta.json").read_text())
    return {"top3": top3, "trials": trials, "confirm": confirm, "ref": ref, "chosen": chosen}


def write_csv(data: dict, key: str, path: Path) -> None:
    rows = [{"kind": "gp_grid", "beta": b, "gp_mean": m, "gp_sd": s}
            for b, m, s in zip(*(data["top3"]["grid"][k] for k in ("beta", "gp_mean", "gp_sd")))]

    def seed_rows(kind, runs, seeds, **extra):
        return [{"kind": kind, **extra, "seed_index": r["seed_index"], "train_seed": seeds[r["seed_index"]],
                 "value": r[key], "penalty_ratio": r["tb"]["q/penalty_ratio"]["mean"],
                 "active_frac": r["tb"]["q/active_frac"]["mean"]} for r in runs]

    for t in data["trials"]:
        rows.append({"kind": "trial_mean", "trial": t["number"], "source": t["source"], "beta": t["beta"],
                     "value": t["value"], "gp_mean": t["gp_mean"], "gp_sd": t["gp_sd"]})
        rows += seed_rows("trial", t["runs"], t["seeds"], trial=t["number"], source=t["source"], beta=t["beta"])
    for c in data["confirm"] + [data["ref"]]:
        kind = "confirm" if c["beta"] else "ref_beta0"
        rows.append({"kind": f"{kind}_mean", "beta": c["beta"], "value": c["objective"]})
        rows += seed_rows(kind, c["runs"], c["seeds"], beta=c["beta"])
    rows.append({"kind": "chosen", "beta": data["chosen"]["beta"]})
    fields = ["kind", "trial", "source", "beta", "seed_index", "train_seed", "value", "penalty_ratio",
              "active_frac", "gp_mean", "gp_sd"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot(base: str, data: dict, key: str, path: Path) -> None:
    top3, trials, confirm, ref = data["top3"], data["trials"], data["confirm"], data["ref"]
    n_eval = trials[0]["runs"][0]["n_episodes"]
    lo, hi = top3["beta_range"]
    x_ref = lo / 2.2  # beta = 0 has no place on a log axis; drawn left of the range

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
    ax, ax_pr, ax_af = axes
    g = {k: np.array(v) for k, v in top3["grid"].items()}
    ax.fill_between(g["beta"], g["gp_mean"] - 2 * g["gp_sd"], g["gp_mean"] + 2 * g["gp_sd"],
                    color=GP, alpha=0.18, lw=0, label="GP posterior mean ± 2 sd")
    ax.plot(g["beta"], g["gp_mean"], color=GP, lw=1.5)

    for source in ("startup", "copied", "GP"):
        ts = [t for t in trials if t["source"] == source]
        if not ts:
            continue
        name = {"startup": "trial (startup)", "copied": "trial (copied from first study)", "GP": "trial (GP)"}
        ax.scatter([t["beta"] for t in ts], [t["value"] for t in ts], s=55, marker=MARKER[source], zorder=3,
                   facecolor="white" if source == "copied" else TRIAL, edgecolor=TRIAL, lw=1.5,
                   label=f"{name[source]}, 3-seed mean")
    for t in trials:
        vals = [r[key] for r in t["runs"]]
        ax.scatter([t["beta"]] * len(vals), vals, s=12, color=TRIAL, alpha=0.5, lw=0, zorder=2)
        for a, r in zip((ax_pr, ax_af), ("q/penalty_ratio", "q/active_frac")):
            a.scatter([t["beta"]] * 3, [x["tb"][r]["mean"] for x in t["runs"]], s=14, color=TRIAL, lw=0)

    for i, c in enumerate(confirm):
        xb = c["beta"] * CONFIRM_SHIFT
        ax.scatter([xb], [c["objective"]], s=60, marker="D", color=CONFIRM, edgecolor="white", lw=1, zorder=4,
                   label="confirm (top 3), 3-seed mean" if i == 0 else None)
        ax.scatter([xb] * 3, [r[key] for r in c["runs"]], s=12, color=CONFIRM, alpha=0.6, lw=0, zorder=2)
        for a, r in zip((ax_pr, ax_af), ("q/penalty_ratio", "q/active_frac")):
            a.scatter([xb] * 3, [x["tb"][r]["mean"] for x in c["runs"]], s=14, color=CONFIRM, lw=0)

    ax.scatter([], [], s=12, color="0.5", lw=0, label="small dots: one per training seed")
    ax.axhline(ref["objective"], color=REF, ls="--", lw=1.2, label="β = 0 on the sweep seeds, 3-seed mean")
    ax.scatter([x_ref] * 3, [r[key] for r in ref["runs"]], s=14, color=REF, lw=0, zorder=2)
    ax_af.scatter([x_ref] * 3, [r["tb"]["q/active_frac"]["mean"] for r in ref["runs"]], s=14, color=REF, lw=0)
    beta_c = data["chosen"]["beta"]
    for a in axes:
        a.axvline(beta_c, color="0.15", ls=":", lw=1.2, label=f"chosen β = {beta_c:.4g}" if a is ax else None)
        a.axvline(lo, color="0.85", lw=0.8)
        a.grid(axis="y", color="0.92", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)

    ax.set_xscale("log")
    ax.set_xlim(x_ref / 1.5, hi * 1.5)
    ax.set_ylabel(f"{key}\n(mean over 3 training seeds)")
    ax.set_title(f"{base}: β sweep — selection episodes (n = {n_eval}), not the test", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax_pr.set_yscale("log")
    ax_pr.set_ylabel("q/penalty_ratio\n(rollout mean)")
    ax_af.set_ylabel("q/active_frac\n(rollout mean)")
    ax_af.set_xlabel("β (log scale); points left of the range: β = 0")
    ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300) if lo <= t <= hi]
    ax_af.set_xticks([x_ref] + ticks, ["0"] + [f"{t:g}" for t in ticks])
    ax_af.minorticks_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one base's beta sweep: GP fit, trials, confirm runs, beta = 0.")
    parser.add_argument("base", choices=list(OBJECTIVE_KEY))
    parser.add_argument("--sweeps", type=Path, default=Path("runs/sweeps"))
    parser.add_argument("--out-dir", type=Path, help="default: results/final/<base>")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    key = OBJECTIVE_KEY[args.base]
    data = load_inputs(args.base, args.sweeps)
    out_dir = args.out_dir or Path("results/final") / args.base
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(data, key, out_dir / "beta_sweep.csv")
    plot(args.base, data, key, out_dir / "beta_sweep.png")
    print(f"[plot_beta_sweep] {args.base}: {len(data['trials'])} trials, chosen beta {data['chosen']['beta']:.4g} "
          f"-> {out_dir}/beta_sweep.png")


if __name__ == "__main__":
    main()
