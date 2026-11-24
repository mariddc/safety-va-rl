"""Rank recorded test episodes as candidates for an illustrative base / ref / ft video.

Reads only the episodes.csv files listed in a final-test index; runs no episodes.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

COLS = ["success", "cost", "hazard_steps", "vase_contact_steps", "length"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("index", help="runs/final_test/<base>_pass1_test_n1000.json")
    p.add_argument("--key", required=True, help="safe-success column, e.g. safe_success_strict")
    p.add_argument("--n-seeds", type=int, default=6)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    index = json.loads((PROJECT_ROOT / args.index).read_text())
    base, beta = index["base"], index["beta"]
    dfs = {name: pd.read_csv(PROJECT_ROOT / e["eval_dir"] / "episodes.csv") for name, e in index["evals"].items()}
    seeds = dfs[base]["seed"].tolist()
    for name, df in dfs.items():
        if df["seed"].tolist() != seeds:
            raise ValueError(f"{name}: episode seeds differ from {base}")

    ft = [dfs[f"{base}_v3_final_b{beta}_s{i}"] for i in range(args.n_seeds)]
    ref = [dfs[f"{base}_v3_final_b0.0_s{i}"] for i in range(args.n_seeds)]
    k = args.key
    lines = [f"# Video candidates: {base}", "",
             f"Source: `{args.index}` (recorded test episodes; nothing re-run). Key: `{k}`. "
             f"ft = β {beta:.2f}, ref = β 0, i = training seed.", "",
             "## Per training seed: episodes where one is safe-successful and the other is not", "",
             "| i | ft safe, ref not | ref safe, ft not | of first: both reach goal | of second: both reach goal |",
             "|---|---|---|---|---|"]
    for i in range(args.n_seeds):
        a = (ft[i][k] == 1) & (ref[i][k] == 0)
        b = (ref[i][k] == 1) & (ft[i][k] == 0)
        both = (ft[i]["success"] == 1) & (ref[i]["success"] == 1)
        lines.append(f"| {i} | {a.sum()} | {b.sum()} | {(a & both).sum()} | {(b & both).sum()} |")

    n_ft = sum(f[k] for f in ft)
    n_ref = sum(r[k] for r in ref)
    score = n_ft - n_ref
    lines += ["", f"Distribution of #ft − #ref over the {len(seeds)} episodes (#ft, #ref = number of the "
              f"{args.n_seeds} seeds that are safe-successful):", "",
              "| #ft − #ref | " + " | ".join(str(v) for v in range(-args.n_seeds, args.n_seeds + 1)) + " |",
              "|---|" + "---|" * (2 * args.n_seeds + 1),
              "| episodes | " + " | ".join(str((score == v).sum()) for v in range(-args.n_seeds, args.n_seeds + 1)) + " |"]

    # A seed i "contrasts" an episode when ft_i is safe-successful and ref_i reaches the goal unsafely;
    # the best i is the one where ref_i's cost is highest (the clearest violation to watch).
    rows = []
    for j, s in enumerate(seeds):
        contrast = [i for i in range(args.n_seeds)
                    if ft[i][k][j] == 1 and ref[i][k][j] == 0 and ref[i]["success"][j] == 1]
        if not contrast:
            continue
        best = max(contrast, key=lambda i: (ref[i]["cost"][j], -i))
        rows.append(dict(j=j, seed=s, score=score[j], n_ft=n_ft[j], n_ref=n_ref[j], base_safe=dfs[base][k][j],
                         contrast=contrast, best=best))
    rows.sort(key=lambda r: (-r["score"], r["base_safe"], -len(r["contrast"]), r["seed"]))
    top = rows[:args.top]

    lines += ["", f"## Top {len(top)} candidates", "",
              "Rank: #ft − #ref (desc), then base not safe-successful first, then number of contrast seeds. "
              "Only episodes with at least one contrast seed (ft_i safe-successful, ref_i reaches the goal but "
              f"is not safe) are listed ({len(rows)} episodes qualify). Best i = contrast seed with the highest ref_i cost.",
              "", f"| rank | episode seed | #ft | #ref | base {k} | ft by i | ref by i | contrast i | best i |",
              "|---|---|---|---|---|---|---|---|---|"]
    for n, r in enumerate(top, 1):
        j = r["j"]
        lines.append(f"| {n} | {r['seed']} | {r['n_ft']} | {r['n_ref']} | {r['base_safe']} | "
                     f"{''.join(str(f[k][j]) for f in ft)} | {''.join(str(x[k][j]) for x in ref)} | "
                     f"{','.join(map(str, r['contrast']))} | {r['best']} |")

    lines += ["", "## Outcomes for base, ref_i, ft_i at the best i", "",
              f"| episode seed | policy | {k} | " + " | ".join(COLS) + " |", "|---|---|---|" + "---|" * len(COLS)]
    for r in top:
        j, i = r["j"], r["best"]
        for label, df in [("base", dfs[base]), (f"ref_{i}", ref[i]), (f"ft_{i}", ft[i])]:
            vals = [f"{df[c][j]:g}" for c in COLS]
            lines.append(f"| {r['seed']} | {label} | {df[k][j]} | " + " | ".join(vals) + " |")

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[video_candidates] wrote {out}")


if __name__ == "__main__":
    main()
