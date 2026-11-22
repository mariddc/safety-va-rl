import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sweeps.beta_select import fit_top3, select_beta
from src.sweeps.finetune_sweep import OBJECTIVE_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose beta: top 3 by GP posterior mean, then best confirm mean.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit", help="Fit the GP to a study's completed trials and write top3.json")
    fit.add_argument("base", choices=list(OBJECTIVE_KEY))
    fit.add_argument("--journal", type=Path, required=True, help="Journal of the study beta_<base> (only read)")
    fit.add_argument("--out-dir", type=Path, required=True)
    sel = sub.add_parser("select", help="Choose beta from the confirm runs of the top 3 and write chosen_beta.json")
    sel.add_argument("base", choices=list(OBJECTIVE_KEY))
    sel.add_argument("--top3", type=Path, required=True, help="top3.json written by fit")
    sel.add_argument("--runs", type=Path, nargs=3, required=True,
                     help="runs.json of each top-3 beta on the confirm seeds")
    sel.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "fit":
        fit_top3(args.base, args.journal, args.out_dir)
    else:
        select_beta(args.base, args.top3, args.runs, args.out_dir)


if __name__ == "__main__":
    main()
