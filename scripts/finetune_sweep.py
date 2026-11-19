import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sweeps.finetune_sweep import OBJECTIVE_KEY, run_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one trial of the beta sweep of a base, then exit.")
    parser.add_argument("base", choices=list(OBJECTIVE_KEY), help="Base experiment whose beta is swept")
    parser.add_argument("--run-pass", required=True, help="Round tag in the study path, run dir names and meta.json")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run with uncommitted code changes (recorded as dirty_allowed=true; not for reported runs)")
    parser.add_argument("--n-trials", type=int, default=12, help="Exit without running once this many trials are complete")
    parser.add_argument("--n-eval", type=int, default=500, help="Evaluate on the first n selection seeds")
    parser.add_argument("--serial", action="store_true", help="Run the seed fine-tunes one after another")
    args = parser.parse_args()
    run_trial(args.base, run_pass=args.run_pass, allow_dirty=args.allow_dirty, n_trials=args.n_trials,
              n_eval=args.n_eval, serial=args.serial)


if __name__ == "__main__":
    main()
