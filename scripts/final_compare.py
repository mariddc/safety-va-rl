import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.final_compare import compare
from src.sweeps.finetune_sweep import OBJECTIVE_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired ft vs ref comparison of one base on its test outputs.")
    parser.add_argument("base", choices=list(OBJECTIVE_KEY))
    parser.add_argument("--index", type=Path, help="default: runs/final_test/<base>_pass1_test_n1000.json")
    parser.add_argument("--out-dir", type=Path, help="default: results/final/<base>")
    args = parser.parse_args()
    index = args.index or Path("runs/final_test") / f"{args.base}_pass1_test_n1000.json"
    compare(args.base, index, args.out_dir or Path("results/final") / args.base)


if __name__ == "__main__":
    main()
