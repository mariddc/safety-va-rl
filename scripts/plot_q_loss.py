import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"L_total": "#2a78d6", "L_G": "#eb6834", "L_B": "#1baf7a", "L_I": "#eda100"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Q training losses; writes q_loss.png next to each log.")
    parser.add_argument("q_train_log", nargs="+", help="Path(s) to q_train_log.json")
    args = parser.parse_args()

    for log_path in map(Path, args.q_train_log):
        out = log_path.parent / "q_loss.png"
        if out.exists():
            raise FileExistsError(out)
        log = json.loads(log_path.read_text())

        fig, ax = plt.subplots(figsize=(7, 4))
        for key, color in COLORS.items():
            ax.plot(log["steps"], log[key], label=key, color=color, linewidth=2)
        # Losses can be exactly 0, so a plain log scale would drop points
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("/".join(log_path.parent.parts[-3:]), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot_q_loss] {out}")


if __name__ == "__main__":
    main()
