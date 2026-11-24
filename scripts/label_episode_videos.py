"""Draws labels on the unlabeled replays written by render_episode_videos.py and writes the shown videos.

Reads only the saved frames and per-step costs, so it gives the same videos on any machine. For each
(episode, training seed) in --raw-dir: one video per policy and a side-by-side of base, β = 0 and β fine-tune.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.video.sb3_video import label_frames, read_mp4, write_mp4


def panel_label(meta: dict) -> str:
    if meta["role"] == "base":
        return "base"
    return f"β = {meta['beta']:.2f} fine-tune (seed {meta['train_seed']})" if meta["role"] == "ft" \
        else f"β = 0 fine-tune (seed {meta['train_seed']})"


def load(json_path: Path):
    meta = json.loads(json_path.read_text())
    frames = read_mp4(json_path.with_suffix(".mp4"))
    if len(frames) != len(meta["steps"]) + 1:
        raise ValueError(f"{json_path}: {len(frames)} frames for {len(meta['steps'])} steps")
    return frames, [tuple(s) for s in meta["steps"]], meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=Path("results/video/raw"))
    p.add_argument("--out-dir", type=Path, help="default: parent of --raw-dir")
    args = p.parse_args()
    raw_dir = PROJECT_ROOT / args.raw_dir
    out_dir = PROJECT_ROOT / args.out_dir if args.out_dir else raw_dir.parent

    for ft_json in sorted(raw_dir.glob("*_ft_s*.json")):
        ft = json.loads(ft_json.read_text())
        stem, i = ft["stem"], ft["train_seed"]
        panels = [load(raw_dir / f"{stem}_{tag}.json") for tag in ["base", f"ref_s{i}", f"ft_s{i}"]]

        for frames, steps, meta in panels:
            label = panel_label(meta)
            path = out_dir / f"{stem}_{meta['tag']}.mp4"
            write_mp4(path, label_frames(frames, steps, label))
            shown = {k: v for k, v in meta.items() if k != "steps"}
            path.with_suffix(".json").write_text(json.dumps({**shown, "label": label}, indent=2))

        n = max(len(f) for f, _, _ in panels)
        side = [np.concatenate(row, axis=1)
                for row in zip(*(label_frames(f, s, panel_label(m), n) for f, s, m in panels))]
        path = out_dir / f"{stem}_s{i}_side_by_side.mp4"
        write_mp4(path, side)
        keys = ["exp_name", "run_dir", "model_sha256", "match"]
        meta = {"stem": stem, "episode_seed": panels[0][2]["episode_seed"], "train_seed": i,
                "panels": [{**{k: m[k] for k in keys}, "label": panel_label(m)} for *_, m in panels],
                "match": all(m["match"] for *_, m in panels)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"[label] wrote {path} (all replays match: {meta['match']})")


if __name__ == "__main__":
    main()
