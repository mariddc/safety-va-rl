"""Replay one recorded test episode for base, β = 0 fine-tune and β fine-tune of one training seed.

Illustration only: the episode seed is given explicitly and must be a row of each policy's recorded
episodes.csv; the replay is checked against that row. The seed pool is never loaded and no eval dir is written.
Writes unlabeled frames (lossless mp4) and per-step costs to <out-dir>; label_episode_videos.py draws the
labels from them on any machine. The replay itself must run on the test host.
"""

import argparse
import csv
import json
import os
import platform
import socket
import sys
from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.video.sb3_video import replay_episode, write_mp4
from src.experiments.registry import get_experiment
from src.utils.run_meta import git_info, sha256_file, utc_now

CHECKED = ["success", "length", "cost", "hazard_steps", "vase_contact_steps"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("index", help="runs/final_test/<base>_pass1_test_n1000.json")
    p.add_argument("--train-seed", type=int, required=True)
    p.add_argument("--episode-seed", type=int, required=True)
    p.add_argument("--out-dir", default="results/video/raw")
    args = p.parse_args()

    index = json.loads((PROJECT_ROOT / args.index).read_text())
    base, beta, i = index["base"], index["beta"], args.train_seed
    policies = [
        ("base", "base", None, base),
        (f"ref_s{i}", "ref", 0.0, f"{base}_v3_final_b0.0_s{i}"),
        (f"ft_s{i}", "ft", beta, f"{base}_v3_final_b{beta}_s{i}"),
    ]
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{base}_ep{args.episode_seed}"
    common = {"stem": stem, "episode_seed": args.episode_seed, "train_seed": i, "index": args.index,
              "test_hostname": index["evals"][base]["hostname"],
              "render": {"backend": os.environ.get("MUJOCO_GL"), "camera": "fixedfar",
                         "size": [256, 256], "fps": 30},
              "machine": {"hostname": socket.gethostname(), "platform": platform.platform(),
                          "processor": platform.processor(), "mujoco": mujoco.__version__},
              "git": git_info(), "time": utc_now()}

    all_match = True
    for tag, role, policy_beta, name in policies:
        rec = index["evals"][name]
        eval_dir = PROJECT_ROOT / rec["eval_dir"]
        eval_meta = json.loads((eval_dir / "meta.json").read_text())
        model_path = PROJECT_ROOT / eval_meta["model_path"]
        sha = sha256_file(model_path)
        if sha != eval_meta["model_sha256"]:
            raise ValueError(f"{name}: model.zip sha256 differs from the one evaluated")
        with open(eval_dir / "episodes.csv") as f:
            rows = [r for r in csv.DictReader(f) if int(r["seed"]) == args.episode_seed]
        if len(rows) != 1:
            raise ValueError(f"{name}: episode seed {args.episode_seed} is not a recorded episode")
        recorded = {k: float(rows[0][k]) for k in CHECKED}

        frames, steps, outcome = replay_episode(get_experiment(name), model_path, args.episode_seed)
        match = all(float(outcome[k]) == recorded[k] for k in CHECKED)
        print(f"[video] {name} ep {args.episode_seed}: replay {outcome} recorded {recorded} match={match}")

        path = out_dir / f"{stem}_{tag}.mp4"
        write_mp4(path, frames, lossless=True)
        meta = {**common, "tag": tag, "role": role, "beta": policy_beta, "exp_name": name,
                "run_dir": eval_meta["run_dir"], "model_sha256": sha, "episodes_csv": rec["eval_dir"] + "/episodes.csv",
                "replay": outcome, "recorded": recorded, "match": match,
                "steps": [list(s) for s in steps]}  # (cost, in hazard, vase contact) per step; frames = steps + 1
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        all_match &= match
    print(f"[video] wrote {out_dir}/{stem}_*.mp4 (all replays match: {all_match})")


if __name__ == "__main__":
    main()
