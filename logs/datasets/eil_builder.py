from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np


LABEL_G = 0  # Good enough
LABEL_B = 1  # Bad
LABEL_I = 2  # Intervention


@dataclass
class BuilderConfig:
    alphaL: float = 0.2
    alphaE: float = 0.0

    # Output
    dataset_id: str = "eil_dataset"
    out_root: str = "logs/datasets/eil"
    seed: int = 0

    # Weighting
    downweight_contact: float = 0.25
    downweight_hazard_step: float = 0.5

    store_a_base_for_G: bool = True
    min_segment_len: int = 5

    override_threshold: float = 0.5 


def _segments(mask01: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return contiguous segments as (start, end, value), end exclusive."""
    if mask01.size == 0:
        return []
    m = mask01.astype(np.int32)
    segments: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(m[0])
    for i in range(1, len(m)):
        v = int(m[i])
        if v != cur:
            segments.append((start, i, cur))
            start = i
            cur = v
    segments.append((start, len(m), cur))
    return segments


def _split_learner_segment(start: int, end: int, alphaL: float, min_len: int) -> List[Tuple[int, int, int]]:
    """Split learner [start,end) into G then B tail (last alphaL fraction)."""
    L = end - start
    if L < min_len:
        return [(start, end, LABEL_G)]

    tail = int(np.ceil(alphaL * L))
    tail = int(np.clip(tail, 0, L))
    split = end - tail

    if tail == 0 or split <= start or split >= end:
        return [(start, end, LABEL_G)]

    return [(start, split, LABEL_G), (split, end, LABEL_B)]


def _split_expert_segment(start: int, end: int, alphaE: float, min_len: int) -> List[Tuple[int, int, int]]:
    """Expert [start,end) is I, optional first alphaE fraction as B."""
    L = end - start
    if L < min_len:
        return [(start, end, LABEL_I)]

    head = int(np.ceil(alphaE * L))
    head = int(np.clip(head, 0, L))
    split = start + head

    if head == 0 or split <= start:
        return [(start, end, LABEL_I)]
    if split >= end:
        return [(start, end, LABEL_B)]

    return [(start, split, LABEL_B), (split, end, LABEL_I)]


def _compute_weights(session: Dict[str, np.ndarray], cfg: BuilderConfig) -> np.ndarray:
    """Per-step weights: downweight contact/hazard steps."""
    T = session["obs"].shape[0]
    w = np.ones((T,), dtype=np.float32)

    if "hazard_in_contact" in session:
        hic = session["hazard_in_contact"].astype(np.float32)
        if hic.ndim == 0:
            hic = np.full((T,), float(hic), dtype=np.float32)
        w *= np.where(hic > 0.0, cfg.downweight_contact, 1.0)

    if "contact_cost" in session:
        cc = session["contact_cost"].astype(np.float32)
        if cc.ndim == 0:
            cc = np.full((T,), float(cc), dtype=np.float32)
        w *= np.where(cc > 0.0, cfg.downweight_contact, 1.0)

    if "hazard_cost_step" in session:
        hcs = session["hazard_cost_step"].astype(np.float32)
        if hcs.ndim == 0:
            hcs = np.full((T,), float(hcs), dtype=np.float32)
        w *= np.where(hcs > 0.0, cfg.downweight_hazard_step, 1.0)

    return w


def build_eil_dataset(session_files: List[str | Path], cfg: BuilderConfig, env_name: str) -> Path:
    """
    Build a unified EIL dataset from session.npz files.
    """
    rng = np.random.default_rng(cfg.seed)

    session_paths = [Path(p) for p in session_files]
    session_paths = [p for p in session_paths if p.exists()]
    if not session_paths:
        raise FileNotFoundError("No session files found.")

    # Shuffle sessions so final dataset order is not always the same
    order = rng.permutation(len(session_paths)).tolist()
    session_paths = [session_paths[i] for i in order]

    out_dir = Path(cfg.out_root) / env_name / cfg.dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "env_name": env_name,
        "dataset_id": cfg.dataset_id,
        "alphaL": cfg.alphaL,
        "alphaE": cfg.alphaE,
        "seed": cfg.seed,
        "downweight_contact": cfg.downweight_contact,
        "downweight_hazard_step": cfg.downweight_hazard_step,
        "override_threshold": cfg.override_threshold,
        "label_map": {"G": LABEL_G, "B": LABEL_B, "I": LABEL_I},
        "data_file": "data.npz",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    index = {
        "sessions": [str(p) for p in session_paths],
        "stats": {
            "steps": 0,
            "label_counts": {"G": 0, "B": 0, "I": 0},
            "num_sessions": len(session_paths),
        },
        "per_session": [],
        "data_file": "data.npz",
    }

    # Compute total_T
    total_T = 0
    obs_dim = None
    act_dim = None

    for p in session_paths:
        data = dict(np.load(p, allow_pickle=False))
        obs = data["obs"]
        a_exec = data["action_exec"]
        T = int(obs.shape[0])

        if obs_dim is None:
            obs_dim = int(obs.shape[1])

        if act_dim is None:
            act_dim = int(a_exec.shape[1])

        total_T += T

    obs_all = np.empty((total_T, obs_dim), dtype=np.float32)
    act_all = np.empty((total_T, act_dim), dtype=np.float32)
    a_base_all = np.empty((total_T, act_dim), dtype=np.float32)
    a_exec_all = np.empty((total_T, act_dim), dtype=np.float32)
    lab_all = np.empty((total_T,), dtype=np.int64)
    wgt_all = np.empty((total_T,), dtype=np.float32)
    sid_all = np.empty((total_T,), dtype=np.int32)
    t_all = np.empty((total_T,), dtype=np.int32)

    label_counts = {"G": 0, "B": 0, "I": 0}
    write_pos = 0

    for s_i, p in enumerate(session_paths):
        data = dict(np.load(p, allow_pickle=False))

        obs = data["obs"].astype(np.float32)
        a_base = data["action_base"].astype(np.float32)
        a_exec = data["action_exec"].astype(np.float32)

        T = obs.shape[0]
        override = data["human_override"]
        override01 = (override.astype(np.float32) > cfg.override_threshold).astype(np.int32)

        segs = _segments(override01)
        w = _compute_weights(data, cfg)

        labeled_intervals: List[Tuple[int, int, int]] = []
        takeover_events = 0

        if all(v == 0 for (_, _, v) in segs):
            labeled_intervals.append((0, T, LABEL_G))
        else:
            i = 0
            while i < len(segs):
                s, e, v = segs[i]
                if v == 0:
                    if i + 1 < len(segs) and segs[i + 1][2] == 1:
                        labeled_intervals.extend(_split_learner_segment(s, e, cfg.alphaL, cfg.min_segment_len))
                        takeover_events += 1
                    else:
                        labeled_intervals.append((s, e, LABEL_G))
                else:
                    labeled_intervals.extend(_split_expert_segment(s, e, cfg.alphaE, cfg.min_segment_len))
                i += 1

        labels = np.empty((T,), dtype=np.int64)
        actions = np.empty_like(a_exec, dtype=np.float32)

        labels.fill(LABEL_G)
        actions[:] = a_base if cfg.store_a_base_for_G else a_exec

        for s, e, lab in labeled_intervals:
            labels[s:e] = lab
            if lab == LABEL_I:
                actions[s:e] = a_exec[s:e]
            else:
                actions[s:e] = a_base[s:e] if cfg.store_a_base_for_G else a_exec[s:e]

        g = int(np.sum(labels == LABEL_G))
        b = int(np.sum(labels == LABEL_B))
        ii = int(np.sum(labels == LABEL_I))
        label_counts["G"] += g
        label_counts["B"] += b
        label_counts["I"] += ii

        per_sess = {
            "path": str(p),
            "T": int(T),
            "override_mean": float(np.mean(override01)),
            "takeover_events": int(takeover_events),
            "label_counts": {"G": g, "B": b, "I": ii},
            "hazard_entries": int(np.sum(data["hazard_entry"])) if "hazard_entry" in data else None,
            "cost_mean": float(np.mean(data["cost"])) if "cost" in data else None,
            "cost_max": float(np.max(data["cost"])) if "cost" in data else None,
        }
        index["per_session"].append(per_sess)

        sl = slice(write_pos, write_pos + T)
        obs_all[sl] = obs
        act_all[sl] = actions
        a_base_all[sl] = a_base
        a_exec_all[sl] = a_exec
        lab_all[sl] = labels
        wgt_all[sl] = w.astype(np.float32)
        sid_all[sl] = np.full((T,), s_i, dtype=np.int32)
        t_all[sl] = np.arange(T, dtype=np.int32)

        write_pos += T

    if write_pos != total_T:
        raise RuntimeError(f"Filled {write_pos} steps but expected {total_T}.")

    np.savez_compressed(
        out_dir / "data.npz",
        obs=obs_all,
        action=act_all,            
        action_base=a_base_all,    
        action_exec=a_exec_all,    
        label=lab_all,
        weight=wgt_all,
        session_id=sid_all,
        t=t_all,
    )

    index["stats"]["steps"] = int(total_T)
    index["stats"]["label_counts"] = label_counts
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    return out_dir