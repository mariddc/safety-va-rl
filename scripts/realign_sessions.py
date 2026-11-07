"""
One-off fix for sessions recorded before the logger bug was fixed (to-do item 9).

Those sessions saved (s_{t+1}, a_t) instead of (s_t, a_t). For each one:
- pair each action with the observation of the row before it, inside its episode;
- drop the first row of each episode, whose true state was never saved;
- keep the original as session_misaligned.npz, and write the result as session.npz
  with obs_aligned=True.

Then point the existing dataset index.json files at session_misaligned.npz, since that
is the data those datasets were built from.

Safe to rerun, and to rerun after an interruption. Nothing is deleted.

    python scripts/realign_sessions.py [--root DIR]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logs.datasets.eil_builder import _episode_bounds

SESSION = "session.npz"
MISALIGNED = "session_misaligned.npz"
TMP = "session.tmp.npz"

COLUMNS = ("sessions", "old_steps", "episodes", "one_step_episodes", "new_steps")


def load_session(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def has_marker(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as z:
        return "obs_aligned" in z.files and bool(z["obs_aligned"])


def realign(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the session with each action paired with the state it was chosen from."""
    T = data["obs"].shape[0]
    # A key that isn't per-step would be cut wrongly. This also stops a marked file
    # being shifted twice, because obs_aligned is a scalar.
    bad = [k for k, v in data.items() if v.ndim == 0 or v.shape[0] != T]
    if bad:
        raise ValueError(f"keys without length T={T}: {bad}")

    keep = np.ones(T, dtype=bool)
    keep[[s for s, _ in _episode_bounds(data, T)]] = False
    new = {k: v[keep] for k, v in data.items()}
    new["obs"] = data["obs"][np.flatnonzero(keep) - 1]
    new["obs_aligned"] = np.array(True)
    return new


def write_temp(arrays: dict[str, np.ndarray], tmp: Path) -> None:
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())


def check_temp(tmp: Path, expected: dict[str, np.ndarray], T_new: int) -> None:
    """Raise unless the file on disk holds exactly the expected arrays."""
    got = load_session(tmp)
    if set(got) != set(expected):
        raise ValueError(f"{tmp}: keys {sorted(got)} != {sorted(expected)}")
    for k, want in expected.items():
        have = got[k]
        if have.dtype != want.dtype or have.shape != want.shape or have.tobytes() != want.tobytes():
            raise ValueError(f"{tmp}: key {k!r} differs from what was written")
    if got["obs"].shape[0] != T_new:
        raise ValueError(f"{tmp}: {got['obs'].shape[0]} steps, expected {T_new}")


def classify(session_dir: Path) -> str:
    session = session_dir / SESSION
    misaligned = session_dir / MISALIGNED

    if session.exists():
        if has_marker(session):
            return "skip"
        return "conflict" if misaligned.exists() else "normal"
    return "resume" if misaligned.exists() else "no_data"


def realign_folder(session_dir: Path, state: str) -> dict[str, int]:
    """Realign one folder. `state` is "normal", or "resume" if only the original is left."""
    session = session_dir / SESSION
    misaligned = session_dir / MISALIGNED
    tmp = session_dir / TMP

    data = load_session(session if state == "normal" else misaligned)
    T_old = data["obs"].shape[0]
    bounds = _episode_bounds(data, T_old)
    new = realign(data)
    T_new = T_old - len(bounds)

    write_temp(new, tmp)
    check_temp(tmp, new, T_new)

    if state == "normal":
        if misaligned.exists():  # os.rename would overwrite it without a word
            raise FileExistsError(misaligned)
        os.rename(session, misaligned)
    os.rename(tmp, session)

    return {
        "sessions": 1,
        "old_steps": T_old,
        "episodes": len(bounds),
        "one_step_episodes": sum(e - s == 1 for s, e in bounds),
        "new_steps": T_new,
    }


def print_totals(totals: dict[str, dict[str, int]]) -> None:
    print(f"{'experiment':<34}" + "".join(f"{c:>19}" for c in COLUMNS))
    for exp in sorted(totals):
        print(f"{exp:<34}" + "".join(f"{totals[exp][c]:>19,}" for c in COLUMNS))
    if totals:
        print(f"{'TOTAL':<34}" + "".join(f"{sum(t[c] for t in totals.values()):>19,}" for c in COLUMNS))


def realign_all(root: Path) -> int:
    base = root / "logs" / "interventions"
    folders = sorted(p for p in base.rglob("session_*") if p.is_dir())
    states = {p: classify(p) for p in folders}

    # Decided before touching anything, so a conflict leaves the whole tree as it was
    conflicts = [p for p, s in states.items() if s == "conflict"]
    no_data = [p for p, s in states.items() if s == "no_data"]
    skipped = sum(s == "skip" for s in states.values())

    if conflicts:
        print(f"[realign] CONFLICT, nothing changed. Unmarked {SESSION} beside {MISALIGNED}:")
        for p in conflicts:
            print(f"  {p.relative_to(root)}")
        return 1

    totals: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(COLUMNS, 0))
    realigned = 0
    for p, state in states.items():
        if state not in ("normal", "resume"):
            continue
        stats = realign_folder(p, state)
        realigned += 1
        exp = totals[p.parent.parent.name]
        for c in COLUMNS:
            exp[c] += stats[c]

    print_totals(totals)
    print(f"[realign] {len(folders)} folders: {realigned} realigned, {skipped} skipped, {len(no_data)} without data")
    for p in no_data:
        print(f"  no data: {p.relative_to(root)}")
    return 0


def mapped_entry(entry: str, root: Path) -> str | None:
    """The repo-relative session_misaligned.npz for an index entry, or None if there is none."""
    if not entry.startswith("/") and entry.endswith(f"/{MISALIGNED}") and (root / entry).exists():
        return entry  # already mapped by an earlier run

    _, sep, rel = entry.partition("/safety-va-rl/")
    if not sep:
        return None
    mapped = rel.replace(f"/{SESSION}", f"/{MISALIGNED}")
    return mapped if (root / mapped).exists() else None


def update_index(path: Path, root: Path) -> tuple[bool, list[str]]:
    """Map one index.json. Returns whether it changed, and the entries left as they were."""
    raw = path.read_bytes().decode("utf-8")
    doc = json.loads(raw)

    entries = list(doc["sessions"]) + [s["path"] for s in doc["per_session"]]
    mapping, unresolved = {}, []
    for entry in dict.fromkeys(entries):
        target = mapped_entry(entry, root)
        if target is None:
            unresolved.append(entry)
        elif target != entry:
            mapping[entry] = target

    if not mapping:
        return False, unresolved

    # Raw text, so nothing but the paths can change
    new_raw = raw
    for old, new in mapping.items():
        new_raw = new_raw.replace(json.dumps(old), json.dumps(new))

    expected = json.loads(raw)
    expected["sessions"] = [mapping.get(p, p) for p in expected["sessions"]]
    for s in expected["per_session"]:
        s["path"] = mapping.get(s["path"], s["path"])
    if json.loads(new_raw) != expected:
        raise ValueError(f"{path}: text replacement changed more than the paths")

    tmp = path.with_name("index.json.tmp")
    tmp.write_bytes(new_raw.encode("utf-8"))
    os.replace(tmp, path)
    return True, unresolved


def update_indexes(root: Path) -> None:
    paths = sorted((root / "logs" / "datasets" / "eil").rglob("index.json"))
    changed = 0
    for path in paths:
        did_change, unresolved = update_index(path, root)
        changed += did_change
        for entry in unresolved:
            print(f"  unresolved, left as is: {entry}\n    in {path.relative_to(root)}")
    print(f"[index] {len(paths)} files: {changed} changed, {len(paths) - changed} unchanged")


def main() -> int:
    parser = argparse.ArgumentParser(description="Realign sessions saved with the observation one step late.")
    parser.add_argument("--root", type=Path, default=ROOT, help="repo root (a copy, for a rehearsal)")
    root = parser.parse_args().root.resolve()

    code = realign_all(root)
    if code != 0:
        return code
    update_indexes(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
