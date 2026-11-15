"""
Reader for the committed frozen-hash file `configs/frozen.json` (D030). The human writes it at the Q freeze (1.4):

    {
      "q":           {"<base_exp>": {"path": "<repo-relative q_theta.pt>", "sha256": "<hex>"}},
      "base_models": {"<base_exp>": {"path": "<repo-relative model.zip>", "sha256": "<hex>"}}
    }

Until the file exists the Q hash is only recorded, not checked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FROZEN_FILE = Path(__file__).resolve().parents[2] / "configs" / "frozen.json"

NO_FROZEN_FILE = "no_frozen_file"
MATCHED = "matched"


def load_frozen(path: Path | None = None) -> dict[str, Any] | None:
    """Parsed frozen file, or None if it does not exist yet."""
    path = FROZEN_FILE if path is None else Path(path)
    if not path.exists():
        return None
    frozen = json.loads(path.read_text())
    if not isinstance(frozen, dict) or not isinstance(frozen.get("q"), dict):
        raise ValueError(f"{path}: expected a JSON object with a 'q' mapping of base_exp -> {{path, sha256}}.")
    return frozen


def check_frozen_q(base_exp: str, q_sha256: str, *, path: Path | None = None) -> str:
    """
    "no_frozen_file" before the freeze; "matched" if the Q hash equals the frozen one.
    A missing entry or a different hash is an error: the fine-tune would not use the frozen Q.
    """
    path = FROZEN_FILE if path is None else Path(path)
    frozen = load_frozen(path)
    if frozen is None:
        return NO_FROZEN_FILE

    entry = frozen["q"].get(base_exp)
    if not isinstance(entry, dict) or "sha256" not in entry:
        raise ValueError(
            f"{path}: no frozen Q entry for base {base_exp!r}. "
            f"Add q.{base_exp}.sha256 to the frozen file, or fine-tune a base that has one."
        )
    if entry["sha256"] != q_sha256:
        raise ValueError(
            f"{path}: Q for base {base_exp!r} has sha256 {q_sha256}, frozen value is {entry['sha256']} "
            f"(frozen path {entry.get('path')}). Point q_path at the frozen Q; Q must not change after the freeze."
        )
    return MATCHED
