from __future__ import annotations

import sys
from pathlib import Path

# The project is not installed in the venv; tests import `src.` and `scripts.` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
