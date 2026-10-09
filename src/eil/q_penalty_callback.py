from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class QPenaltyStatsCallback(BaseCallback):
    """
    Prints mean q_penalty and mean q_value from infos over each rollout collection window.

    Works with SubprocVecEnv because infos are gathered in the main process.
    """
    def __init__(self, print_every_rollout: bool = True, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every_rollout = print_every_rollout
        self._sum_pen = 0.0
        self._sum_q = 0.0
        self._count = 0

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "q_penalty" in info:
                self._sum_pen += float(info["q_penalty"])
                self._count += 1
            if "q_value" in info:
                self._sum_q += float(info["q_value"])
        return True

    def _on_rollout_end(self) -> None:
        if self._count == 0:
            mean_pen = 0.0
            mean_q = 0.0
        else:
            mean_pen = self._sum_pen / self._count
            mean_q = self._sum_q / self._count

        if self.print_every_rollout:
            print(f"[QPenalty] mean_penalty={mean_pen:.6f}  mean_q={mean_q:.6f}  n={self._count}")

        # reset window
        self._sum_pen = 0.0
        self._sum_q = 0.0
        self._count = 0
