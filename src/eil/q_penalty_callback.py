from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class QPenaltyStatsCallback(BaseCallback):
    """
    Prints and logs Q-penalty stats from infos over each rollout collection window:
    mean penalty, mean Q, fraction of steps with Q > B, mean shaped step reward,
    and the penalty's share of the reward before the penalty was applied.

    Works with SubprocVecEnv because infos are gathered in the main process.
    """
    def __init__(self, print_every_rollout: bool = True, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every_rollout = print_every_rollout
        self._reset_window()

    def _reset_window(self) -> None:
        self._sum_pen = 0.0
        self._sum_q = 0.0
        self._n_active = 0
        self._count = 0
        self._sum_rew = 0.0
        self._n_rew = 0

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
                if "q_B" in info and float(info["q_value"]) > float(info["q_B"]):
                    self._n_active += 1

        # Per-step reward after shaping (penalty included)
        rewards = self.locals.get("rewards")
        if rewards is not None:
            rewards = np.asarray(rewards, dtype=np.float64)
            self._sum_rew += float(rewards.sum())
            self._n_rew += int(rewards.size)
        return True

    def _on_rollout_end(self) -> None:
        if self._count == 0:
            mean_pen = 0.0
            mean_q = 0.0
            active_frac = 0.0
        else:
            mean_pen = self._sum_pen / self._count
            mean_q = self._sum_q / self._count
            active_frac = self._n_active / self._count

        mean_rew = self._sum_rew / self._n_rew if self._n_rew else 0.0
        # Share of the pre-penalty reward magnitude taken by the penalty
        pre_pen = abs(mean_rew + mean_pen)
        penalty_ratio = mean_pen / pre_pen if pre_pen > 0 else 0.0

        self.logger.record("q/mean_penalty", mean_pen)
        self.logger.record("q/mean_q", mean_q)
        self.logger.record("q/active_frac", active_frac)
        self.logger.record("q/mean_step_reward", mean_rew)
        self.logger.record("q/penalty_ratio", penalty_ratio)

        if self.print_every_rollout:
            print(
                f"[QPenalty] mean_penalty={mean_pen:.6f}  mean_q={mean_q:.6f}  "
                f"active_frac={active_frac:.3f}  mean_step_reward={mean_rew:.4f}  "
                f"penalty_ratio={penalty_ratio:.4f}  n={self._count}"
            )

        self._reset_window()
