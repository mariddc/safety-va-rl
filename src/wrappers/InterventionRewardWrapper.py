from __future__ import annotations

from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


class InterventionRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, q_path: str, beta: float = 1.0, B: float = 0.0, device: str = "cpu"):
        super().__init__(env)
        self.beta = float(beta)
        self.B = float(B)
        self.device = torch.device(device)

        ckpt = torch.load(q_path, map_location=self.device)
        obs_dim = int(ckpt["obs_dim"])
        act_dim = int(ckpt["act_dim"])

        self.q = QNetwork(obs_dim, act_dim).to(self.device)
        self.q.load_state_dict(ckpt["state_dict"])
        self.q.eval()

        self._last_obs: Optional[np.ndarray] = None

        self._ep_steps: int = 0
        self._ep_q_sum: float = 0.0
        self._ep_pen_sum: float = 0.0
        self._ep_active: int = 0  # count of steps where q > B

    def _reset_episode_stats(self) -> None:
        self._ep_steps = 0
        self._ep_q_sum = 0.0
        self._ep_pen_sum = 0.0
        self._ep_active = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        # Gymnasium-style reset
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._reset_episode_stats()
        return obs, dict(info)

    def step(self, action):
        if self._last_obs is None:
            out = self.env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                obs0, _info0 = out
            else:
                obs0, _info0 = out, {}
            self._last_obs = np.asarray(obs0, dtype=np.float32)
            self._reset_episode_stats()

        # Penalty computed from obs BEFORE applying action
        obs_in = np.asarray(self._last_obs, dtype=np.float32)
        act_in = np.asarray(action, dtype=np.float32)

        with torch.no_grad():
            o = torch.from_numpy(obs_in).to(self.device).unsqueeze(0)
            a = torch.from_numpy(act_in).to(self.device).unsqueeze(0)
            q_val = float(self.q(o, a).item())

        # penalty = self.beta * max(0.0, q_val - self.B)
        penalty = self.beta * max(0.0, self.B - q_val)

        # Step env
        out = self.env.step(action)

        # Safety-Gymnasium: (obs, reward, cost, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            info = dict(info)
            info["cost"] = float(cost)

        # Gymnasium: (obs, reward, terminated, truncated, info)
        elif isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            info["cost"] = float(info.get("cost", 0.0))

        # Old gym: (obs, reward, done, info)
        else:
            obs, reward, done, info = out
            info = dict(info)
            terminated, truncated = bool(done), False
            info["cost"] = float(info.get("cost", 0.0))

        # Apply shaping
        reward = float(reward) - float(penalty)

        info["q_penalty"] = float(penalty)
        info["q_value"] = float(q_val)
        info["q_B"] = float(self.B)
        info["q_beta"] = float(self.beta)

        self._ep_steps += 1
        self._ep_q_sum += float(q_val)
        self._ep_pen_sum += float(penalty)
        if q_val > self.B:
            self._ep_active += 1

        if bool(terminated) or bool(truncated):
            steps = max(1, self._ep_steps)
            q_mean = self._ep_q_sum / steps
            pen_mean = self._ep_pen_sum / steps
            active_frac = float(self._ep_active) / float(steps)

            info["q_episode"] = {
                "steps": int(steps),
                "q_mean": float(q_mean),
                "qpen_mean": float(pen_mean),
                "qpen_sum": float(self._ep_pen_sum),
                "q_active_frac": float(active_frac),
                "B": float(self.B),
                "beta": float(self.beta),
            }

            ep = dict(info.get("episode", {}))
            ep.update(
                {
                    "q_mean": float(q_mean),
                    "qpen_mean": float(pen_mean),
                    "qpen_sum": float(self._ep_pen_sum),
                    "q_active_frac": float(active_frac),
                }
            )
            info["episode"] = ep

        # Update last obs
        self._last_obs = np.asarray(obs, dtype=np.float32)

        return obs, reward, bool(terminated), bool(truncated), info
