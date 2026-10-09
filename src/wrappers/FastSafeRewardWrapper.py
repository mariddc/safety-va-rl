import gymnasium as gym
import numpy as np

class FastSafeRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        goal_bonus=50.0,
        time_penalty=1.0,
        hazard_entry_penalty=200.0,
        hazard_step_penalty=10.0,
        hazard_key="cost_hazards",
        scale_step_penalty_by_cost=False,
        terminate_on_goal=True,
        use_progress_shaping=True,
        shaping_k=1.0,
    ):
        super().__init__(env)
        self.goal_bonus = float(goal_bonus)
        self.time_penalty = float(time_penalty)

        self.hazard_entry_penalty = float(hazard_entry_penalty)
        self.hazard_step_penalty = float(hazard_step_penalty)
        self.hazard_key = hazard_key

        self.scale_step_penalty_by_cost = scale_step_penalty_by_cost

        self.terminate_on_goal = terminate_on_goal
        self.use_progress_shaping = use_progress_shaping
        self.shaping_k = float(shaping_k)

        self._was_in_hazard = False
        self._prev_goal_dist = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)

        # Gymnasium-style reset
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        self._was_in_hazard = False
        self._prev_goal_dist = self._goal_distance_or_none()
        return obs, info

    def step(self, action):
        out = self.env.step(action)

        # Safety-Gymnasium
        if isinstance(out, tuple) and len(out) == 6:
            obs, _env_reward, cost, terminated, truncated, info = out
            info = dict(info)
            info["cost"] = float(cost)

        # Gymnasium
        elif isinstance(out, tuple) and len(out) == 5:
            obs, _env_reward, terminated, truncated, info = out
            info = dict(info)

        # Old Gym
        else:
            obs, _env_reward, done, info = out
            terminated, truncated = bool(done), False
            info = dict(info)

        # Min steps (fast)
        reward = -self.time_penalty

        # Progress shaping (dense signal)
        if self.use_progress_shaping:
            cur = self._goal_distance_or_none()
            if self._prev_goal_dist is not None and cur is not None:
                reward += self.shaping_k * (self._prev_goal_dist - cur)
            self._prev_goal_dist = cur

        # Goal bonus
        goal_met = bool(info.get("goal_met", False))
        if goal_met:
            reward += self.goal_bonus
            if self.terminate_on_goal:
                terminated = True

        # Hazard penalties
        hz = float(info.get(self.hazard_key, 0.0))
        in_hazard = hz > 0.0

        hazard_entry = bool(in_hazard and (not self._was_in_hazard))

        if hazard_entry:
            reward -= self.hazard_entry_penalty

        if in_hazard:
            if self.scale_step_penalty_by_cost:
                reward -= self.hazard_step_penalty * hz
            else:
                reward -= self.hazard_step_penalty

        self._was_in_hazard = in_hazard

        # Expose hazard tracking signals for eval
        info["hazard_cost_step"] = hz
        info["hazard_in_contact"] = bool(in_hazard)
        info["hazard_entry"] = hazard_entry
        info["cost_hazards_used"] = hz

        info["goal_met_used"] = goal_met

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _goal_distance_or_none(self):
        """
        Compute distance from agent to current goal.
        """
        try:
            e = self.env.unwrapped
            agent_xy = np.asarray(e.task.agent.pos[:2], dtype=np.float32)
            goal_xy = np.asarray(e.task.goal.pos[:2], dtype=np.float32)
            return float(np.linalg.norm(agent_xy - goal_xy))

        except Exception:
            return None