import gymnasium as gym
import numpy as np


class FastSafeCompleteRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        goal_bonus=800.0,
        time_penalty=1.0,
        hazard_entry_penalty=200.0,
        hazard_step_penalty=10.0,
        hazard_key="cost_hazards",
        vase_entry_penalty=200.0,
        vase_step_penalty=10.0,
        scale_step_penalty_by_cost=False,
        terminate_on_goal=True,
        use_progress_shaping=True,
        shaping_k=1.5,
        vase_step_penalize_velocity=False,
        vase_velocity_step_scale=1.0,
    ):
        super().__init__(env)
        self.goal_bonus = float(goal_bonus)
        self.time_penalty = float(time_penalty)

        self.hazard_entry_penalty = float(hazard_entry_penalty)
        self.hazard_step_penalty = float(hazard_step_penalty)
        self.hazard_key = hazard_key

        self.vase_entry_penalty = float(vase_entry_penalty)
        self.vase_step_penalty = float(vase_step_penalty)

        self.scale_step_penalty_by_cost = bool(scale_step_penalty_by_cost)
        self.terminate_on_goal = bool(terminate_on_goal)
        self.use_progress_shaping = bool(use_progress_shaping)
        self.shaping_k = float(shaping_k)

        self.vase_step_penalize_velocity = bool(vase_step_penalize_velocity)
        self.vase_velocity_step_scale = float(vase_velocity_step_scale)

        self._was_in_hazard = False
        self._was_in_vase_contact = False
        self._prev_goal_dist = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        self._was_in_hazard = False
        self._was_in_vase_contact = False
        self._prev_goal_dist = self._goal_distance_or_none()
        return obs, info

    def step(self, action):
        out = self.env.step(action)

        if isinstance(out, tuple) and len(out) == 6:
            obs, _env_reward, cost, terminated, truncated, info = out
            info = dict(info)
            info["cost"] = float(cost)
        elif isinstance(out, tuple) and len(out) == 5:
            obs, _env_reward, terminated, truncated, info = out
            info = dict(info)
        else:
            obs, _env_reward, done, info = out
            terminated, truncated = bool(done), False
            info = dict(info)

        reward = -self.time_penalty

        if self.use_progress_shaping:
            cur = self._goal_distance_or_none()
            if self._prev_goal_dist is not None and cur is not None:
                reward += self.shaping_k * (self._prev_goal_dist - cur)
            self._prev_goal_dist = cur

        goal_met = bool(info.get("goal_met", False))
        if goal_met:
            reward += self.goal_bonus
            if self.terminate_on_goal:
                terminated = True

        hz = float(info.get(self.hazard_key, 0.0))
        in_hazard = hz > 0.0
        hazard_entry = bool(in_hazard and (not self._was_in_hazard))

        if hazard_entry:
            reward -= self.hazard_entry_penalty
        if in_hazard:
            reward -= (self.hazard_step_penalty * hz) if self.scale_step_penalty_by_cost else self.hazard_step_penalty

        self._was_in_hazard = in_hazard

        info["hazard_cost_step"] = hz
        info["hazard_in_contact"] = bool(in_hazard)
        info["hazard_entry"] = bool(hazard_entry)
        info["cost_hazards_used"] = hz

        vc = float(info.get("cost_vases_contact", 0.0))
        vv = float(info.get("cost_vases_velocity", 0.0))

        in_vase_contact = vc > 0.0
        in_vase_velocity = vv > 0.0
        vase_entry = bool(in_vase_contact and (not self._was_in_vase_contact))

        if vase_entry:
            reward -= self.vase_entry_penalty

        if in_vase_contact:
            if self.scale_step_penalty_by_cost:
                reward -= self.vase_step_penalty * vc
            else:
                reward -= self.vase_step_penalty
        elif self.vase_step_penalize_velocity and in_vase_velocity:
            if self.scale_step_penalty_by_cost:
                reward -= (self.vase_step_penalty * self.vase_velocity_step_scale) * vv
            else:
                reward -= self.vase_step_penalty * self.vase_velocity_step_scale

        self._was_in_vase_contact = in_vase_contact

        info["vase_contact_cost_step"] = float(vc)
        info["vase_velocity_cost_step"] = float(vv)
        info["vase_cost_step"] = float(vc + vv)

        info["vase_in_contact"] = bool(in_vase_contact)
        info["vase_in_velocity"] = bool(in_vase_velocity)
        info["vase_entry"] = bool(vase_entry)

        info["cost_vases_contact_used"] = float(vc)
        info["cost_vases_velocity_used"] = float(vv)
        info["cost_vases_used"] = float(vc + vv)

        info["goal_met_used"] = goal_met

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _goal_distance_or_none(self):
        try:
            e = self.env.unwrapped
            agent_xy = np.asarray(e.task.agent.pos[:2], dtype=np.float32)
            goal_xy = np.asarray(e.task.goal.pos[:2], dtype=np.float32)
            return float(np.linalg.norm(agent_xy - goal_xy))
        except Exception:
            return None
