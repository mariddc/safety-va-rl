from pathlib import Path

import gymnasium as gym
import imageio
import matplotlib
import numpy as np
import safety_gymnasium
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from src.wrappers.reward import wrap_reward

HEADER = 32  # frame height 256 + 32 is a multiple of 16, so ffmpeg does not rescale


def make_env(exp: dict):
    env = safety_gymnasium.make(
        exp["env_id"],
        render_mode="rgb_array",
        camera_name="fixedfar",
    )
    return wrap_reward(env, exp)


def save_video(exp_name: str, exp: dict, run_dir: Path):

    env_id = exp["env_id"]
    algo = exp["algorithm"]

    model_path = run_dir / "model" / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    video_dir = run_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    video_length = exp.get("video_length", 1000)

    env = DummyVecEnv([lambda: make_env(exp)])

    if algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported SB3 algorithm for video: {algo}")

    env = VecVideoRecorder(
        env,
        video_folder=str(video_dir),
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=exp_name,
    )

    obs = env.reset()
    for step in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        if dones.any() and step < video_length - 1:
            obs = env.reset()

    env.close()

    print(f"[SB3] Video saved to {video_dir}")


class _FrameRecorder(gym.Wrapper):
    """Renders after reset and after every step, so the last frame is kept before DummyVecEnv auto-resets."""

    def __init__(self, env):
        super().__init__(env)
        self.frames = []

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.frames.append(self.env.render())
        return out

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(self.env.render())
        return out


def replay_episode(exp: dict, model_path: Path, seed: int):
    """One deterministic seeded episode, set up and counted like the seeded eval in sb3_eval.py.

    Returns frames (initial state + one per step), per-step (cost, hazard, vase contact) and the outcome.
    """
    recorder = _FrameRecorder(make_env(exp))
    env = DummyVecEnv([lambda: recorder])
    model = (SAC if exp["algorithm"] == "sac" else PPO).load(model_path, env=env)

    env.seed(seed)
    obs = env.reset()
    steps, success = [], 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        info = infos[0]
        if info.get("goal_met_used", False) or info.get("goal_met", False) or info.get("is_success", False) or info.get("success", False):
            success = 1
        hz = float(info.get("hazard_cost_step", info.get("cost_hazards_used", info.get("cost_hazards", 0.0))))
        vc = float(info.get("cost_vases_contact", info.get("cost_vases_contact_used", 0.0)))
        steps.append((float(info.get("cost", 0.0)), bool(info.get("hazard_in_contact", False)) or hz > 0.0, vc > 0.0))
        if done[0]:
            break
    env.close()

    frames = recorder.frames[:len(steps) + 1]  # drop the auto-reset frame of the next episode
    outcome = {
        "success": success,
        "length": len(steps),
        "cost": sum(s[0] for s in steps),
        "hazard_steps": sum(s[1] for s in steps),
        "vase_contact_steps": sum(s[2] for s in steps),
    }
    return frames, steps, outcome


def label_frames(frames, steps, label: str, n_frames: int | None = None):
    """Adds a header (label, step, cumulative cost) and a red border on steps with cost > 0.

    Frames past the episode's end repeat the last one, so several episodes can be shown side by side.
    """
    # Pillow's default font has no β; DejaVu ships with matplotlib
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 13)
    out, cum = [], 0.0
    for t in range(n_frames or len(frames)):
        k = min(t, len(frames) - 1)
        cost, hazard, vase = steps[k - 1] if k > 0 else (0.0, False, False)
        if t == k:
            cum += cost
        img = Image.new("RGB", (frames[k].shape[1], frames[k].shape[0] + HEADER), "white")
        img.paste(Image.fromarray(frames[k]), (0, HEADER))
        d = ImageDraw.Draw(img)
        d.text((4, 1), label, fill="black", font=font)
        if t > k:
            status = "ended"
        else:
            status = "in hazard" if hazard else "vase touch" if vase else "vase moving" if cost > 0 else ""
        d.text((4, 16), f"step {k}/{len(steps)}  cost {cum:g}  {status}", fill="black", font=font)
        if t == k and cost > 0:
            d.rectangle([0, HEADER, img.width - 1, img.height - 1], outline="red", width=4)
        out.append(np.asarray(img))
    return out


def write_mp4(path: Path, frames, fps: int = 30):
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
