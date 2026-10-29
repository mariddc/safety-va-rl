from pathlib import Path

import safety_gymnasium
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from src.wrappers.reward import wrap_reward


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
