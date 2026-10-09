import numpy as np
import pygame
import safety_gymnasium
from stable_baselines3 import PPO, SAC

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wrappers.FastSafeCompleteRewardWrapper import FastSafeCompleteRewardWrapper
from src.logging.session_logger import SessionLogger
from src.experiments.registry import get_experiment, get_run_dir
from src.utils.path import get_intervention_root

def _is_number(x):
    return isinstance(x, (int, float, np.number))

def extract_cost_components(info: dict) -> dict:
    """
    Return dict of all numeric cost-like components found in info.
    Robust to different naming conventions across wrappers/env versions.
    """
    comps = {}

    for k, v in info.items():
        if not _is_number(v):
            continue
        lk = k.lower()
        if lk.startswith("cost_") or lk.endswith("_cost") or ("cost" in lk and lk != "cost"):
            comps[k] = float(v)

    if "cost" in info and _is_number(info["cost"]):
        comps["cost"] = float(info["cost"])

    return comps

def sum_matching_costs(components: dict, substrings) -> float:
    substrings = [s.lower() for s in substrings]
    s = 0.0
    for k, v in components.items():
        lk = k.lower()
        if any(sub in lk for sub in substrings):
            s += float(v)
    return s


class KeyboardController:
    def __init__(self, action_space, throttle=0.3, turn=1.0):
        self.low = np.array(action_space.low, dtype=np.float32)
        self.high = np.array(action_space.high, dtype=np.float32)
        self.throttle = float(throttle)
        self.turn = float(turn)

    def get_action(self):
        keys = pygame.key.get_pressed()
        action = np.zeros_like(self.low, dtype=np.float32)

        # arrows: up/down forward/reverse, left/right turn
        if keys[pygame.K_UP]:
            action[0] = +self.throttle
        elif keys[pygame.K_DOWN]:
            action[0] = -self.throttle

        if keys[pygame.K_LEFT]:
            action[1] = +self.turn
        elif keys[pygame.K_RIGHT]:
            action[1] = -self.turn

        return np.clip(action, self.low, self.high)


class HumanOverrideWrapper:
    
    def __init__(self, env, human_controller):
        self.env = env
        self.human = human_controller

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, policy_action):
        def _to_1d(a):
            return np.array(a, dtype=np.float32).reshape(-1)

        pygame.event.pump()
        keys = pygame.key.get_pressed()

        base = _to_1d(policy_action)

        if keys[pygame.K_SPACE]:
            exec_a = _to_1d(self.human.get_action())
            human_override = True
        else:
            exec_a = base
            human_override = False

        out = self.env.step(exec_a)

        # Safety-Gymnasium API
        if isinstance(out, tuple) and len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            info = dict(info)
            cost = float(cost)

        # Gymnasium API
        elif isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info)
            cost = float(info.get("cost", 0.0))

        # Old Gym fallback
        else:
            obs, reward, done, info = out
            info = dict(info)
            terminated, truncated = bool(done), False
            cost = float(info.get("cost", 0.0))

        # ALWAYS attach these for logging/learning
        info["human_override"] = human_override
        info["cost"] = cost
        info["action_base"] = base
        info["action_exec"] = exec_a

        return obs, reward, cost, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def frame_to_surface(frame_rgb: np.ndarray) -> pygame.Surface:
    frame_rgb = np.ascontiguousarray(frame_rgb)
    return pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))


def draw_label(screen, font, lines):
    if isinstance(lines, str):
        lines = [lines]

    rendered = [font.render(line, True, (255, 255, 255)) for line in lines]
    w = max(r.get_width() for r in rendered) + 16
    h = sum(r.get_height() for r in rendered) + 10 + (len(rendered) - 1) * 4

    bg = pygame.Surface((w, h))
    bg.set_alpha(160)
    bg.fill((0, 0, 0))
    screen.blit(bg, (10, 10))

    y = 15
    for r in rendered:
        screen.blit(r, (18, y))
        y += r.get_height() + 4


def _resolve_model_path(run_dir: Path) -> Path:
    
    p1 = run_dir / "model" / "model.zip"
    if p1.exists():
        return p1
    p2 = run_dir / "model" / "model"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Model not found in {run_dir / 'model'} (expected model.zip or model)")


def _load_model(algo: str, model_path: Path):
    algo = algo.lower()
    if algo == "ppo":
        return PPO.load(str(model_path), device="cpu")
    if algo == "sac":
        return SAC.load(str(model_path), device="cpu")
    raise ValueError(f"Unsupported algorithm: {algo}")


def main():
    exp_name = sys.argv[1]
    exp = get_experiment(exp_name)

    env_id = exp["env_id"]
    algo = exp["algorithm"]
    timesteps = exp.get("timesteps", None)

    run_dir = get_run_dir(exp)
    model_path = _resolve_model_path(run_dir)

    interventions_root = get_intervention_root(exp, ROOT, exp_name)
    
    # Pygame window
    W, H = 1200, 900
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Safety-Gymnasium | SPACE = HUMAN override | ESC quit")
    font = pygame.font.SysFont(None, 30)

    # Env
    env_raw = safety_gymnasium.make(
        env_id,
        render_mode="rgb_array",
        width=W,
        height=H,
        camera_name="fixedfar",
    )
    env_wrapped = FastSafeCompleteRewardWrapper(env_raw)

    human = KeyboardController(env_raw.action_space, throttle=0.3, turn=1.0)
    env = HumanOverrideWrapper(env_wrapped, human)

    model = _load_model(algo, model_path)
    obs, _ = env.reset()

    meta = {
        "experiment_name": exp_name,
        "env_id": env_id,
        "algorithm": algo,
        "timesteps": timesteps,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "wrapper": "FastSafeCompleteRewardWrapper",
    }


    logger = SessionLogger(out_root=interventions_root, prefix="session", meta=meta)
    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            policy_action, _ = model.predict(obs, deterministic=True)
            obs, reward, cost, terminated, truncated, info = env.step(policy_action)

            cost_comps = extract_cost_components(info)

            # Vase cost
            vase_cost_step = sum_matching_costs(cost_comps, ["vase", "vases"])
            info["vase_cost_step"] = float(vase_cost_step)

            # Hazard cost
            haz_cost_step = float(info.get("hazard_cost_step", info.get("cost_hazards_used", 0.0)))
            if haz_cost_step == 0.0:
                haz_cost_step = sum_matching_costs(cost_comps, ["hazard", "hazards"])
            info["hazard_cost_step"] = float(haz_cost_step)

            # Define contact-by-cost
            info["vase_in_contact"] = bool(info.get("vase_in_contact", vase_cost_step > 0.0))
            info["hazard_in_contact"] = bool(info.get("hazard_in_contact", haz_cost_step > 0.0))


            logger.add(obs, info, reward, cost, terminated, truncated)

            if terminated or truncated:
                obs, _ = env.reset()

            frame = env.render()
            if frame is not None:
                screen.blit(frame_to_surface(frame), (0, 0))

                hz = float(info.get("hazard_cost_step", 0.0))
                vz = float(info.get("vase_cost_step", 0.0))

                mode = "HUMAN" if info.get("human_override") else "POLICY"

                lines = [
                    f"EXP: {exp_name}",
                    f"MODE: {mode}",
                    f"hazard {hz:.3f}",
                    f"vase {vz:.3f}",
                    f"reward: {float(reward):+.2f} | cost: {float(cost):.2f}",
                ]
                draw_label(screen, font, lines)
                pygame.display.flip()

            clock.tick(30)

    finally:
        env.close()
        pygame.quit()
        logger.save()


if __name__ == "__main__":
    main()