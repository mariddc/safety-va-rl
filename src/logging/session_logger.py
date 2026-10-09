import time
from pathlib import Path
import numpy as np
import json

class SessionLogger:
    def __init__(self, out_root: Path, prefix="session", meta: dict | None = None):
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = out_root / f"{prefix}_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.npz_path = self.session_dir / "session.npz"
        self.meta_path = self.session_dir / "meta.json"

        if meta is not None:
            with open(self.meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        self.obs = []
        self.a_base = []
        self.a_exec = []
        self.override = []
        self.reward = []
        self.cost = []

        self.hz = []
        self.hz_in = []
        self.hz_entry = []

        self.contact_cost = []
        self.velocity_cost = []

        self.terminated = []
        self.truncated = []

    def add(self, obs, info, reward, cost, terminated, truncated):
        self.obs.append(np.array(obs, dtype=np.float32).copy())

        self.a_base.append(np.array(info.get("action_base"), dtype=np.float32).reshape(-1))
        self.a_exec.append(np.array(info.get("action_exec"), dtype=np.float32).reshape(-1))

        self.override.append(int(bool(info.get("human_override", False))))
        self.reward.append(float(reward))
        self.cost.append(float(cost))

        hz = float(info.get("hazard_cost_step", info.get("cost_hazards_used", 0.0)))
        self.hz.append(hz)
        self.hz_in.append(int(bool(info.get("hazard_in_contact", hz > 0.0))))
        self.hz_entry.append(int(bool(info.get("hazard_entry", False))))

        self.contact_cost.append(float(info.get("contact_cost", 0.0)))
        self.velocity_cost.append(float(info.get("velocity_cost", 0.0)))

        self.terminated.append(int(bool(terminated)))
        self.truncated.append(int(bool(truncated)))

    def save(self):
        np.savez_compressed(
            self.npz_path,
            obs=np.stack(self.obs),
            action_base=np.stack(self.a_base),
            action_exec=np.stack(self.a_exec),
            human_override=np.array(self.override, dtype=np.int8),
            reward=np.array(self.reward, dtype=np.float32),
            cost=np.array(self.cost, dtype=np.float32),
            hazard_cost_step=np.array(self.hz, dtype=np.float32),
            hazard_in_contact=np.array(self.hz_in, dtype=np.int8),
            hazard_entry=np.array(self.hz_entry, dtype=np.int8),
            contact_cost=np.array(self.contact_cost, dtype=np.float32),
            velocity_cost=np.array(self.velocity_cost, dtype=np.float32),
            terminated=np.array(self.terminated, dtype=np.int8),
            truncated=np.array(self.truncated, dtype=np.int8),
        )
        print(f"[logger] saved {len(self.override)} steps to {self.npz_path}")