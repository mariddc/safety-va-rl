from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.experiments.registry import get_experiment
from src.utils.path import get_intervention_root, get_eil_dataset_root
from logs.datasets.eil_builder import BuilderConfig, build_eil_dataset
from src.eil.q_losses import compute_total_loss


ROOT = Path(__file__).resolve().parents[2]

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


def build_dataset_from_sessions(exp_name: str, dataset_id: str) -> Path:
    exp = get_experiment(exp_name)

    interventions_root = get_intervention_root(exp, ROOT, exp_name)
    session_files = [Path(p) for p in glob.glob(str(interventions_root / "session_*" / "session.npz"))]
    if not session_files:
        raise FileNotFoundError(f"No sessions found under: {interventions_root}")

    out_root = get_eil_dataset_root(exp, ROOT, exp_name, dataset_id)

    cfg = BuilderConfig(
        dataset_id=dataset_id,
        out_root=str(out_root.parent),
        seed=0,
        alphaL=0.2,
        alphaE=0.0,
        downweight_contact=0.25,
        downweight_hazard_step=0.5,
        override_threshold=0.5,
    )

    out_dir = build_eil_dataset(session_files, cfg, env_name=exp["env_id"])
    return out_dir


def train_qtheta(dataset_dir: Path, *, B: float = 0.0, lambda_I: float = 1.0,
                 lr: float = 3e-4, batch_size: int = 2048, steps: int = 20_000,
                 device: str = "cpu") -> Path:
    data_path = dataset_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing: {data_path}")

    data = np.load(data_path, allow_pickle=False)

    obs = data["obs"].astype(np.float32)
    a_base = data["action_base"].astype(np.float32)
    a_exec = data["action_exec"].astype(np.float32)
    labels = data["label"].astype(np.int64)
    weights = data["weight"].astype(np.float32)

    N, obs_dim = obs.shape
    act_dim = a_base.shape[1]

    # Torch tensors
    obs_t = torch.from_numpy(obs).to(device)
    a_base_t = torch.from_numpy(a_base).to(device)
    a_exec_t = torch.from_numpy(a_exec).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    w_t = torch.from_numpy(weights).to(device)

    idx_G = torch.where(labels_t == 0)[0]
    idx_B = torch.where(labels_t == 1)[0]
    idx_I = torch.where(labels_t == 2)[0]

    model = QNetwork(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # simple sampler: mix G/B/I each step
    def sample_idx(idx: torch.Tensor, k: int) -> torch.Tensor:
        if idx.numel() == 0:
            return idx
        r = torch.randint(0, idx.numel(), (k,), device=device)
        return idx[r]

    log = {"steps": [], "L_total": [], "L_G": [], "L_B": [], "L_I": []}

    for it in range(steps):
        opt.zero_grad()

        k = batch_size
        kG = k // 3
        kB = k // 3
        kI = k - kG - kB

        bG = sample_idx(idx_G, kG)
        bB = sample_idx(idx_B, kB)
        bI = sample_idx(idx_I, kI) if idx_I.numel() > 0 else torch.empty((0,), dtype=torch.long, device=device)

        batch_idx = torch.cat([bG, bB, bI], dim=0) if bI.numel() > 0 else torch.cat([bG, bB], dim=0)

        o = obs_t[batch_idx]
        ab = a_base_t[batch_idx]
        lab = labels_t[batch_idx]
        ww = w_t[batch_idx]

        Q_all = model(o, ab)
        Q_expert_I = Q_other_I = weights_I = None

        if bI.numel() > 0:
            oI = obs_t[bI]
            aI_exec = a_exec_t[bI]
            aI_base = a_base_t[bI]
            weights_I = w_t[bI]
            Q_expert_I = model(oI, aI_exec)
            Q_other_I = model(oI, aI_base)

        total, parts = compute_total_loss(
            Q_all=Q_all,
            labels=lab,
            weights=ww,
            B=B,
            lambda_I=lambda_I,
            Q_expert_I=Q_expert_I,
            Q_other_I=Q_other_I,
            weights_I=weights_I,
        )

        total.backward()
        opt.step()

        if it % 200 == 0:
            log["steps"].append(int(it))
            log["L_total"].append(float(parts["L_total"].detach().cpu()))
            log["L_G"].append(float(parts["L_G"].detach().cpu()))
            log["L_B"].append(float(parts["L_B"].detach().cpu()))
            log["L_I"].append(float(parts["L_I"].detach().cpu()))
            print(f"[{it:05d}] L={log['L_total'][-1]:.4f}  G={log['L_G'][-1]:.4f}  B={log['L_B'][-1]:.4f}  I={log['L_I'][-1]:.4f}")

    # Save
    out_path = dataset_dir / "q_theta.pt"
    torch.save({"state_dict": model.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim}, out_path)
    (dataset_dir / "q_train_log.json").write_text(json.dumps(log, indent=2))

    return out_path


if __name__ == "__main__":
    exp_name = "ppo_5m700k_goal2"
    dataset_id = "eil_aL0.2_aE0.0_v1"

    dataset_dir = build_dataset_from_sessions(exp_name, dataset_id)
    q_path = train_qtheta(dataset_dir, device="cpu")
