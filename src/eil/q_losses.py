from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

LABEL_G = 0 
LABEL_B = 1 
LABEL_I = 2 


def _weighted_mean(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * w).sum() / (w.sum() + eps)


def loss_G(Q: torch.Tensor, w: torch.Tensor, B: float) -> torch.Tensor:
    return _weighted_mean(F.relu(Q - B), w)


def loss_B(Q: torch.Tensor, w: torch.Tensor, B: float) -> torch.Tensor:
    return _weighted_mean(F.relu(B - Q), w)


def loss_I(Q_expert: torch.Tensor, Q_other: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _weighted_mean(F.relu(Q_expert - Q_other), w)


def compute_total_loss(
    Q_all: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    B: float = 0.0,
    lambda_I: float = 1.0,
 
    Q_expert_I: Optional[torch.Tensor] = None,
    Q_other_I: Optional[torch.Tensor] = None,
    weights_I: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the objective:
        L = L_G + L_B + lambda_I * L_I
    """
    Q_all = Q_all.reshape(-1)
    labels = labels.reshape(-1)
    weights = weights.reshape(-1).to(Q_all.dtype)

    # G loss
    mG = (labels == LABEL_G)
    LG = Q_all.new_tensor(0.0) if mG.sum() == 0 else loss_G(Q_all[mG], weights[mG], B)

    # B loss
    mB = (labels == LABEL_B)
    LB = Q_all.new_tensor(0.0) if mB.sum() == 0 else loss_B(Q_all[mB], weights[mB], B)

    # I loss
    LI = Q_all.new_tensor(0.0)
    if Q_expert_I is not None and Q_other_I is not None and weights_I is not None:
        LI = loss_I(
            Q_expert_I.reshape(-1),
            Q_other_I.reshape(-1),
            weights_I.reshape(-1).to(Q_all.dtype),
        )

    total = LG + LB + lambda_I * LI
    return total, {"L_G": LG, "L_B": LB, "L_I": LI, "L_total": total}