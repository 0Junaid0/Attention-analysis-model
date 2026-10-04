"""Posture classifier over YOLO11-Pose keypoints (SCB-Dataset)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from attention_pipeline.types import POSE_KEYPOINTS, POSTURE_LABELS


class PostureMLP(nn.Module):
    def __init__(
        self,
        num_classes: int = len(POSTURE_LABELS),
        hidden: int = 64,
    ) -> None:
        super().__init__()
        in_dim = POSE_KEYPOINTS * 2  # normalized x, y
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.inference_mode()
    def predict_proba(self, keypoints_xy: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(keypoints_xy), dim=1)

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> "PostureMLP":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_state_dict(state)
        self.eval()
        return self


def normalize_keypoints(kpts: torch.Tensor, bbox_xyxy: torch.Tensor) -> torch.Tensor:
    """Map (N, 17, 2) into bbox-relative [0, 1] coordinates."""
    x1, y1, x2, y2 = bbox_xyxy.unbind(-1)
    w = (x2 - x1).clamp(min=1.0).unsqueeze(-1)
    h = (y2 - y1).clamp(min=1.0).unsqueeze(-1)
    x = (kpts[..., 0] - x1.unsqueeze(-1)) / w
    y = (kpts[..., 1] - y1.unsqueeze(-1)) / h
    return torch.stack([x, y], dim=-1).reshape(kpts.shape[0], -1).clamp(0, 1)
