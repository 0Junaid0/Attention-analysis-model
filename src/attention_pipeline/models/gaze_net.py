"""L2CS-style gaze: shared backbone, two binned heads for pitch and yaw."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

IDX_TENSOR = torch.arange(90).float()


class GazeNet(nn.Module):
    """Predict gaze pitch/yaw in degrees via 90-bin classification (L2CS)."""

    def __init__(self, num_bins: int = 90) -> None:
        super().__init__()
        self.num_bins = num_bins
        backbone = resnet18(weights=None)
        feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fc_pitch = nn.Linear(feat, num_bins)
        self.fc_yaw = nn.Linear(feat, num_bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.fc_pitch(feat), self.fc_yaw(feat)

    @torch.inference_mode()
    def predict_degrees(self, face_batch: torch.Tensor) -> torch.Tensor:
        """Return (N, 2) as yaw, pitch in degrees, mapped from 0–89 bins to -90..90."""
        pitch_logits, yaw_logits = self.forward(face_batch)
        idx = IDX_TENSOR.to(face_batch.device)
        pitch_bin = (torch.softmax(pitch_logits, dim=1) * idx).sum(dim=1)
        yaw_bin = (torch.softmax(yaw_logits, dim=1) * idx).sum(dim=1)
        pitch = pitch_bin * 2.0 - 90.0
        yaw = yaw_bin * 2.0 - 90.0
        return torch.stack([yaw, pitch], dim=1)

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> "GazeNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_state_dict(state)
        self.eval()
        return self
