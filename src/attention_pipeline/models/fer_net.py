"""7-class FER CNN. Pretrain FER2013, fine-tune RAF-DB."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18


class FERNet(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.inference_mode()
    def predict_proba(self, face_bgr_batch: torch.Tensor) -> torch.Tensor:
        logits = self.forward(face_bgr_batch)
        return torch.softmax(logits, dim=1)

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> "FERNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_state_dict(state)
        self.eval()
        return self
