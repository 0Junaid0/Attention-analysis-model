"""Temporal fusion model: 32-D frame features → attention score 0–100."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from attention_pipeline.types import DAISEE_LABELS, FEAT_DIM


class AttentionFusionNet(nn.Module):
    """Stage 4 of the pipeline.

    Input:  (B, T, feat_dim) packed multimodal features
    Output: score in [0, 100], plus 4 DAiSEE auxiliary logits (0–3 regression)
    """

    def __init__(
        self,
        feat_dim: int = FEAT_DIM,
        hidden: int = 64,
        num_aux: int = len(DAISEE_LABELS),
    ) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.temporal = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        fused = hidden * 2
        self.score_head = nn.Sequential(
            nn.Linear(fused, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.aux_head = nn.Linear(fused, num_aux)

    def encode_window(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        encoded = self.frame_encoder(x.reshape(b * t, -1)).reshape(b, t, -1)
        out, _ = self.temporal(encoded)
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_window(x)
        score = self.score_head(h).squeeze(-1) * 100.0
        aux = self.aux_head(h)
        return score, aux

    @torch.inference_mode()
    def score(self, window: torch.Tensor) -> torch.Tensor:
        if window.dim() == 2:
            window = window.unsqueeze(0)
        scores, _ = self.forward(window)
        return scores

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> "AttentionFusionNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_state_dict(state)
        self.eval()
        return self
