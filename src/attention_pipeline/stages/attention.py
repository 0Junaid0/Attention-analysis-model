"""Stage 4: windowed AttentionFusionNet scoring."""

from __future__ import annotations

import numpy as np
import torch

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.rubric import RubricResult, classify_attention, classify_from_features
from attention_pipeline.types import StudentFrame, TrackState


class AttentionScorer:
    def __init__(
        self,
        model: AttentionFusionNet | None,
        device: torch.device | None,
        window: int = 16,
    ) -> None:
        self.model = model
        self.device = device
        self.window = window
        self.use_heuristic = False

    def push(
        self,
        track: TrackState,
        feat: np.ndarray,
        timestamp: float,
        obs: StudentFrame | None = None,
        lesson_yaw: float | None = None,
    ) -> tuple[float, RubricResult | None]:
        track.feature_window.append(feat.astype(np.float32))
        track.timestamps.append(float(timestamp))
        if len(track.feature_window) > self.window:
            track.feature_window = track.feature_window[-self.window :]
            track.timestamps = track.timestamps[-self.window :]
            track.scores = track.scores[-(self.window - 1) :]

        stacked = np.stack(track.feature_window, axis=0)
        if stacked.shape[0] < self.window:
            pad = np.repeat(stacked[:1], self.window - stacked.shape[0], axis=0)
            stacked = np.concatenate([pad, stacked], axis=0)

        rubric = None
        if self.use_heuristic:
            rubric = (
                classify_attention(obs, lesson_yaw=lesson_yaw)
                if obs is not None
                else classify_from_features(feat, lesson_yaw=lesson_yaw)
            )
            score = rubric.score
        else:
            x = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
            score = float(self.model.score(x).item())
        track.scores.append(score)
        return score, rubric
