"""Stage 5: sustained / sudden attention-drop detector (not a neural net)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from attention_pipeline.features import dominant_modality
from attention_pipeline.types import AttentionEvent, TrackState


@dataclass
class DropConfig:
    low_threshold: float = 40.0
    sustain_seconds: float = 3.0
    sudden_delta: float = 20.0
    sudden_seconds: float = 2.0
    cooldown_seconds: float = 8.0
    ema_alpha: float = 0.3


class AttentionDropDetector:
    def __init__(self, cfg: DropConfig | None = None) -> None:
        self.cfg = cfg or DropConfig()
        self._ema: dict[int, float] = {}

    def reset(self) -> None:
        self._ema.clear()

    def update(self, track: TrackState, feat: np.ndarray) -> Optional[AttentionEvent]:
        if not track.scores or not track.timestamps:
            return None
        sid = track.student_id
        score = track.scores[-1]
        ts = track.timestamps[-1]
        prev = self._ema.get(sid, score)
        ema = self.cfg.ema_alpha * score + (1.0 - self.cfg.ema_alpha) * prev
        self._ema[sid] = ema

        if ts - track.last_event_ts < self.cfg.cooldown_seconds:
            return None

        kind = self._sustained(track) or self._sudden(track)
        if kind is None:
            return None

        track.last_event_ts = ts
        return AttentionEvent(
            student_id=sid,
            timestamp=ts,
            score=float(ema),
            kind=kind,
            dominant_modality=dominant_modality(feat),
            explanation_features={
                "ema_score": float(ema),
                "raw_score": float(score),
                "window_min": float(min(track.scores[-16:])),
                "window_max": float(max(track.scores[-16:])),
            },
        )

    def _sustained(self, track: TrackState) -> Optional[str]:
        t_end = track.timestamps[-1]
        t_start = t_end - self.cfg.sustain_seconds
        vals = [s for s, t in zip(track.scores, track.timestamps) if t >= t_start]
        times = [t for t in track.timestamps if t >= t_start]
        if len(vals) < 3:
            return None
        if times[-1] - times[0] < self.cfg.sustain_seconds * 0.8:
            return None
        if all(v < self.cfg.low_threshold for v in vals):
            return "sustained"
        return None

    def _sudden(self, track: TrackState) -> Optional[str]:
        t_end = track.timestamps[-1]
        t_start = t_end - self.cfg.sudden_seconds
        earlier = [s for s, t in zip(track.scores, track.timestamps) if t_start <= t < t_end]
        if not earlier:
            return None
        peak = max(earlier)
        if peak - track.scores[-1] >= self.cfg.sudden_delta:
            return "sudden"
        return None
