from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EMOTION_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

# SCB-dataset3 (https://github.com/Whiffe/SCB-dataset)
SCB_BEHAVIOR_LABELS = (
    "hand-raising",
    "reading",
    "writing",
    "using-phone",
    "bowing-head",
    "leaning-over-table",
)
POSTURE_LABELS = SCB_BEHAVIOR_LABELS

DAISEE_LABELS = ("engagement", "boredom", "confusion", "frustration")

FEAT_DIM = 32
POSE_KEYPOINTS = 17


@dataclass
class StudentFrame:
    """Per-student, per-frame observation after stages 2–3."""

    student_id: int
    timestamp: float
    bbox: np.ndarray  # xyxy
    keypoints: np.ndarray  # (17, 3) x, y, conf
    head_pose: np.ndarray  # yaw, pitch, roll degrees
    gaze: np.ndarray  # yaw, pitch degrees
    emotion_probs: np.ndarray  # (7,)
    posture_probs: np.ndarray  # (6,)
    motion: float
    eye_aspect: float
    mouth_aspect: float
    features: np.ndarray  # (FEAT_DIM,)


@dataclass
class TrackState:
    student_id: int
    feature_window: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    last_bbox: Optional[np.ndarray] = None
    last_event_ts: float = -1e9


@dataclass
class AttentionEvent:
    student_id: int
    timestamp: float
    score: float
    kind: str  # sustained | sudden
    dominant_modality: str
    explanation_features: dict
