"""Pack pose / L2CS gaze / SCB behavior outputs into the 32-D fusion vector."""

from __future__ import annotations

import numpy as np

from attention_pipeline.types import FEAT_DIM, StudentFrame

_EPS = 1e-6


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def facing_front_score(yaw_deg: float) -> float:
    return _clip01(1.0 - abs(yaw_deg) / 45.0)


def torso_lean(keypoints: np.ndarray) -> float:
    """0 = upright, 1 = strongly leaned. Uses shoulders (5,6) and hips (11,12)."""
    if keypoints.shape[0] < 13:
        return 0.0
    sh = keypoints[[5, 6], :2].mean(axis=0)
    hp = keypoints[[11, 12], :2].mean(axis=0)
    vec = sh - hp
    mag = np.linalg.norm(vec) + _EPS
    horizontal = abs(vec[0]) / mag
    return _clip01(horizontal)


def pack_features(
    head_pose: np.ndarray,
    gaze: np.ndarray,
    emotion_probs: np.ndarray,
    posture_probs: np.ndarray,
    motion: float,
    eye_aspect: float,
    mouth_aspect: float,
    keypoints: np.ndarray,
) -> np.ndarray:
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    feat[0:3] = np.clip(np.asarray(head_pose, dtype=np.float32) / 180.0, -1, 1)
    feat[3:5] = np.clip(np.asarray(gaze, dtype=np.float32) / 180.0, -1, 1)
    feat[5:12] = np.asarray(emotion_probs, dtype=np.float32).reshape(-1)[:7]
    feat[12:18] = np.asarray(posture_probs, dtype=np.float32).reshape(-1)[:6]
    feat[18] = _clip01(motion)
    feat[19] = _clip01(eye_aspect)
    feat[20] = _clip01(mouth_aspect)
    feat[21] = facing_front_score(float(head_pose[0]))
    feat[22] = torso_lean(keypoints)
    return feat


def heuristic_attention_score(feat: np.ndarray) -> float:
    """Rubric score from the 32-D vector when no StudentFrame is available."""
    from attention_pipeline.rubric import classify_from_features

    return classify_from_features(feat).score


def dominant_modality(feat: np.ndarray) -> str:
    """Heuristic: which cue most explains a low score for the LLM payload."""
    yaw = abs(feat[0] * 180.0)
    gaze = np.linalg.norm(feat[3:5] * 180.0)
    emotion_neg = float(feat[5] + feat[6] + feat[7] + feat[9])  # angry, disgust, fear, sad
    looking_down = float(feat[15] + feat[17])  # using-phone + leaning-over-table
    motion = float(feat[18])
    scores = {
        "head_pose": yaw / 45.0 + (1.0 - feat[21]),
        "gaze": gaze / 30.0,
        "emotion": emotion_neg,
        "posture": looking_down + feat[22],
        "facial_motion": motion,
    }
    return max(scores, key=scores.get)


def to_student_frame(
    student_id: int,
    timestamp: float,
    bbox: np.ndarray,
    keypoints: np.ndarray,
    head_pose: np.ndarray,
    gaze: np.ndarray,
    emotion_probs: np.ndarray,
    posture_probs: np.ndarray,
    motion: float,
    eye_aspect: float,
    mouth_aspect: float,
) -> StudentFrame:
    features = pack_features(
        head_pose,
        gaze,
        emotion_probs,
        posture_probs,
        motion,
        eye_aspect,
        mouth_aspect,
        keypoints,
    )
    return StudentFrame(
        student_id=student_id,
        timestamp=timestamp,
        bbox=np.asarray(bbox, dtype=np.float32),
        keypoints=np.asarray(keypoints, dtype=np.float32),
        head_pose=np.asarray(head_pose, dtype=np.float32),
        gaze=np.asarray(gaze, dtype=np.float32),
        emotion_probs=np.asarray(emotion_probs, dtype=np.float32),
        posture_probs=np.asarray(posture_probs, dtype=np.float32),
        motion=float(motion),
        eye_aspect=float(eye_aspect),
        mouth_aspect=float(mouth_aspect),
        features=features,
    )
