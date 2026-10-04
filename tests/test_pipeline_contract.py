from __future__ import annotations

import numpy as np

from attention_pipeline.features import heuristic_attention_score, pack_features
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.drop_detector import AttentionDropDetector, DropConfig
from attention_pipeline.types import FEAT_DIM, TrackState


def _feat(yaw: float, looking_down: bool) -> np.ndarray:
    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    posture = np.zeros(6, np.float32)
    posture[1 if looking_down else 0] = 1.0
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 2] = 1.0
    return pack_features(
        head_pose=np.array([yaw, 0.0, 0.0], np.float32),
        gaze=np.array([0.0, -30.0 if looking_down else 0.0], np.float32),
        emotion_probs=emotion,
        posture_probs=posture,
        motion=0.5 if looking_down else 0.1,
        eye_aspect=0.3,
        mouth_aspect=0.2,
        keypoints=kpts,
    )


def test_feature_dim() -> None:
    f = _feat(0, False)
    assert f.shape == (FEAT_DIM,)
    assert 80 <= heuristic_attention_score(f) <= 100
    reading = _feat(10, True)
    assert heuristic_attention_score(reading) >= 70
    asleep = pack_features(
        head_pose=np.array([0.0, 40.0, 0.0], np.float32),
        gaze=np.array([0.0, 40.0], np.float32),
        emotion_probs=np.array([0, 0, 0, 0, 0, 0, 1], np.float32),
        posture_probs=np.array([0, 0, 0, 0, 0, 1], np.float32),
        motion=0.01,
        eye_aspect=0.2,
        mouth_aspect=0.2,
        keypoints=np.zeros((17, 3), np.float32),
    )
    assert heuristic_attention_score(asleep) < 40


def test_scb_behavior_match_and_override() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention
    from attention_pipeline.stages.scb_behavior import SCBBehaviorDetector, SCBDet
    from attention_pipeline.types import SCB_BEHAVIOR_LABELS

    det = SCBBehaviorDetector(weights=None)
    box = np.array([10, 10, 110, 210], np.float32)
    dets = [
        SCBDet(bbox_xyxy=box, cls=1, name="reading", conf=0.9),
        SCBDet(
            bbox_xyxy=np.array([500, 500, 600, 700], np.float32),
            cls=3,
            name="using-phone",
            conf=0.99,
        ),
    ]
    probs = det.match_probs(box, dets)
    assert probs is not None
    assert SCB_BEHAVIOR_LABELS[int(np.argmax(probs))] == "reading"

    emotion = np.zeros(7, np.float32)
    emotion[6] = 1.0
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 2] = 1.0

    def frame(idx: int, pitch: float = 16.0):
        posture = np.zeros(6, np.float32)
        posture[idx] = 1.0
        return to_student_frame(
            1,
            0.0,
            bbox=np.array([0, 0, 100, 200], np.float32),
            keypoints=kpts,
            head_pose=np.array([8.0, pitch, 0], np.float32),
            gaze=np.array([8.0, pitch], np.float32),
            emotion_probs=emotion,
            posture_probs=posture,
            motion=0.1,
            eye_aspect=0.3,
            mouth_aspect=0.2,
        )

    assert classify_attention(frame(1)).band == "HIGH"
    assert classify_attention(frame(2)).band == "HIGH"
    assert classify_attention(frame(3)).band == "LOW"
    assert classify_attention(frame(3)).cue == "using phone"


def test_classroom_rubric() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import classify_attention

    def frame(yaw, pitch, posture_idx, motion=0.1):
        emotion = np.zeros(7, np.float32)
        emotion[6] = 1.0
        posture = np.zeros(6, np.float32)
        posture[posture_idx] = 1.0
        kpts = np.zeros((17, 3), np.float32)
        kpts[:, 2] = 1.0
        return to_student_frame(
            1,
            0.0,
            bbox=np.array([0, 0, 100, 200], np.float32),
            keypoints=kpts,
            head_pose=np.array([yaw, pitch, 0], np.float32),
            gaze=np.array([yaw, pitch], np.float32),
            emotion_probs=emotion,
            posture_probs=posture,
            motion=motion,
            eye_aspect=0.3,
            mouth_aspect=0.2,
        )

    high = classify_attention(frame(4, 0, 0))
    assert high.band == "HIGH"
    reading = classify_attention(frame(8, 16, 0))
    assert reading.band == "HIGH"
    watching = classify_attention(frame(40, 2, 0), lesson_yaw=42.0)
    assert watching.band == "HIGH"
    low = classify_attention(frame(55, 40, 5, motion=0.02), lesson_yaw=0.0)
    assert low.band == "LOW"


def test_geometric_posture_keypoints() -> None:
    from attention_pipeline.stages.modalities import geometric_posture

    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 1.0
    k[:, 0] = np.linspace(10, 90, 17)
    k[:, 1] = np.linspace(20, 180, 17)
    probs = geometric_posture(k)
    assert probs.shape == (6,)
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_fusion_forward() -> None:
    net = AttentionFusionNet()
    x = torch_window()
    score, aux = net(x)
    assert score.shape == (2,)
    assert aux.shape == (2, 4)
    assert float(score.detach().min()) >= 0 and float(score.detach().max()) <= 100


def torch_window():
    import torch

    return torch.zeros(2, 16, FEAT_DIM)


def test_drop_sustained() -> None:
    det = AttentionDropDetector(
        DropConfig(low_threshold=40, sustain_seconds=3, cooldown_seconds=0, ema_alpha=1.0)
    )
    track = TrackState(student_id=7)
    feat = _feat(70, True)
    event = None
    for i in range(40):
        track.scores.append(20.0)
        track.timestamps.append(i * 0.1)
        event = det.update(track, feat)
    assert event is not None
    assert event.kind == "sustained"
    assert event.student_id == 7


def test_attention_bands_and_overlay() -> None:
    import numpy as np

    from attention_pipeline.stages.visualize import attention_band, draw_attention

    assert attention_band(82) == "HIGH"
    assert attention_band(55) == "MODERATE"
    assert attention_band(20) == "LOW"
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = draw_attention(
        frame,
        {
            "timestamp": 1.2,
            "students": [
                {"student_id": 1, "bbox": [40, 40, 160, 280], "score": 81.0},
                {"student_id": 2, "bbox": [200, 50, 320, 300], "score": 22.0},
            ],
            "events": [],
        },
    )
    assert out.shape == frame.shape
    assert out.sum() > 0
