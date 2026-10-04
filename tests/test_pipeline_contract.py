from __future__ import annotations

from pathlib import Path

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
        head_pose=np.array([0.0, 58.0, 0.0], np.float32),
        gaze=np.array([0.0, 58.0], np.float32),
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


def test_looking_sleeping_and_peer_talk() -> None:
    from attention_pipeline.features import to_student_frame
    from attention_pipeline.rubric import apply_peer_cues, classify_attention

    def frame(yaw, pitch, motion=0.1, bbox=None, keypoints=None):
        emotion = np.zeros(7, np.float32)
        emotion[6] = 1.0
        posture = np.full(6, 0.1, np.float32)
        kpts = np.zeros((17, 3), np.float32) if keypoints is None else keypoints
        box = np.array([0, 0, 100, 180], np.float32) if bbox is None else bbox
        return to_student_frame(
            1,
            0.0,
            bbox=box,
            keypoints=kpts,
            head_pose=np.array([yaw, pitch, 0], np.float32),
            gaze=np.array([yaw, pitch], np.float32),
            emotion_probs=emotion,
            posture_probs=posture,
            motion=motion,
            eye_aspect=0.3,
            mouth_aspect=0.2,
        )

    assert classify_attention(frame(-50, 2), lesson_yaw=0.0).cue == "looking left"
    assert classify_attention(frame(50, 2), lesson_yaw=0.0).cue == "looking right"
    assert classify_attention(frame(0, 58, motion=0.01), lesson_yaw=0.0).cue == "sleeping"
    assert classify_attention(frame(0, 55, motion=0.2), lesson_yaw=0.0).cue == "head on table"
    # Facing the lesson, even if the class looks to the right, is not talking.
    lesson = frame(30, 2, bbox=np.array([0, 40, 80, 200], np.float32))
    neighbor = frame(30, 2, bbox=np.array([120, 40, 200, 200], np.float32))
    kept = apply_peer_cues(
        [lesson, neighbor],
        [classify_attention(lesson, lesson_yaw=30.0), classify_attention(neighbor, lesson_yaw=30.0)],
        30.0,
    )
    assert all(item.cue != "talking" for item in kept)

    left = frame(40, 2, bbox=np.array([0, 40, 90, 210], np.float32))
    right = frame(-40, 2, bbox=np.array([110, 40, 200, 210], np.float32))
    talked = apply_peer_cues(
        [left, right],
        [classify_attention(left, lesson_yaw=0.0), classify_attention(right, lesson_yaw=0.0)],
        0.0,
    )
    assert talked[0].cue == "talking"
    assert talked[1].cue == "talking"

    kpts = np.zeros((17, 3), np.float32)
    kpts[9] = [150, 80, 1.0]  # wrist inside the neighbor box
    kpts[0, 2] = 0.0
    reacher = frame(40, 2, motion=0.2, bbox=np.array([0, 40, 90, 210], np.float32), keypoints=kpts)
    quiet = frame(0, 2, bbox=np.array([110, 40, 200, 210], np.float32))
    disturbed = apply_peer_cues(
        [reacher, quiet],
        [classify_attention(reacher, lesson_yaw=0.0), classify_attention(quiet, lesson_yaw=0.0)],
        0.0,
    )
    assert disturbed[0].cue == "disturbing others"
    assert disturbed[1].cue != "disturbing others"


def test_video_accuracy_is_a_new_file() -> None:
    from attention_pipeline.video_accuracy import measure_rows, unique_report_path, write_video_accuracy

    high = {
        "band": "HIGH",
        "head_pose": [0.0, 10.0, 0.0],
        "posture": [0.0, 0.9, 0.05, 0.0, 0.0, 0.05],
    }
    low = {
        "band": "LOW",
        "head_pose": [0.0, 60.0, 0.0],
        "posture": [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    }
    mismatch = {
        "band": "HIGH",
        "head_pose": [0.0, 70.0, 0.0],
        "posture": [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    }
    metrics = measure_rows([{"students": [high, low, mismatch]}])
    assert metrics["support"] == 3
    assert abs(metrics["accuracy"] - (2 / 3)) < 1e-6
    assert metrics["per_band"]["LOW"]["support"] == 2

    folder = Path("runs") / "_accuracy_test"
    folder.mkdir(parents=True, exist_ok=True)
    video = Path("video") / "Classroom_video.mp4"
    first = unique_report_path(video, folder)
    first.write_text("keep\n", encoding="utf-8")
    second = unique_report_path(video, folder)
    assert first != second
    assert first.read_text(encoding="utf-8") == "keep\n"
    scores = folder / "scores.jsonl"
    scores.write_text(
        '{"students": ['
        + '{"band": "HIGH", "head_pose": [0, 10, 0], "posture": [0, 0.9, 0, 0, 0, 0.1]}'
        + "]}\n",
        encoding="utf-8",
    )
    written = write_video_accuracy(video, Path("runs/overlay.mp4"), scores, folder)
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert "MODEL ACCURACY" in text
    assert "VIDEO ACCURACY" in text
    assert first.read_text(encoding="utf-8") == "keep\n"


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


def _person(x: float, head_y: float, thigh: float) -> tuple:
    kpts = np.zeros((17, 3), np.float32)
    kpts[0] = [x, head_y + 28, 1.0]
    kpts[5] = [x - 18, head_y + 70, 1.0]
    kpts[6] = [x + 18, head_y + 70, 1.0]
    hip = head_y + 150
    kpts[11] = [x - 12, hip, 1.0]
    kpts[12] = [x + 12, hip, 1.0]
    kpts[13] = [x - 12, hip + thigh, 1.0]
    kpts[14] = [x + 12, hip + thigh, 1.0]
    box = np.array([x - 36, head_y, x + 36, head_y + 130], np.float32)
    return box, 0.8, kpts


def test_standing_instructor_is_not_marked() -> None:
    from attention_pipeline.stages.detect_track import _drop_instructors

    students = [_person(120 + i * 90, 280, 36) for i in range(8)]
    instructor = _person(1500, 90, 120)
    kept, remembered, misses = _drop_instructors(students + [instructor], 1080, None, 0)
    assert len(kept) == 8
    assert remembered is not None
    assert remembered[1] < 150
    assert misses == 0
    # A seated person with long visible legs stays, because their head is in the rows.
    front = _person(400, 520, 90)
    kept2, _, _ = _drop_instructors(students + [front], 1080, None, 0)
    assert len(kept2) == 9
