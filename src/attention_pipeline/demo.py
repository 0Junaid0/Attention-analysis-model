"""Synthetic two-student run: stages 4–6 without a classroom video."""

from __future__ import annotations

import numpy as np

from attention_pipeline.features import to_student_frame
from attention_pipeline.models.drop_detector import AttentionDropDetector, DropConfig
from attention_pipeline.stages.attention import AttentionScorer
from attention_pipeline.stages.intervention import recommend
from attention_pipeline.types import TrackState


def _one_hot(n: int, k: int) -> np.ndarray:
    v = np.zeros(n, np.float32)
    v[k] = 1.0
    return v


def _student_signals(student: int, t: float) -> dict:
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 2] = 0.9
    kpts[:, 0] = np.linspace(0.2, 0.8, 17)
    kpts[:, 1] = np.linspace(0.2, 0.9, 17)
    if student == 1:
        yaw = 4.0 + 3 * np.sin(t)
        gaze = np.array([2.0, -3.0], np.float32)
        emotion = _one_hot(7, 6)
        posture = _one_hot(6, 0)
        motion = 0.08
    else:
        dropping = t >= 4.0
        yaw = 8.0 if not dropping else 55.0 + 5 * np.sin(2 * t)
        gaze = np.array([4.0, -6.0], np.float32) if not dropping else np.array([28.0, -35.0], np.float32)
        emotion = _one_hot(7, 6) if not dropping else _one_hot(7, 4)
        posture = _one_hot(6, 0) if not dropping else _one_hot(6, 1)
        motion = 0.12 if not dropping else 0.4
        if dropping:
            kpts[0, 1] += 20
            kpts[11:13, 0] += 25
    return dict(
        head_pose=np.array([yaw, -8.0 if student == 2 and t >= 4 else -2.0, 1.0], np.float32),
        gaze=gaze,
        emotion_probs=emotion,
        posture_probs=posture,
        motion=motion,
        eye_aspect=0.28,
        mouth_aspect=0.22,
        keypoints=kpts,
        bbox=np.array([80.0 + student * 200, 60, 220.0 + student * 200, 420], np.float32),
    )


def main() -> None:
    scorer = AttentionScorer(model=None, device=None, window=16)
    scorer.use_heuristic = True
    detector = AttentionDropDetector(
        DropConfig(low_threshold=40, sustain_seconds=3, sudden_delta=20, sudden_seconds=2, cooldown_seconds=6)
    )
    tracks = {1: TrackState(1), 2: TrackState(2)}
    print("t(s)  S1  S2  events")
    for i in range(120):
        t = i / 10.0
        row = [f"{t:4.1f}"]
        for sid in (1, 2):
            sig = _student_signals(sid, t)
            obs = to_student_frame(sid, t, **sig)
            score, rubric = scorer.push(tracks[sid], obs.features, t, obs=obs)
            row.append(f"{score:5.1f}")
            event = detector.update(tracks[sid], obs.features)
            if event:
                rec = recommend(event, class_mean_score=float(np.mean([tracks[1].scores[-1], tracks[2].scores[-1]])))
                print(" ".join(row), f"  DROP student={sid} kind={event.kind} via {event.dominant_modality}")
                print(f"    {rec['explanation']}")
                print(f"    {rec['teaching_recommendation']}")
                print(f"    {rec['intervention_suggestion']}")
        if i % 10 == 0:
            print(" ".join(row))
    print("Demo finished. Train AttentionFusionNet to replace the heuristic scorer.")


if __name__ == "__main__":
    main()
