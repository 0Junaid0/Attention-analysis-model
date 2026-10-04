"""Classroom attention rubric: HIGH / MODERATE / LOW from posture, face, and action cues."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attention_pipeline.types import POSTURE_LABELS, StudentFrame

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SH, R_SH = 5, 6
L_WR, R_WR = 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class RubricResult:
    band: str
    score: float
    cue: str


def _angle_diff(a: float, b: float) -> float:
    d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(d)


def classify_attention(obs: StudentFrame, lesson_yaw: float | None = None) -> RubricResult:
    """Score attention relative to the lesson, not the camera.

    Side-classroom cameras make 'facing the teacher' look like a turned head.
    Reading and writing look like a lowered head; that is HIGH, not LOW.
    """
    yaw = float(obs.head_pose[0])
    pitch = float(obs.head_pose[1])
    motion = float(obs.motion)
    posture = {name: float(obs.posture_probs[i]) for i, name in enumerate(POSTURE_LABELS)}
    cues = _pose_cues(obs.keypoints, obs.bbox, yaw, pitch, motion, posture, lesson_yaw)
    from attention_pipeline.stages.scb_behavior import scb_attention_override

    idx = int(np.argmax(obs.posture_probs)) if obs.posture_probs.size else 0
    conf = float(obs.posture_probs[idx]) if obs.posture_probs.size else 0.0
    if conf >= 0.45:
        ov = scb_attention_override(POSTURE_LABELS[idx], cues=cues, conf=conf)
        if ov is not None:
            band, score, cue = ov
            return RubricResult(band, score, cue)
    return _decide(cues)


def classify_from_features(feat: np.ndarray, lesson_yaw: float | None = None) -> RubricResult:
    yaw = float(feat[0] * 180.0)
    pitch = float(feat[1] * 180.0)
    motion = float(feat[18])
    posture = {name: float(feat[12 + i]) for i, name in enumerate(POSTURE_LABELS)}
    cues = _pose_cues(None, None, yaw, pitch, motion, posture, lesson_yaw)
    from attention_pipeline.stages.scb_behavior import scb_attention_override

    idx = int(np.argmax([posture.get(n, 0.0) for n in POSTURE_LABELS]))
    conf = posture.get(POSTURE_LABELS[idx], 0.0)
    if conf >= 0.45:
        ov = scb_attention_override(POSTURE_LABELS[idx], cues=cues, conf=conf)
        if ov is not None:
            band, score, cue = ov
            return RubricResult(band, score, cue)
    return _decide(cues)


def _ok(kpts: np.ndarray, idx: int, thr: float = 0.2) -> bool:
    return kpts is not None and kpts.shape[0] > idx and float(kpts[idx, 2]) >= thr


def _pose_cues(kpts, bbox, yaw, pitch, motion, posture, lesson_yaw) -> dict[str, bool]:
    # Camera-relative facing is only a fallback. Lesson direction is the class median yaw.
    if lesson_yaw is None:
        facing_lesson = abs(yaw) < 50
        off_lesson = abs(yaw) >= 75
        glance = 50 <= abs(yaw) < 75
    else:
        delta = _angle_diff(yaw, lesson_yaw)
        facing_lesson = delta < 40
        glance = 40 <= delta < 65
        off_lesson = delta >= 65

    reading_tilt = 6 < pitch <= 32
    head_up = pitch <= 6
    collapsed = pitch > 32 or posture.get("leaning-over-table", 0) > 0.7
    on_desk = posture.get("leaning-over-table", 0) > 0.7
    hand_raised = posture.get("hand-raising", 0) > 0.5
    slouch = posture.get("leaning-over-table", 0) > 0.5
    upright = False
    standing = False
    writing = posture.get("writing", 0) > 0.45 or posture.get("reading", 0) > 0.45
    phone = posture.get("using-phone", 0) > 0.45
    # Don't treat raw SCB "reading" as writing unless pose also shows desk work.
    if posture.get("reading", 0) > 0.45 and posture.get("writing", 0) < 0.45:
        writing = False
    if posture.get("bowing-head", 0) > 0.45:
        reading_tilt = True

    usable = (
        kpts is not None
        and bbox is not None
        and float(np.max(kpts[:, :2])) > 2.0
        and _ok(kpts, NOSE)
        and _ok(kpts, L_SH)
        and _ok(kpts, R_SH)
    )
    if usable:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bh = max(y2 - y1, 1.0)
        bw = max(x2 - x1, 1.0)
        aspect = bh / bw
        nose = kpts[NOSE, :2]
        sh = (kpts[L_SH, :2] + kpts[R_SH, :2]) / 2.0
        hip = None
        if _ok(kpts, L_HIP) and _ok(kpts, R_HIP):
            hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2.0
        head_clearance = (sh[1] - nose[1]) / bh
        # Tighter desk-work band; side views often have mid clearance while watching.
        if 0.045 <= head_clearance <= 0.11 and pitch > 8:
            reading_tilt = True
        if head_clearance > 0.12:
            upright = True
            collapsed = False
            reading_tilt = False if pitch <= 8 else reading_tilt
        if head_clearance < 0.0:
            on_desk = True
            collapsed = True
        if hip is not None:
            torso = (hip[1] - sh[1]) / bh
            if 0.16 <= torso < 0.24:
                slouch = True
            if torso < 0.12 and head_clearance < 0.04:
                on_desk = True
                collapsed = True
            # Standing: long vertical box + visible hips farther down than seated desk row.
            if aspect > 2.0 and torso > 0.28 and upright:
                standing = True
        elif aspect > 2.2 and upright:
            standing = True
        wr_l = kpts[L_WR] if _ok(kpts, L_WR) else None
        wr_r = kpts[R_WR] if _ok(kpts, R_WR) else None
        if wr_l is not None and wr_l[1] < sh[1] - 0.08 * bh:
            hand_raised = True
        if wr_r is not None and wr_r[1] < sh[1] - 0.08 * bh:
            hand_raised = True
        wrists = [w for w in (wr_l, wr_r) if w is not None]
        if wrists and hip is not None:
            mid_torso_y = (sh[1] + hip[1]) / 2.0
            near_desk = all(float(w[1]) > mid_torso_y for w in wrists)
            together = False
            if wr_l is not None and wr_r is not None:
                together = abs(float(wr_l[0]) - float(wr_r[0])) < 0.22 * bw
            if near_desk and not collapsed and reading_tilt:
                writing = True
            if collapsed and together and near_desk:
                phone = True

    # Desk work from wrists near desk + head tilt (not tilt alone).
    # Side cameras make "watching teacher" look like a small pitch; that is not writing.

    return {
        "facing_lesson": facing_lesson,
        "glance": glance,
        "off_lesson": off_lesson,
        "head_up": head_up,
        "reading_tilt": reading_tilt,
        "collapsed": collapsed,
        "on_desk": on_desk,
        "upright": upright,
        "slouch": slouch,
        "hand_raised": hand_raised,
        "writing": writing,
        "phone": phone,
        "standing": standing,
        "still": motion < 0.05,
        "active": motion >= 0.08,
    }


def _decide(c: dict[str, bool]) -> RubricResult:
    # True disengagement only.
    if c["on_desk"] and c["still"]:
        return RubricResult("LOW", 18.0, "sleeping / head on desk")
    if c["phone"]:
        return RubricResult("LOW", 22.0, "phone-like pose")
    if c["collapsed"] and c["still"]:
        return RubricResult("LOW", 26.0, "collapsed on desk")
    if c["off_lesson"] and not c["writing"] and not c["standing"]:
        return RubricResult("LOW", 32.0, "turned away from lesson")

    # Active learning: notes, reading, facing the teacher, hand raise.
    if c["hand_raised"]:
        return RubricResult("HIGH", 92.0, "hand raised")
    if c["standing"] and c["facing_lesson"]:
        return RubricResult("HIGH", 88.0, "standing / presenting")
    if c["writing"] and not c["off_lesson"]:
        return RubricResult("HIGH", 88.0, "reading / writing")
    if c["facing_lesson"] and c["head_up"]:
        return RubricResult("HIGH", 86.0, "watching instructor")
    if c["facing_lesson"] and c["reading_tilt"]:
        return RubricResult("HIGH", 84.0, "reading / taking notes")
    if c["facing_lesson"] and not c["slouch"]:
        return RubricResult("HIGH", 80.0, "oriented to lesson")

    # Partial engagement.
    if c["glance"]:
        return RubricResult("MODERATE", 55.0, "glancing away")
    if c["slouch"] and c["facing_lesson"]:
        return RubricResult("MODERATE", 58.0, "listening, relaxed")
    if c["facing_lesson"]:
        return RubricResult("MODERATE", 60.0, "listening")
    return RubricResult("MODERATE", 52.0, "partially engaged")
