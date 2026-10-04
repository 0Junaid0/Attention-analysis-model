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
    return abs(_signed_diff(a, b))


def _signed_diff(a: float, b: float) -> float:
    """Signed yaw gap in degrees. Positive means `a` is to the right of `b`."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


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
    disengaged = _disengaged(cues)
    if disengaged is not None:
        return disengaged
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
    disengaged = _disengaged(cues)
    if disengaged is not None:
        return disengaged
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
    # Positive signed yaw is toward image-right, relative to the lesson.
    signed = yaw if lesson_yaw is None else _signed_diff(yaw, lesson_yaw)
    delta = abs(signed)
    facing_lesson = delta < 18
    glance = 18 <= delta < 34
    off_lesson = delta >= 34

    reading_tilt = 6 < pitch <= 42
    head_up = pitch <= 6
    collapsed = pitch > 32 or posture.get("leaning-over-table", 0) > 0.7
    on_desk = posture.get("leaning-over-table", 0) > 0.7
    hand_raised = posture.get("hand-raising", 0) > 0.5
    slouch = posture.get("leaning-over-table", 0) > 0.5
    upright = False
    standing = False
    nose_on_desk = False
    writing = posture.get("writing", 0) > 0.45 or posture.get("reading", 0) > 0.45
    phone = posture.get("using-phone", 0) > 0.45
    # Don't treat raw SCB "reading" as writing unless pose also shows desk work.
    if posture.get("reading", 0) > 0.45 and posture.get("writing", 0) < 0.45:
        writing = False
    if posture.get("bowing-head", 0) > 0.45 and pitch <= 32:
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
            nose_on_desk = True
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
    desk_work = bool(writing and reading_tilt and pitch <= 42)
    # A lowered head on notes stays classwork. Head-on-desk is a much deeper drop.
    head_on_table = bool((pitch > 48 or nose_on_desk) and not (desk_work and pitch <= 42 and not nose_on_desk))
    if upright and pitch <= 20 and not nose_on_desk:
        head_on_table = False
        collapsed = False
    still = motion < 0.05
    sleeping = bool(head_on_table and still and not writing)
    # A clear turn away from the lesson. Reading with the head down is not "looking away".
    turned = off_lesson and (not desk_work or delta >= 55) and (
        head_up or not reading_tilt or delta >= 55
    )
    look_left = bool(turned and signed < 0)
    look_right = bool(turned and signed > 0)
    glance_left = bool(glance and signed < 0 and not desk_work and not head_on_table)
    glance_right = bool(glance and signed > 0 and not desk_work and not head_on_table)

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
        "still": still,
        "active": motion >= 0.08,
        "look_left": look_left,
        "look_right": look_right,
        "glance_left": glance_left,
        "glance_right": glance_right,
        "head_on_table": head_on_table,
        "sleeping": sleeping,
    }


def _disengaged(cues: dict[str, bool]) -> RubricResult | None:
    """Behaviors that are not classwork. These win over a reading/writing label."""
    if cues.get("sleeping"):
        return RubricResult("LOW", 16.0, "sleeping")
    if cues.get("head_on_table"):
        return RubricResult("LOW", 20.0, "head on table")
    if cues.get("look_left"):
        return RubricResult("LOW", 30.0, "looking left")
    if cues.get("look_right"):
        return RubricResult("LOW", 30.0, "looking right")
    return None


def _decide(c: dict[str, bool]) -> RubricResult:
    # True disengagement only.
    if c["phone"]:
        return RubricResult("LOW", 22.0, "phone-like pose")
    if c.get("head_on_table") and c["still"]:
        return RubricResult("LOW", 16.0, "sleeping")
    if c.get("head_on_table"):
        return RubricResult("LOW", 20.0, "head on table")
    if c["off_lesson"] and not c["writing"] and not c["standing"]:
        side = "left" if c.get("look_left") else "right"
        return RubricResult("LOW", 32.0, f"looking {side}")

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
    if c.get("glance_left"):
        return RubricResult("MODERATE", 52.0, "glancing left")
    if c.get("glance_right"):
        return RubricResult("MODERATE", 52.0, "glancing right")
    if c["glance"]:
        return RubricResult("MODERATE", 55.0, "glancing away")
    if c["slouch"] and c["facing_lesson"]:
        return RubricResult("MODERATE", 58.0, "listening, relaxed")
    if c["facing_lesson"]:
        return RubricResult("MODERATE", 60.0, "listening")
    return RubricResult("MODERATE", 52.0, "partially engaged")


_CLASSWORK_CUES = {
    "hand raised",
    "reading",
    "writing notes",
    "reading / writing",
    "reading / taking notes",
    "reading / head to desk",
    "watching instructor",
    "standing / presenting",
}


def apply_peer_cues(
    frames: list[StudentFrame],
    rubrics: list[RubricResult],
    lesson_yaw: float | None,
) -> list[RubricResult]:
    """Mark same-row students who turn toward each other.

    Turning toward the lesson is not talking, even on a side camera.
    A hand that reaches into a neighbor's space, or an active turn toward
    them, is disturbing.
    """
    updated = list(rubrics)
    talking: set[int] = set()
    disturbing: set[int] = set()
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            relation = _row_relation(frames[i], frames[j])
            if relation is None:
                continue
            side_i, close = relation
            i_looks = _faces_side(frames[i], lesson_yaw, side_i)
            j_looks = _faces_side(frames[j], lesson_yaw, -side_i)
            i_reaches = _reaches(frames[i], frames[j])
            j_reaches = _reaches(frames[j], frames[i])
            i_active = float(frames[i].motion) >= 0.12
            j_active = float(frames[j].motion) >= 0.12
            if i_looks and (j_looks or close):
                talking.add(i)
            if j_looks and (i_looks or close):
                talking.add(j)
            if i_reaches and (i_looks or i_active):
                disturbing.add(i)
            if j_reaches and (j_looks or j_active):
                disturbing.add(j)
            if i_looks and j_looks and i_active:
                disturbing.add(i)
            if i_looks and j_looks and j_active:
                disturbing.add(j)

    for index, rubric in enumerate(updated):
        if rubric.cue in {"sleeping", "head on table"}:
            continue
        if index in disturbing and rubric.cue != "hand raised":
            updated[index] = RubricResult("LOW", 18.0, "disturbing others")
        elif index in talking and rubric.cue not in _CLASSWORK_CUES:
            updated[index] = RubricResult("LOW", 28.0, "talking")
    return updated


def _center(obs: StudentFrame) -> tuple[float, float]:
    kpts = obs.keypoints
    if kpts is not None and kpts.shape[0] > 0 and float(kpts[0, 2]) >= 0.25:
        return float(kpts[0, 0]), float(kpts[0, 1])
    box = obs.bbox
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def _row_relation(a: StudentFrame, b: StudentFrame) -> tuple[float, bool] | None:
    ax, ay = _center(a)
    bx, by = _center(b)
    width = (
        max(float(a.bbox[2] - a.bbox[0]), 1.0) + max(float(b.bbox[2] - b.bbox[0]), 1.0)
    ) / 2.0
    height = (
        max(float(a.bbox[3] - a.bbox[1]), 1.0) + max(float(b.bbox[3] - b.bbox[1]), 1.0)
    ) / 2.0
    dx = bx - ax
    if abs(by - ay) > 0.55 * height:
        return None
    gap = abs(dx)
    if gap < 0.35 * width or gap > 2.2 * width:
        return None
    return (1.0 if dx > 0 else -1.0), gap < 1.35 * width


def _faces_side(obs: StudentFrame, lesson_yaw: float | None, side: float) -> bool:
    yaw = float(obs.head_pose[0])
    signed = yaw if lesson_yaw is None else _signed_diff(yaw, lesson_yaw)
    return signed * side > 0 and abs(signed) >= 22.0


def _reaches(obs: StudentFrame, other: StudentFrame) -> bool:
    kpts = obs.keypoints
    if kpts is None or kpts.shape[0] < 11:
        return False
    box = np.asarray(other.bbox, dtype=np.float32)
    pad_x = 0.08 * max(float(box[2] - box[0]), 1.0)
    pad_y = 0.08 * max(float(box[3] - box[1]), 1.0)
    for index in (7, 8, 9, 10):
        if float(kpts[index, 2]) < 0.35:
            continue
        x, y = float(kpts[index, 0]), float(kpts[index, 1])
        if (float(box[0]) - pad_x) <= x <= (float(box[2]) + pad_x) and (
            float(box[1]) - pad_y
        ) <= y <= (float(box[3]) + pad_y):
            return True
    return False
