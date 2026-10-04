"""Stage 2: one box per visible person, with no class-size cap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    keypoints: np.ndarray  # (17, 3)
    conf: float
    track_id: int


class DetectorTracker:
    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        conf: float = 0.08,
        iou: float = 0.5,
        device: str = "cpu",
        max_age: int = 30,
        n_init: int = 1,
        imgsz: int = 960,
        tiles: bool = True,
        person_weights: str = "yolo11n.pt",
    ) -> None:
        self.conf = conf
        self.iou = iou
        self._yolo = None
        self._person = None
        self.weights = weights
        self.person_weights = person_weights
        self.device = device
        self.max_age = max_age
        self.n_init = n_init
        self.imgsz = imgsz
        self.tiles = tiles
        self._next_id = 1
        self._prev: list[tuple[int, np.ndarray]] = []
        self._instructor: tuple[float, float] | None = None
        self._instructor_misses = 0

    def _lazy_load(self) -> None:
        if self._yolo is not None:
            return
        from ultralytics import YOLO

        self._yolo = YOLO(self.weights)
        if self.person_weights:
            try:
                self._person = YOLO(self.person_weights)
            except Exception:
                self._person = None

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """One box per visible person. The count is whoever is in this frame."""
        self._lazy_load()
        packed = self._detect_people(frame)
        frame_h = int(frame.shape[0])
        packed, self._instructor, self._instructor_misses = _drop_instructors(
            packed, frame_h, self._instructor, self._instructor_misses
        )
        if not packed:
            self._prev = []
            return []

        pairs: list[tuple[float, int, int]] = []
        for di, (box, _conf, _k) in enumerate(packed):
            for pi, (_tid, prev_box) in enumerate(self._prev):
                score = _track_score(box, prev_box)
                if score >= 0.35:
                    pairs.append((score, di, pi))
        pairs.sort(reverse=True)
        taken_det: set[int] = set()
        used_prev: set[int] = set()
        assigned: list[tuple[int, np.ndarray, float, np.ndarray]] = []
        for _score, di, pi in pairs:
            if di in taken_det or pi in used_prev:
                continue
            taken_det.add(di)
            used_prev.add(pi)
            box, conf, kpts = packed[di]
            assigned.append((self._prev[pi][0], box, conf, kpts))

        for di, (box, conf, kpts) in enumerate(packed):
            if di in taken_det:
                continue
            assigned.append((self._next_id, box, conf, kpts))
            self._next_id += 1

        self._prev = [(tid, box.copy()) for tid, box, _c, _k in assigned]
        dets: list[Detection] = []
        for tid, box, conf, kpts in assigned:
            k = kpts
            if k.ndim == 2 and k.shape[1] == 2:
                conf_col = np.ones((k.shape[0], 1), np.float32)
                k = np.concatenate([k, conf_col], axis=1)
            dets.append(
                Detection(
                    bbox_xyxy=box.astype(np.float32),
                    keypoints=k.astype(np.float32),
                    conf=float(conf),
                    track_id=int(tid),
                )
            )
        return dets

    def _detect_people(self, frame: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        h, w = frame.shape[:2]
        pose_hits = self._collect_pose(frame)
        people = _cluster_by_head(pose_hits, w)
        if self._person is not None:
            people = _add_unmatched_people(people, self._collect_person_boxes(frame), w)
        return _suppress_duplicate_boxes(people, w)

    def _regions(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        regions = [(0, 0, w, h)]
        if not self.tiles:
            return regions
        if w >= 1000:
            for y0, y1 in ((0, int(h * 0.72)), (int(h * 0.28), h)):
                for x0, x1 in ((0, int(w * 0.68)), (int(w * 0.32), w)):
                    regions.append((x0, y0, x1, y1))
        elif w >= 640:
            regions.append((0, 0, int(w * 0.62), h))
            regions.append((int(w * 0.38), 0, w, h))
        return regions

    def _collect_pose(self, frame: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        h, w = frame.shape[:2]
        found: list[tuple[np.ndarray, float, np.ndarray]] = []
        for x0, y0, x1, y1 in self._regions(h, w):
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            result = self._yolo.predict(
                crop,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )[0]
            found.extend(_shift_pose(result, x0, y0))
        return found

    def _collect_person_boxes(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        h, w = frame.shape[:2]
        found: list[tuple[np.ndarray, float]] = []
        for x0, y0, x1, y1 in self._regions(h, w):
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            result = self._person.predict(
                crop,
                conf=max(self.conf, 0.12),
                iou=self.iou,
                imgsz=self.imgsz,
                classes=[0],
                verbose=False,
                device=self.device,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = result.boxes.conf.cpu().numpy()
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            for box, conf in zip(xyxy, confs):
                if _box_area(box) < 24 * 36:
                    continue
                found.append((box, float(conf)))
        return found


def _shift_pose(result, x0: int, y0: int) -> list[tuple[np.ndarray, float, np.ndarray]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.cpu().numpy()
    if result.keypoints is not None:
        kpts = result.keypoints.data.cpu().numpy().astype(np.float32)
    else:
        kpts = np.zeros((len(xyxy), 17, 3), np.float32)
    xyxy[:, [0, 2]] += x0
    xyxy[:, [1, 3]] += y0
    kpts[:, :, 0] += x0
    kpts[:, :, 1] += y0
    out = []
    for box, conf, k in zip(xyxy, confs, kpts):
        if _box_area(box) < 18 * 28:
            continue
        out.append((box, float(conf), k))
    return out


def _cluster_by_head(
    found: list[tuple[np.ndarray, float, np.ndarray]],
    frame_w: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Merge tile duplicates that share a head. Keep neighbors whose heads differ."""
    if not found:
        return []
    heads = [_head_point(box, kpts, frame_w) for box, _c, kpts in found]
    order = sorted(range(len(found)), key=lambda i: found[i][1], reverse=True)
    used: set[int] = set()
    kept: list[tuple[np.ndarray, float, np.ndarray]] = []
    for i in order:
        if i in used:
            continue
        group = [i]
        used.add(i)
        hx, hy, hs = heads[i]
        for j in order:
            if j in used:
                continue
            jx, jy, js = heads[j]
            if _heads_match(hx, hy, hs, jx, jy, js):
                group.append(j)
                used.add(j)
        best = max(group, key=lambda idx: found[idx][1])
        _box, conf, kpts = found[best]
        merged = kpts.copy()
        for idx in group:
            other = found[idx][2]
            better = other[:, 2] > merged[:, 2]
            merged[better] = other[better]
        head = _head_point(_box, merged, frame_w)
        kept.append((_tight_box(_box, merged, head), conf, merged))
    return kept


def _add_unmatched_people(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    person_boxes: list[tuple[np.ndarray, float]],
    frame_w: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Add a person the pose model missed. Do not add a second box on someone already marked."""
    heads = [_head_point(box, kpts, frame_w) for box, _c, kpts in people]
    for box, conf in sorted(person_boxes, key=lambda item: item[1], reverse=True):
        hx, hy, hs = _person_head(box, frame_w)
        if any(_heads_match(hx, hy, hs, px, py, ps) for px, py, ps in heads):
            continue
        # Skip when this detection is a second box on someone already marked.
        covered = False
        for person_box, _c, _k in people:
            if person_box[0] <= hx <= person_box[2] and person_box[1] <= hy <= person_box[3]:
                covered = True
                break
        if covered:
            continue
        kpts = np.zeros((17, 3), np.float32)
        tight = _tight_box(box, kpts, (hx, hy, hs))
        if any(
            _duplicate_of(tight, kpts, (hx, hy, hs), person_box, person_kpts, head)
            for (person_box, _c, person_kpts), head in zip(people, heads)
        ):
            continue
        people.append((tight, conf, kpts))
        heads.append((hx, hy, hs))
    return people


def _drop_instructors(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    frame_h: int,
    remembered: tuple[float, float] | None,
    misses: int,
) -> tuple[list[tuple[np.ndarray, float, np.ndarray]], tuple[float, float] | None, int]:
    """Remove a standing instructor at the front. Seated students stay marked.

    The instructor is the person whose legs are extended and whose head sits
    clearly above the seated rows. At most two people match. A recent instructor
    head is kept unmarked for a short time if the legs are hidden for a frame.
    """
    if len(people) < 4:
        return people, remembered, misses + 1

    heads = [_head_top(box, kpts) for box, _conf, kpts in people]
    band = sorted(head[1] for head in heads)[max(0, int(0.2 * (len(heads) - 1)))]
    margin = max(50.0, 0.05 * frame_h)
    standing_above: list[int] = []
    for index, ((box, _conf, kpts), head) in enumerate(zip(people, heads)):
        if _legs_extended(kpts) and head[1] < band - margin:
            standing_above.append(index)

    # More than two matches means the pose cue is not separating an instructor.
    if len(standing_above) > 2:
        standing_above = []

    drop: set[int] = set(standing_above)
    if remembered is not None and len(standing_above) <= 2:
        reach = max(80.0, 0.06 * frame_h)
        for index, head in enumerate(heads):
            dist = ((head[0] - remembered[0]) ** 2 + (head[1] - remembered[1]) ** 2) ** 0.5
            if dist <= reach and head[1] < band - margin * 0.5:
                drop.add(index)

    if not drop:
        misses += 1
        if misses > 12:
            remembered = None
        return people, remembered, misses

    kept = [person for index, person in enumerate(people) if index not in drop]
    chosen = min(drop, key=lambda index: heads[index][1])
    remembered = heads[chosen]
    return kept, remembered, 0


def _head_top(box: np.ndarray, kpts: np.ndarray) -> tuple[float, float]:
    xs = [(float(box[0]) + float(box[2])) / 2.0]
    ys = [float(box[1])]
    for index in (0, 1, 2, 3, 4):
        if _kp_ok(kpts, index, 0.25):
            xs.append(float(kpts[index, 0]))
            ys.append(float(kpts[index, 1]))
    return sum(xs) / len(xs), min(ys)


def _legs_extended(kpts: np.ndarray) -> bool:
    """True when the thighs run downward, which seated desk rows do not show."""
    if not (_kp_ok(kpts, 11, 0.4) and _kp_ok(kpts, 12, 0.4)):
        return False
    knee_ys = [float(kpts[index, 1]) for index in (13, 14) if _kp_ok(kpts, index, 0.45)]
    if not knee_ys:
        return False
    hip = (float(kpts[11, 1]) + float(kpts[12, 1])) / 2.0
    knee = sum(knee_ys) / len(knee_ys)
    thigh = knee - hip
    if thigh < 24.0:
        return False
    if _kp_ok(kpts, 0, 0.25):
        torso = hip - float(kpts[0, 1])
    elif _kp_ok(kpts, 5, 0.3) and _kp_ok(kpts, 6, 0.3):
        shoulder = (float(kpts[5, 1]) + float(kpts[6, 1])) / 2.0
        torso = hip - shoulder
    else:
        return False
    return thigh >= 0.45 * max(torso, 8.0)


def _kp_ok(kpts: np.ndarray, index: int, thresh: float = 0.25) -> bool:
    return (
        kpts is not None
        and kpts.ndim == 2
        and kpts.shape[0] > index
        and kpts.shape[1] >= 3
        and float(kpts[index, 2]) > thresh
    )


def _head_point(box: np.ndarray, kpts: np.ndarray, frame_w: int) -> tuple[float, float, float]:
    """Head center and head size in pixels. Size comes from the face, not the body box."""
    cap = max(28.0, 0.07 * frame_w)

    def capped(size: float) -> float:
        return float(min(max(size, 12.0), cap))

    if _kp_ok(kpts, 1) and _kp_ok(kpts, 2):
        dist = _dist(kpts[1], kpts[2])
        cx = (float(kpts[1, 0]) + float(kpts[2, 0])) / 2.0
        cy = (float(kpts[1, 1]) + float(kpts[2, 1])) / 2.0
        return cx, cy, capped(dist * 2.6)
    if _kp_ok(kpts, 3) and _kp_ok(kpts, 4):
        dist = _dist(kpts[3], kpts[4])
        cx = (float(kpts[3, 0]) + float(kpts[4, 0])) / 2.0
        cy = (float(kpts[3, 1]) + float(kpts[4, 1])) / 2.0
        return cx, cy, capped(dist * 1.2)
    if _kp_ok(kpts, 0):
        size = 0.34 * max(float(box[2] - box[0]), 12.0)
        if _kp_ok(kpts, 5) and _kp_ok(kpts, 6):
            size = 0.42 * _dist(kpts[5], kpts[6])
        return float(kpts[0, 0]), float(kpts[0, 1]), capped(size)
    if _kp_ok(kpts, 5) and _kp_ok(kpts, 6):
        span = _dist(kpts[5], kpts[6])
        cx = (float(kpts[5, 0]) + float(kpts[6, 0])) / 2.0
        cy = min(float(kpts[5, 1]), float(kpts[6, 1])) - 0.35 * span
        return cx, cy, capped(0.45 * span)
    bw = max(float(box[2] - box[0]), 12.0)
    bh = max(float(box[3] - box[1]), 12.0)
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        float(box[1]) + 0.2 * bh,
        capped(0.36 * bw),
    )


def _person_head(box: np.ndarray, frame_w: int) -> tuple[float, float, float]:
    bw = max(float(box[2] - box[0]), 12.0)
    bh = max(float(box[3] - box[1]), 12.0)
    cap = max(28.0, 0.07 * frame_w)
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        float(box[1]) + 0.18 * bh,
        float(min(max(0.34 * bw, 12.0), cap)),
    )


def _heads_match(ax: float, ay: float, a_size: float, bx: float, by: float, b_size: float) -> bool:
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return dist < max(18.0, 0.85 * min(a_size, b_size))


def _suppress_duplicate_boxes(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    frame_w: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Drop a second box on the same body after boxes have been tightened."""
    kept: list[tuple[np.ndarray, float, np.ndarray]] = []
    heads: list[tuple[float, float, float]] = []
    for box, conf, kpts in sorted(people, key=lambda item: item[1], reverse=True):
        hx, hy, hs = _head_point(box, kpts, frame_w)
        duplicate = False
        for (other, _conf, other_kpts), (ox, oy, os_) in zip(kept, heads):
            if _duplicate_of(box, kpts, (hx, hy, hs), other, other_kpts, (ox, oy, os_)):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((box, conf, kpts))
        heads.append((hx, hy, hs))
    return kept


def _duplicate_of(
    box: np.ndarray,
    kpts: np.ndarray,
    head: tuple[float, float, float],
    other: np.ndarray,
    other_kpts: np.ndarray,
    other_head: tuple[float, float, float],
) -> bool:
    hx, hy, hs = head
    ox, oy, os_ = other_head
    dist = ((hx - ox) ** 2 + (hy - oy) ** 2) ** 0.5
    if dist < max(20.0, 0.9 * min(hs, os_)):
        return True
    iou = _iou(box, other)
    iomin = _iomin(box, other)
    overlap = iou >= 0.4 or iomin >= 0.55
    if overlap and dist < max(42.0, 1.3 * min(hs, os_)):
        return True
    dy = abs(hy - oy)
    if dy < 0.45 * max(hs, os_) and dist < 1.35 * max(hs, os_) and iomin >= 0.35:
        return True
    return _torso_under_face(box, kpts, head, other, other_kpts, other_head)


def _torso_under_face(
    box: np.ndarray,
    kpts: np.ndarray,
    head: tuple[float, float, float],
    other: np.ndarray,
    other_kpts: np.ndarray,
    other_head: tuple[float, float, float],
) -> bool:
    """A chest box with no face, directly under a real face, is the same person."""
    hx, hy, _hs = head
    ox, oy, _os = other_head
    dx = abs(hx - ox)
    width = max(float(box[2] - box[0]), float(other[2] - other[0]), 1.0)
    if dx >= 0.45 * width or not _stacked(box, other):
        return False
    if hy >= oy:
        lower_k, upper_k = kpts, other_kpts
    else:
        lower_k, upper_k = other_kpts, kpts
    return _face_confident(upper_k) and not _face_confident(lower_k)


def _face_confident(kpts: np.ndarray) -> bool:
    return _kp_ok(kpts, 0, 0.45) or (_kp_ok(kpts, 1, 0.35) and _kp_ok(kpts, 2, 0.35))


def _stacked(a: np.ndarray, b: np.ndarray) -> bool:
    top = max(float(a[1]), float(b[1]))
    bottom = min(float(a[3]), float(b[3]))
    gap = top - bottom
    smaller_h = min(float(a[3] - a[1]), float(b[3] - b[1]))
    return gap < 0.2 * max(smaller_h, 1.0)


def _tight_box(
    box: np.ndarray,
    kpts: np.ndarray,
    head: tuple[float, float, float],
) -> np.ndarray:
    """Upper-body box around the head so a neighbor behind is not covered."""
    cx, cy, size = head
    width = size * 1.9
    if _kp_ok(kpts, 5) and _kp_ok(kpts, 6):
        width = max(width, _dist(kpts[5], kpts[6]) * 1.05)
    top = cy - size * 0.9
    bottom = cy + size * 2.15
    left = cx - width / 2.0
    right = cx + width / 2.0
    # Keep the box inside the original detection so it cannot jump onto a neighbor.
    left = max(left, float(box[0]))
    right = min(right, float(box[2]))
    top = max(top, float(box[1]))
    bottom = min(bottom, float(box[3]))
    if right - left < 18:
        left, right = float(box[0]), float(box[2])
    if bottom - top < 24:
        top, bottom = float(box[1]), float(box[3])
    return np.array([left, top, right, bottom], np.float32)


def _iomin(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    small = min(_box_area(a), _box_area(b))
    return inter / small if small > 0 else 0.0


def _track_score(a: np.ndarray, b: np.ndarray) -> float:
    iou = _iou(a, b)
    cax = (float(a[0]) + float(a[2])) / 2.0
    cay = (float(a[1]) + float(a[3])) / 2.0
    cbx = (float(b[0]) + float(b[2])) / 2.0
    cby = (float(b[1]) + float(b[3])) / 2.0
    dist = ((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5
    side = min(
        max(float(a[2] - a[0]), 1.0),
        max(float(a[3] - a[1]), 1.0),
        max(float(b[2] - b[0]), 1.0),
        max(float(b[3] - b[1]), 1.0),
    )
    if dist < 0.5 * side:
        return 1.0
    return iou


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5)


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = _box_area(a)
    area_b = _box_area(b)
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0
