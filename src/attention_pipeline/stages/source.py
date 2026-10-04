"""Stage 1: classroom video → timestamped frames."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def iter_frames(
    source: str | int,
    target_fps: float = 10.0,
) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(source if not isinstance(source, Path) else str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {source}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(native / target_fps, 1.0)
    index = 0
    emitted = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= emitted:
                ts = index / native
                yield ts, frame
                emitted += interval
            index += 1
    finally:
        cap.release()
