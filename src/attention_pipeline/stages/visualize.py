"""Draw high / moderate / low attention overlays on classroom frames."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# OpenCV BGR
COLOR_HIGH = (70, 175, 80)
COLOR_MOD = (0, 175, 255)
COLOR_LOW = (50, 50, 230)
COLOR_DROP = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (20, 20, 20)


def attention_band(score: float, high: float = 70.0, low: float = 40.0) -> str:
    if score >= high:
        return "HIGH"
    if score >= low:
        return "MODERATE"
    return "LOW"


def band_color(band: str) -> tuple[int, int, int]:
    if band == "HIGH":
        return COLOR_HIGH
    if band == "MODERATE":
        return COLOR_MOD
    return COLOR_LOW


def _label_bg(frame: np.ndarray, x: int, y: int, text: str, color, scale: float, thick: int) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 4
    y1 = max(y - th - pad, 0)
    x2 = min(x + tw + pad * 2, frame.shape[1] - 1)
    y2 = min(y + pad, frame.shape[0] - 1)
    cv2.rectangle(frame, (x, y1), (x2, y2), color, -1)
    cv2.putText(
        frame,
        text,
        (x + pad, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_TEXT,
        thick,
        cv2.LINE_AA,
    )


def draw_attention(
    frame: np.ndarray,
    result: dict[str, Any] | None,
    high: float = 70.0,
    low: float = 40.0,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w / 1280.0, 1.2))
    thick = 2 if w >= 900 else 1
    students = (result or {}).get("students") or []
    drop_ids = {
        e["event"].student_id
        for e in (result or {}).get("events") or []
        if e.get("event") is not None
    }

    counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
    for s in students:
        score = float(s.get("score", 0.0))
        band = str(s.get("band") or attention_band(score, high, low))
        counts[band] += 1
        color = COLOR_DROP if s["student_id"] in drop_ids else band_color(band)
        x1, y1, x2, y2 = [int(v) for v in s["bbox"]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(thick + 1, 2))

        tag = "DROP" if s["student_id"] in drop_ids else band
        cue = str(s.get("cue") or s.get("behavior") or "")
        label = f"ID {s['student_id']}  {tag}  {score:.0f}"
        ly = y1 - 8 if y1 > 36 else y2 + 22
        _label_bg(out, x1, ly, label, color, 0.5 * scale, thick)
        if cue and s["student_id"] not in drop_ids:
            _label_bg(out, x1, ly + int(18 * scale), cue, color, 0.42 * scale, thick)

        bar_w = max(x2 - x1, 40)
        bar_y = min(y2 + 6, h - 8)
        cv2.rectangle(out, (x1, bar_y), (x1 + bar_w, bar_y + 6), (40, 40, 40), -1)
        fill = int(bar_w * np.clip(score / 100.0, 0, 1))
        cv2.rectangle(out, (x1, bar_y), (x1 + fill, bar_y + 6), color, -1)

    _draw_legend(out, counts, result, high, low, scale, thick)
    return out


def _draw_legend(
    frame: np.ndarray,
    counts: dict[str, int],
    result: dict[str, Any] | None,
    high: float,
    low: float,
    scale: float,
    thick: int,
) -> None:
    ts = float((result or {}).get("timestamp") or 0.0)
    n = sum(counts.values())
    lines = [
        f"t={ts:5.1f}s   students={n}",
        f"HIGH  watching/reading/notes {counts['HIGH']}",
        f"MOD   relaxed / glancing     {counts['MODERATE']}",
        f"LOW   asleep / turned away   {counts['LOW']}",
    ]
    colors = [COLOR_TEXT, COLOR_HIGH, COLOR_MOD, COLOR_LOW]
    x, y = 12, 28
    (tw, th), _ = cv2.getTextSize(lines[2], cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, thick)
    cv2.rectangle(
        frame,
        (6, 8),
        (18 + max(tw, 240), 16 + len(lines) * 22),
        COLOR_PANEL,
        -1,
    )
    for i, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(
            frame,
            line,
            (x, y + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale,
            color,
            thick,
            cv2.LINE_AA,
        )
