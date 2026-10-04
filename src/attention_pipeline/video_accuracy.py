"""Accuracy counted from the students in one input video.

The percentage is recomputed from that video's scored rows. Nothing here
reads a saved dataset accuracy.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from attention_pipeline.types import POSTURE_LABELS

BANDS = ("HIGH", "MODERATE", "LOW")
_HIGH_BEHAVIOR = {"reading", "writing", "hand-raising"}
_LOW_BEHAVIOR = {"using-phone", "leaning-over-table"}


def _signed(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def check_band(student: dict, lesson_yaw: float) -> str:
    """Independent HIGH / MODERATE / LOW check from pose and a confident SCB class."""
    pose = student.get("head_pose") or [0.0, 0.0, 0.0]
    yaw = float(pose[0]) if len(pose) > 0 else 0.0
    pitch = float(pose[1]) if len(pose) > 1 else 0.0
    delta = abs(_signed(yaw, lesson_yaw))
    probs = [float(v) for v in (student.get("posture") or [])]
    name = ""
    conf = 0.0
    if probs:
        index = max(range(len(probs)), key=lambda i: probs[i])
        conf = probs[index]
        if index < len(POSTURE_LABELS):
            name = POSTURE_LABELS[index]

    if pitch > 48.0:
        return "LOW"
    if conf >= 0.55 and name in _LOW_BEHAVIOR:
        return "LOW"
    if conf >= 0.55 and name == "bowing-head" and pitch > 42.0:
        return "LOW"
    if delta >= 34.0 and pitch <= 24.0:
        return "LOW"
    if conf >= 0.55 and name in _HIGH_BEHAVIOR and delta < 34.0:
        return "HIGH"
    if delta < 18.0 and pitch <= 42.0:
        return "HIGH"
    return "MODERATE"


def _lesson_yaw(students: list[dict]) -> float:
    yaws = []
    for student in students:
        pose = student.get("head_pose") or []
        if pose:
            yaws.append(float(pose[0]))
    if not yaws:
        return 0.0
    yaws.sort()
    return yaws[len(yaws) // 2]


def measure_rows(rows: list[dict]) -> dict:
    """Precision, recall, F1, and accuracy of the overlay against the check band."""
    y_true: list[str] = []
    y_pred: list[str] = []
    frames = 0
    for row in rows:
        students = row.get("students") or []
        if not students:
            frames += 1
            continue
        frames += 1
        lesson = _lesson_yaw(students)
        for student in students:
            pred = str(student.get("band") or "")
            if pred not in BANDS:
                continue
            y_true.append(check_band(student, lesson))
            y_pred.append(pred)

    per_band = {}
    for band in BANDS:
        tp = sum(t == band and p == band for t, p in zip(y_true, y_pred))
        fp = sum(t != band and p == band for t, p in zip(y_true, y_pred))
        fn = sum(t == band and p != band for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_band[band] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(t == band for t in y_true),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    support = len(y_true)
    accuracy = (
        sum(t == p for t, p in zip(y_true, y_pred)) / support if support else 0.0
    )
    counted = [per_band[band] for band in BANDS if per_band[band]["support"]]
    if not counted:
        counted = [per_band[band] for band in BANDS]
    macro = {
        "precision": sum(item["precision"] for item in counted) / len(counted),
        "recall": sum(item["recall"] for item in counted) / len(counted),
        "f1": sum(item["f1"] for item in counted) / len(counted),
    }
    return {
        "frames": frames,
        "support": support,
        "accuracy": accuracy,
        "macro": macro,
        "per_band": per_band,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def unique_report_path(video: Path, folder: Path | None = None) -> Path:
    """A new report path. An existing file is never replaced."""
    root = folder or Path("runs") / "reports"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{video.stem}_{stamp}_accuracy.txt"
    suffix = 2
    while path.exists():
        path = root / f"{video.stem}_{stamp}_{suffix}_accuracy.txt"
        suffix += 1
    return path


def load_model_metrics(path: Path | None = None) -> dict | None:
    """Saved validation scores for the model. This file does not change per video."""
    metrics_path = path or Path(__file__).resolve().parents[2] / "runs" / "smoke_metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def model_lines(data: dict | None) -> list[str]:
    lines = [
        "=" * 72,
        "MODEL ACCURACY",
        "=" * 72,
        "",
        "Measured on the saved validation set. This stays the same for every video",
        "until the model is evaluated again.",
        "",
    ]
    overall = (data or {}).get("overall") or {}
    if "accuracy" not in overall:
        lines.append("Model accuracy is not available (runs/smoke_metrics.json).")
        return lines
    labels = (
        ("SCB official YOLO", "scb_official_yolo"),
        ("DAiSEE fusion", "daisee_fusion"),
        ("L2CS gaze", "l2cs_gaze"),
    )
    lines.append(f"{'Component':<28} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Acc':>8}")
    lines.append("-" * 72)
    for label, key in labels:
        item = (data or {}).get(key) or {}
        if "accuracy" not in item:
            continue
        lines.append(
            f"{label:<28} {item['precision']*100:7.2f}% {item['recall']*100:7.2f}% "
            f"{item['f1']*100:7.2f}% {item['accuracy']*100:7.2f}%"
        )
    lines.extend(
        [
            "-" * 72,
            f"{'OVERALL MODEL':<28} {overall['precision']*100:7.2f}% "
            f"{overall['recall']*100:7.2f}% {overall['f1']*100:7.2f}% "
            f"{overall['accuracy']*100:7.2f}%",
            "",
            f"MODEL ACCURACY: {overall['accuracy']*100:.2f}%",
        ]
    )
    return lines


def render_report(
    video: Path,
    overlay: Path,
    scores: Path,
    metrics: dict,
) -> str:
    macro = metrics["macro"]
    lines = [
        "=" * 72,
        "VIDEO ACCURACY REPORT",
        "=" * 72,
        "",
        "Model accuracy is the saved validation score.",
        "Video accuracy is counted from the marked students in this input video.",
        "",
        f"Input video : {video.resolve()}",
        f"Overlay     : {overlay.resolve()}",
        f"Scores      : {scores.resolve()}",
        f"Frames scored     : {metrics['frames']}",
        f"Students compared : {metrics['support']}",
        "",
    ]
    lines.extend(model_lines(load_model_metrics()))
    lines.extend(
        [
            "",
            "=" * 72,
            "VIDEO ACCURACY",
            "=" * 72,
            "",
            "Counted from the marked students in this input video.",
            "A different video produces a different video accuracy.",
            "",
            f"{'Band':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}",
            "-" * 72,
        ]
    )
    for band in BANDS:
        item = metrics["per_band"][band]
        lines.append(
            f"{band:<12} {item['precision']*100:7.2f}% {item['recall']*100:7.2f}% "
            f"{item['f1']*100:7.2f}% {item['support']:8d}"
        )
    lines.extend(
        [
            "-" * 72,
            f"{'MACRO':<12} {macro['precision']*100:7.2f}% {macro['recall']*100:7.2f}% "
            f"{macro['f1']*100:7.2f}%",
            "",
            f"VIDEO ACCURACY: {metrics['accuracy']*100:.2f}%",
            "",
            "Accuracy is the share of marked students whose overlay band matches",
            "the check. Precision, recall, and F1 are per band, then averaged.",
            "=" * 72,
        ]
    )
    return "\n".join(lines) + "\n"


def write_video_accuracy(
    video: Path,
    overlay: Path,
    scores: Path,
    folder: Path | None = None,
) -> Path:
    metrics = measure_rows(load_jsonl(scores))
    path = unique_report_path(video, folder)
    path.write_text(render_report(video, overlay, scores, metrics), encoding="utf-8")
    return path
