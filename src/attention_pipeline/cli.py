"""Run the pipeline on a classroom video and write an annotated overlay video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.stages.visualize import attention_band, draw_attention


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    candidates = [
        (path.with_suffix(".mp4"), "mp4v"),
        (path.with_suffix(".avi"), "XVID"),
        (path.with_suffix(".avi"), "MJPG"),
    ]
    for out, codec in candidates:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            return writer, out
        writer.release()
    raise SystemExit("Could not open a video writer. Try another folder or install codecs.")


def _dump_result(result: dict, high: float, low: float) -> dict:
    return {
        "timestamp": result["timestamp"],
        "students": [
            {
                **s,
                "band": s.get("band") or attention_band(float(s["score"]), high, low),
            }
            for s in result["students"]
        ],
        "events": [
            {
                "student_id": e["event"].student_id,
                "timestamp": e["event"].timestamp,
                "score": e["event"].score,
                "kind": e["event"].kind,
                "dominant_modality": e["event"].dominant_modality,
                "recommendation": {
                    k: v for k, v in (e["recommendation"] or {}).items() if k != "prompt"
                },
            }
            for e in result["events"]
        ],
    }


def infer_main() -> None:
    parser = argparse.ArgumentParser(description="Classroom attention overlay video")
    parser.add_argument("video", help="Path to your classroom video")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--out", default="runs/inference.jsonl")
    parser.add_argument(
        "--out-video",
        default=None,
        help="Annotated video path (default: runs/attention_overlay.mp4)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Only process the first N seconds (try 20 for a quick test)",
    )
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"Video not found: {video.resolve()}")

    pipe = AttentionPipeline.from_yaml(args.config)
    status = pipe.dataset_status
    print(
        "Weights: "
        f"L2CS={'on' if status['l2cs_gaze'] else 'off'}  "
        f"SCB={'on' if status['scb_behavior'] else 'off'}  "
        f"SCB-data={'on' if status['scb_yaml'] else 'off'}"
    )
    if not status["l2cs_gaze"] or not status["scb_behavior"]:
        print("Missing weights. Run: python scripts/download_datasets.py")
    viz_cfg = pipe.cfg.get("viz", {})
    high = float(viz_cfg.get("high", 70.0))
    low = float(viz_cfg.get("moderate_low", 40.0))
    out_video = Path(args.out_video or viz_cfg.get("out_video", "runs/attention_overlay.mp4"))
    json_path = Path(args.out)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video.resolve()}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    infer_fps = float(pipe.cfg.get("fps", 10))
    infer_dt = 1.0 / max(infer_fps, 1.0)

    writer, written_path = _open_writer(out_video, native_fps, (width, height))

    print(f"Input:  {video.resolve()}")
    print(f"Output: {written_path.resolve()}")
    print("Boxes: GREEN=HIGH (focused)  ORANGE=MODERATE  RED=LOW (distracted)")
    print("First run may download YOLO11-Pose weights.")

    last_result: dict = {"timestamp": 0.0, "students": [], "events": []}
    next_infer = 0.0
    last_print = -1.0
    idx = 0
    with json_path.open("w", encoding="utf-8") as jf:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = idx / native_fps
            if args.seconds is not None and ts > args.seconds:
                break

            if ts + 1e-4 >= next_infer:
                last_result = pipe.process_frame(ts, frame)
                jf.write(json.dumps(_dump_result(last_result, high, low)) + "\n")
                next_infer += infer_dt
                if ts - last_print >= 1.0 or last_result["events"]:
                    parts = [
                        f"id{s['student_id']}:{s.get('band', '?')[:3]}:{s['score']:.0f}"
                        for s in last_result["students"]
                    ]
                    print(
                        f"t={ts:5.1f}s  n={len(last_result['students'])}  "
                        + (", ".join(parts) or "no students")
                    )
                    last_print = ts
                for e in last_result["events"]:
                    ev = e["event"]
                    print(
                        f"  DROP student={ev.student_id}  {ev.kind}  "
                        f"{ev.dominant_modality}  score={ev.score:.0f}"
                    )

            annotated = draw_attention(frame, last_result, high=high, low=low)
            writer.write(annotated)
            idx += 1

    cap.release()
    writer.release()
    report_path = json_path.with_name(json_path.stem + "_report.txt")
    _write_txt_report(report_path, video, written_path, json_path, pipe.dataset_status)
    from attention_pipeline.video_accuracy import write_video_accuracy

    accuracy_path = write_video_accuracy(video, written_path, json_path)
    print(f"Wrote overlay video: {written_path.resolve()}")
    print(f"Wrote scores:        {json_path.resolve()}")
    print(f"Wrote report:        {report_path.resolve()}")
    print(f"Wrote accuracy:      {accuracy_path.resolve()}")


def _write_txt_report(
    path: Path,
    video: Path,
    overlay: Path,
    jsonl: Path,
    status: dict | None,
) -> None:
    from collections import Counter

    bands: Counter[str] = Counter()
    cues: Counter[str] = Counter()
    behaviors: Counter[str] = Counter()
    frames = 0
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            frames += 1
            for s in row.get("students") or []:
                bands[str(s.get("band") or "?")] += 1
                cues[str(s.get("cue") or "")] += 1
                behaviors[str(s.get("behavior") or "")] += 1
    total = sum(bands.values()) or 1
    lines = [
        "=" * 72,
        "CLASSROOM TEST REPORT",
        "=" * 72,
        "",
        f"Input video : {video.resolve()}",
        f"Overlay     : {overlay.resolve()}",
        f"Scores      : {jsonl.resolve()}",
        f"Frames scored: {frames}",
        f"Student labels: {sum(bands.values())}",
        "",
        "Weights",
        f"  L2CS gaze : {'on' if (status or {}).get('l2cs_gaze') else 'off'}",
        f"  SCB       : {'on' if (status or {}).get('scb_behavior') else 'off'}",
        "",
        "Attention bands",
    ]
    for name in ("HIGH", "MODERATE", "LOW"):
        n = bands.get(name, 0)
        lines.append(f"  {name:<10} {n:6d}  ({100.0 * n / total:5.1f}%)")
    lines.append("")
    lines.append("Top cues")
    for cue, n in cues.most_common(10):
        lines.append(f"  {n:6d}  {cue or '(none)'}")
    lines.append("")
    lines.append("Top behaviors")
    for name, n in behaviors.most_common(8):
        lines.append(f"  {n:6d}  {name or '(none)'}")

    from attention_pipeline.video_accuracy import load_model_metrics, measure_rows, model_lines

    metrics = measure_rows(
        [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if jsonl.exists()
        else []
    )
    macro = metrics["macro"]
    lines.extend(["", *model_lines(load_model_metrics()), ""])
    lines.extend(
        [
            "=" * 72,
            "VIDEO ACCURACY",
            "=" * 72,
            "",
            "Counted from the marked students in the input video above.",
            "A different video produces a different video accuracy.",
            "",
            f"{'Band':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}",
            "-" * 72,
        ]
    )
    for band in ("HIGH", "MODERATE", "LOW"):
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
            f"Students compared: {metrics['support']}",
            "=" * 72,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    infer_main()
