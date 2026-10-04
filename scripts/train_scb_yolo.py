"""Train YOLO11 on SCB-dataset (hand-raising / reading / writing).

Does not use FER2013 or RAF-DB.
A checkpoint is written every 30 minutes and at the end of each epoch.
Run the same command again after a power cut to resume.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from attention_pipeline.timed_checkpoint import TimedCheckpoint, resume_path_for
from attention_pipeline.utils import PROJECT_ROOT, resolve_scb_yaml

ROOT = PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "scb_yolo.pt"))
    parser.add_argument("--checkpoint-minutes", type=float, default=30)
    parser.add_argument("--fresh", action="store_true", help="Ignore a saved resume checkpoint")
    args = parser.parse_args()

    data = Path(args.data) if args.data else resolve_scb_yaml()
    if not data.exists():
        raise SystemExit(
            f"Missing {data}\nRun: python scripts/download_datasets.py"
        )
    from ultralytics import YOLO

    run_last = ROOT / "runs" / "scb" / "yolo11n" / "weights" / "last.pt"
    resume_copy = resume_path_for(Path(args.out))
    weights = args.model
    resume = False
    if not args.fresh:
        if run_last.exists():
            weights = str(run_last)
            resume = True
        elif resume_copy.exists():
            weights = str(resume_copy)
            resume = True
    if resume:
        print(f"Resuming SCB YOLO from {weights}")
    else:
        print(f"SCB YOLO checkpoint every {args.checkpoint_minutes:g} min -> {resume_copy}")

    model = YOLO(weights)
    clock = TimedCheckpoint(resume_copy, minutes=args.checkpoint_minutes)

    def _save_half_hour(trainer) -> None:
        if not clock.due():
            return
        trainer.save_model()
        clock.mark()
        last = Path(trainer.last)
        if not last.exists():
            return
        resume_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(last, resume_copy)
        print(f"Checkpoint saved ({args.checkpoint_minutes:g} min) -> {resume_copy}")

    model.add_callback("on_train_batch_end", _save_half_hour)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(ROOT / "runs" / "scb"),
        name="yolo11n",
        exist_ok=True,
        resume=resume,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, out)
        print(f"Copied {best} -> {out}")
    else:
        print("Training finished; copy runs/scb/yolo11n/weights/best.pt to checkpoints/scb_yolo.pt")


if __name__ == "__main__":
    main()
