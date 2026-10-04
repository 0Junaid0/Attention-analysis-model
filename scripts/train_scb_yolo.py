"""Train YOLO11 on SCB-dataset (hand-raising / reading / writing).

Does not use FER2013 or RAF-DB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from attention_pipeline.utils import PROJECT_ROOT, resolve_scb_yaml

ROOT = PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "scb_yolo.pt"))
    args = parser.parse_args()

    data = Path(args.data) if args.data else resolve_scb_yaml()
    if not data.exists():
        raise SystemExit(
            f"Missing {data}\nRun: python scripts/download_datasets.py"
        )
    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(ROOT / "runs" / "scb"),
        name="yolo11n",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        import shutil

        shutil.copy2(best, out)
        print(f"Copied {best} -> {out}")
    else:
        print("Training finished; copy runs/scb/yolo11n/weights/best.pt to checkpoints/scb_yolo.pt")


if __name__ == "__main__":
    main()
