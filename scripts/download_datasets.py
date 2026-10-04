"""Download and attach every dataset this thesis pipeline uses.

  SCB:   https://github.com/Whiffe/SCB-dataset.git  (+ HF images/weights)
  L2CS:  https://github.com/Ahmednull/L2CS-Net.git (+ Gaze360 weights)
  DAiSEE (optional fusion): olgaparfenova/daisee via kagglehub

FER2013 and RAF-DB are not downloaded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-daisee", action="store_true", help="Skip ~14GB DAiSEE download")
    parser.add_argument("--skip-scb-zip", action="store_true", help="Skip SCB image zip (weights still download)")
    args = parser.parse_args()

    scb_args = []
    if args.skip_scb_zip:
        scb_args.append("--skip-scb-zip")
    _run("setup_scb_l2cs.py", *scb_args)

    if not args.skip_daisee:
        try:
            _run("download_daisee_kaggle.py")
        except subprocess.CalledProcessError as exc:
            print(f"DAiSEE download skipped/failed ({exc}). SCB+L2CS are enough for the overlay.")

    # Refresh YOLO data yaml with an absolute path for training.
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.utils import resolve_scb_root, resolve_scb_weights, resolve_l2cs_weights

    scb_root = resolve_scb_root()
    yaml_path = scb_root / "scb.yaml"
    yaml_path.write_text(
        f"path: {str(scb_root).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 3\n"
        "names: ['hand-raising','reading','writing']\n",
        encoding="utf-8",
    )
    print(f"\nSCB yaml: {yaml_path}")
    print(f"L2CS weights: {resolve_l2cs_weights()}")
    print(f"SCB weights:  {resolve_scb_weights()}")
    print("\nDatasets ready. Run:")
    print('  python -m attention_pipeline.cli "video\\Classroom_video.mp4" --seconds 20')


if __name__ == "__main__":
    main()
