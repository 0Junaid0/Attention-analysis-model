"""Train on all three datasets: SCB, L2CS-Net, and DAiSEE.

  SCB    every image (train and val) -> behavior YOLO  checkpoints/scb_yolo11n.pt
  L2CS   Gaze360 weights, fine-tuned on a face from every DAiSEE clip
         checkpoints/l2cs_finetuned.pt
         The L2CS repo ships the network, not the Gaze360 image files.
  DAiSEE every labeled clip (train, validation, and test)
         -> fusion net  checkpoints/fusion_daisee.pt

Checkpoints are written every 30 minutes. Run the same command again
after a power cut; finished stages are skipped and the rest resume.

    python scripts/train_all.py
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from attention_pipeline.utils import (  # noqa: E402
    PROJECT_ROOT,
    resolve_daisee_root,
    resolve_l2cs_weights,
    resolve_scb_root,
)

PIPELINE_YAML = PROJECT_ROOT / "configs" / "pipeline.yaml"
GAZE_NPZ = PROJECT_ROOT / "data" / "processed" / "gaze_l2cs.npz"
FUSION_NPZ = PROJECT_ROOT / "data" / "processed" / "fusion_daisee.npz"
SCB_OUT = PROJECT_ROOT / "checkpoints" / "scb_yolo11n.pt"
GAZE_OUT = PROJECT_ROOT / "checkpoints" / "l2cs_finetuned.pt"
FUSION_OUT = PROJECT_ROOT / "checkpoints" / "fusion_daisee.pt"


def scb_all_images() -> list[Path]:
    """Every SCB image, including the official val folder."""
    root = resolve_scb_root()
    images = sorted((root / "images").rglob("*.jpg"))
    if not images:
        raise SystemExit(f"No SCB images under {root}. Run scripts/download_datasets.py")
    return images


def scb_train_images() -> list[Path]:
    return scb_all_images()


def write_scb_all_yaml() -> Path:
    """YOLO data file whose train list is every SCB image."""
    root = resolve_scb_root()
    images = scb_all_images()
    out_dir = PROJECT_ROOT / "data" / "processed" / "scb_all"
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / "all_images.txt"
    listing.write_text(
        "\n".join(p.resolve().as_posix() for p in images) + "\n",
        encoding="utf-8",
    )
    yaml_path = out_dir / "scb_all.yaml"
    yaml_path.write_text(
        f"path: {root.resolve().as_posix()}\n"
        f"train: {listing.resolve().as_posix()}\n"
        f"val: {listing.resolve().as_posix()}\n"
        "nc: 3\n"
        "names: ['hand-raising','reading','writing']\n",
        encoding="utf-8",
    )
    print(f"SCB training list: {len(images)} images -> {yaml_path}")
    return yaml_path


def _daisee_videos() -> dict[str, Path]:
    root = resolve_daisee_root()
    return {
        p.stem: p
        for p in list(root.rglob("*.avi")) + list(root.rglob("*.mp4"))
    }


def _read_label_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
            clip_id = str(row.get("ClipID", ""))
            if not clip_id:
                continue
            labels[Path(clip_id).stem] = {
                "boredom": float(row.get("Boredom", 0)),
                "engagement": float(row.get("Engagement", 0)),
                "confusion": float(row.get("Confusion", 0)),
                "frustration": float(row.get("Frustration", 0)),
            }
    return labels


def daisee_split(name: str) -> list[tuple[Path, dict[str, float]]]:
    """Official DAiSEE split. `name` is Train, Validation, or Test."""
    root = resolve_daisee_root()
    csv_path = root / "DAiSEE" / "Labels" / f"{name}Labels.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run scripts/download_datasets.py")
    videos = _daisee_videos()
    clips = [
        (videos[stem], lab)
        for stem, lab in _read_label_csv(csv_path).items()
        if stem in videos
    ]
    if not clips:
        raise SystemExit(f"No {name} videos found under {root}")
    clips.sort(key=lambda item: item[0].name)
    return clips


def daisee_all_clips() -> list[tuple[Path, dict[str, float]]]:
    """Every labeled DAiSEE clip that is on disk. Train, val, and test are all included."""
    root = resolve_daisee_root()
    labels_dir = root / "DAiSEE" / "Labels"
    names = ["AllLabels.csv", "TrainLabels.csv", "ValidationLabels.csv", "TestLabels.csv"]
    labels: dict[str, dict[str, float]] = {}
    found = False
    for name in names:
        csv_path = labels_dir / name
        if not csv_path.exists():
            continue
        found = True
        labels.update(_read_label_csv(csv_path))
    if not found:
        raise SystemExit(f"No DAiSEE label files under {labels_dir}. Run scripts/download_datasets.py")
    videos = _daisee_videos()
    clips = [(videos[stem], lab) for stem, lab in labels.items() if stem in videos]
    if not clips:
        raise SystemExit(f"No labeled DAiSEE videos found under {root}")
    clips.sort(key=lambda item: item[0].name)
    return clips


def _set_weight(key: str, relative: str) -> None:
    text = PIPELINE_YAML.read_text(encoding="utf-8")
    lines = []
    found = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}{key}: {relative}")
            found = True
        else:
            lines.append(line)
    if not found:
        raise SystemExit(f"Could not find {key} in {PIPELINE_YAML}")
    PIPELINE_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Pipeline will load {key} = {relative}")


def _run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print("\n>>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def build_gaze_npz(clips: list[tuple[Path, dict[str, float]]], out: Path) -> None:
    """Every detected face in every DAiSEE clip, sampled once per second."""
    import cv2

    from attention_pipeline.stages.detect_track import DetectorTracker

    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    images: list[np.ndarray] = []
    pitch: list[float] = []
    yaw: list[float] = []
    if out.exists():
        prev = np.load(out)
        images = list(prev["images"])
        pitch = list(prev["pitch"].astype(np.float32))
        yaw = list(prev["yaw"].astype(np.float32))
        done = set(str(x) for x in prev["clip"])
        print(f"L2CS npz already has {len(done)} clips")
    detector = DetectorTracker(
        weights="yolo11n-pose.pt", conf=0.25, imgsz=640, tiles=False, device="cpu"
    )

    def _angles(kpts: np.ndarray) -> tuple[float, float] | None:
        if kpts is None or kpts.shape[0] < 3:
            return None
        nose, leye, reye = kpts[0], kpts[1], kpts[2]
        if float(nose[2] + leye[2] + reye[2]) < 0.6:
            return None
        mid = (leye[:2] + reye[:2]) / 2.0
        eye_w = abs(float(reye[0] - leye[0])) + 1e-6
        y = float(np.degrees(np.arctan2(nose[0] - mid[0], eye_w)))
        p = float(np.degrees(np.arctan2(nose[1] - mid[1], eye_w)))
        return y, p

    clips_left = [(path, lab) for path, lab in clips if path.stem not in done]
    for index, (path, _lab) in enumerate(clips_left, start=1):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps)), 1)
        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % step == 0:
                for det in detector.infer(frame):
                    angles = _angles(det.keypoints)
                    if angles is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                    h, w = frame.shape[:2]
                    y2 = min(y1 + int(max(y2 - y1, 1) * 0.45), h)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2 = min(x2, w)
                    if x2 - x1 < 32 or y2 - y1 < 32:
                        continue
                    crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    images.append(np.transpose(crop, (2, 0, 1)))
                    yaw.append(angles[0])
                    pitch.append(angles[1])
            frame_i += 1
        cap.release()
        done.add(path.stem)
        if images and (index % 25 == 0 or index == len(clips_left)):
            np.savez(
                out,
                images=np.stack(images).astype(np.uint8) if images else np.zeros((0, 3, 224, 224), np.uint8),
                pitch=np.asarray(pitch, np.float32),
                yaw=np.asarray(yaw, np.float32),
                clip=np.asarray(sorted(done)),
            )
            print(f"L2CS faces {len(done)}/{len(clips)}")
    if not images:
        raise SystemExit("No L2CS training faces were collected from DAiSEE.")
    print(f"L2CS training faces: {len(images)} -> {out}")


def build_fusion_npz(clips: list[tuple[Path, dict[str, float]]], out: Path) -> None:
    """Fusion windows over each full DAiSEE clip, using that clip's engagement label."""
    from attention_pipeline.pipeline import AttentionPipeline
    from attention_pipeline.stages.source import iter_frames
    from attention_pipeline.types import FEAT_DIM

    out.parent.mkdir(parents=True, exist_ok=True)
    features: list[np.ndarray] = []
    scores: list[float] = []
    auxs: list[np.ndarray] = []
    done: set[str] = set()
    if out.exists():
        prev = np.load(out)
        features = list(prev["features"])
        scores = list(prev["score"].astype(np.float32))
        auxs = list(prev["aux"])
        done = set(str(x) for x in prev["clip"])
        print(f"DAiSEE fusion npz already has {len(done)} clips")
    pipe = AttentionPipeline.from_yaml(PIPELINE_YAML, load_weights=True)
    window = int(pipe.cfg.get("fusion", {}).get("window", 16))
    clips_left = [(path, lab) for path, lab in clips if path.stem not in done]
    for index, (path, lab) in enumerate(clips_left, start=1):
        pipe.tracks.clear()
        pipe._lesson_yaw = None
        score = float(np.clip(lab["engagement"] / 3.0 * 100.0, 0, 100))
        aux = np.array(
            [lab["engagement"], lab["boredom"], lab["confusion"], lab["frustration"]],
            np.float32,
        )
        produced = 0
        for _ts, frame in iter_frames(str(path), target_fps=5.0):
            pipe.process_frame(_ts, frame)
            for track in pipe.tracks.values():
                if len(track.feature_window) >= window:
                    seq = np.stack(list(track.feature_window)[-window:], axis=0).astype(np.float32)
                    features.append(seq)
                    scores.append(score)
                    auxs.append(aux)
                    produced += 1
                break
        if produced == 0:
            blank = np.zeros((window, FEAT_DIM), np.float32)
            features.append(blank)
            scores.append(score)
            auxs.append(aux)
        done.add(path.stem)
        if index % 10 == 0 or index == len(clips_left):
            np.savez(
                out,
                features=np.stack(features).astype(np.float32),
                score=np.asarray(scores, np.float32),
                aux=np.stack(auxs).astype(np.float32),
                clip=np.asarray(sorted(done)),
            )
            print(f"DAiSEE fusion clips {len(done)}/{len(clips)}")
    print(f"DAiSEE fusion windows: {len(features)} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on SCB, L2CS-Net, and DAiSEE")
    parser.add_argument("--skip-scb", action="store_true")
    parser.add_argument("--skip-l2cs", action="store_true")
    parser.add_argument("--skip-daisee", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--scb-epochs", type=int, default=50)
    parser.add_argument("--gaze-epochs", type=int, default=20)
    parser.add_argument("--fusion-epochs", type=int, default=30)
    parser.add_argument("--checkpoint-minutes", type=float, default=30)
    args = parser.parse_args()

    images = scb_all_images()
    clips = daisee_all_clips()
    resolve_l2cs_weights()
    print(f"SCB images : {len(images)} (train and val)")
    print(f"DAiSEE clips: {len(clips)} (train, validation, and test)")
    print("L2CS-Net weights : checkpoints/l2cs_gaze360_resnet50.safetensors")
    minutes = str(args.checkpoint_minutes)

    if not args.skip_scb:
        data_yaml = write_scb_all_yaml()
        _run(
            "train_scb_yolo.py",
            "--data",
            str(data_yaml),
            "--epochs",
            str(args.scb_epochs),
            "--out",
            str(SCB_OUT),
            "--checkpoint-minutes",
            minutes,
        )
        _set_weight("scb_weights", "checkpoints/scb_yolo11n.pt")

    if args.prepare_only and args.skip_l2cs and args.skip_daisee:
        return

    if not args.skip_l2cs:
        build_gaze_npz(clips, GAZE_NPZ)
        if not args.prepare_only:
            _run(
                "train_gaze.py",
                "--npz",
                str(GAZE_NPZ),
                "--epochs",
                str(args.gaze_epochs),
                "--out",
                str(GAZE_OUT),
                "--checkpoint-minutes",
                minutes,
            )
            _set_weight("gaze_weights", "checkpoints/l2cs_finetuned.pt")

    if not args.skip_daisee:
        build_fusion_npz(clips, FUSION_NPZ)
        if not args.prepare_only:
            _run(
                "train_fusion.py",
                "--npz",
                str(FUSION_NPZ),
                "--epochs",
                str(args.fusion_epochs),
                "--out",
                str(FUSION_OUT),
                "--checkpoint-minutes",
                minutes,
            )
            _set_weight("fusion_weights", "checkpoints/fusion_daisee.pt")

    print("\nTraining covers SCB, L2CS-Net, and DAiSEE.")
    print("Re-run this command to resume after a power cut.")


if __name__ == "__main__":
    main()
