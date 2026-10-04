"""Quick smoke train before full training, then test on the classroom video.

Uses:
  - 20 SCB images  → tiny YOLO behavior fine-tune
  - 10 DAiSEE clips → tiny AttentionFusionNet train
  - L2CS Gaze360 stays pretrained (not re-trained here)

Then runs inference on video/Classroom_video.mp4.

FER2013 / RAF-DB are not used.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "processed" / "smoke"
SCB_SRC = (
    ROOT
    / "data"
    / "raw"
    / "scb"
    / "SCB5-Handrise-Read-write"
    / "SCB5-Handrise-Read-write-2024-9-17"
)
DAISEE = ROOT / "data" / "raw" / "daisee"
CLASSROOM = ROOT / "video" / "Classroom_video.mp4"


def _copy_scb_subset(n_images: int = 20, seed: int = 0) -> Path:
    """Copy n_images SCB samples into a YOLO-ready mini dataset."""
    train_imgs = sorted((SCB_SRC / "images" / "train").glob("*.jpg"))
    if len(train_imgs) < n_images:
        raise SystemExit(f"Need {n_images} SCB images, found {len(train_imgs)}. Run download_datasets.py")

    rng = random.Random(seed)
    picked = rng.sample(train_imgs, n_images)
    n_val = max(2, n_images // 5)
    val_set = set(picked[:n_val])
    train_set = picked[n_val:]

    out = SMOKE / "scb"
    if out.exists():
        shutil.rmtree(out)
    for split, files in (("train", train_set), ("val", list(val_set))):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, out / "images" / split / img.name)
            lbl = SCB_SRC / "labels" / "train" / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, out / "labels" / split / lbl.name)
            else:
                (out / "labels" / split / (img.stem + ".txt")).write_text("", encoding="utf-8")

    yaml_path = out / "scb_smoke.yaml"
    yaml_path.write_text(
        f"path: {str(out).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 3\n"
        "names: ['hand-raising','reading','writing']\n",
        encoding="utf-8",
    )
    print(f"SCB smoke set: {len(train_set)} train + {len(val_set)} val images -> {out}")
    return yaml_path


def _train_scb_yolo(data_yaml: Path, epochs: int, out_ckpt: Path) -> Path:
    from ultralytics import YOLO

    # Start from the official SCB weights when available; else YOLO11n.
    init = ROOT / "checkpoints" / "scb_yolo.pt"
    # Official HF weights are YOLOv7 — Ultralytics cannot resume them.
    # Train a tiny YOLO11n on the 20-image subset instead.
    model = YOLO("yolo11n.pt")
    print(f"Training YOLO11n on smoke SCB ({epochs} epochs)...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=4,
        project=str(SMOKE / "runs"),
        name="scb_yolo",
        exist_ok=True,
        patience=epochs,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, out_ckpt)
        print(f"Saved {out_ckpt}")
        return out_ckpt
    last = Path(results.save_dir) / "weights" / "last.pt"
    if last.exists():
        shutil.copy2(last, out_ckpt)
        print(f"Saved {out_ckpt} (last.pt)")
        return out_ckpt
    raise SystemExit("YOLO smoke train finished but no weights were written.")


def _daisee_label_map() -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    for name in ("TrainLabels.csv", "ValidationLabels.csv", "TestLabels.csv", "AllLabels.csv"):
        path = DAISEE / "DAiSEE" / "Labels" / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                # headers may have trailing spaces
                row = {k.strip(): v.strip() for k, v in row.items() if k}
                cid = row.get("ClipID", "")
                if not cid:
                    continue
                stem = Path(cid).stem
                labels[stem] = {
                    "boredom": float(row.get("Boredom", 0)),
                    "engagement": float(row.get("Engagement", 0)),
                    "confusion": float(row.get("Confusion", 0)),
                    "frustration": float(row.get("Frustration", 0)),
                }
    return labels


def _pick_daisee_clips(n_clips: int = 10, seed: int = 0) -> list[tuple[Path, dict[str, float]]]:
    labels = _daisee_label_map()
    videos = list(DAISEE.rglob("*.avi")) + list(DAISEE.rglob("*.mp4"))
    if not videos:
        raise SystemExit("No DAiSEE clips found. Run: python scripts/download_daisee_kaggle.py")

    labeled = []
    for v in videos:
        lab = labels.get(v.stem)
        if lab is not None:
            labeled.append((v, lab))
    if len(labeled) < n_clips:
        # fall back to any videos with synthetic mid engagement
        rng = random.Random(seed)
        picked = rng.sample(videos, min(n_clips, len(videos)))
        return [(p, {"boredom": 0, "engagement": 2, "confusion": 0, "frustration": 0}) for p in picked]

    rng = random.Random(seed)
    return rng.sample(labeled, n_clips)


def _build_fusion_npz(n_clips: int, seconds: float, seed: int) -> Path:
    """Run the live pipeline on 10 short DAiSEE clips and pack fusion windows."""
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.pipeline import AttentionPipeline
    from attention_pipeline.types import FEAT_DIM

    clips = _pick_daisee_clips(n_clips=n_clips, seed=seed)
    print(f"DAiSEE smoke clips: {len(clips)}")
    for p, lab in clips:
        print(f"  {p.name}  engagement={lab['engagement']}")

    # Use pretrained L2CS + official SCB for feature extraction (before smoke YOLO overwrites config).
    pipe = AttentionPipeline.from_yaml(ROOT / "configs" / "pipeline.yaml", load_weights=True)

    features: list[np.ndarray] = []
    scores: list[float] = []
    auxs: list[np.ndarray] = []
    window = int(pipe.cfg.get("fusion", {}).get("window", 16))

    for path, lab in clips:
        pipe.tracks.clear()
        pipe._lesson_yaw = None
        eng = float(lab["engagement"])
        score = float(np.clip(eng / 3.0 * 100.0, 0, 100))
        aux = np.array(
            [lab["engagement"], lab["boredom"], lab["confusion"], lab["frustration"]],
            np.float32,
        )
        # Collect up to `window` feature vectors from the first student track.
        collected: list[np.ndarray] = []
        for ts, frame in _iter_clip(path, max_seconds=seconds, target_fps=5.0):
            result = pipe.process_frame(ts, frame)
            # Prefer features from the scorer windows if any track has them
            for track in pipe.tracks.values():
                if track.feature_window:
                    collected = list(track.feature_window[-window:])
                    break
            if len(collected) >= window:
                break
        if not collected:
            # empty clip — synthetic neutral window so training still runs
            collected = [np.zeros(FEAT_DIM, np.float32) for _ in range(window)]
        while len(collected) < window:
            collected = [collected[0]] + collected
        stacked = np.stack(collected[-window:], axis=0).astype(np.float32)
        features.append(stacked)
        scores.append(score)
        auxs.append(aux)

    out = SMOKE / "fusion_smoke.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        features=np.stack(features, axis=0),
        score=np.asarray(scores, np.float32),
        aux=np.stack(auxs, axis=0),
    )
    print(f"Wrote {out}  features={np.stack(features).shape}")
    return out


def _iter_clip(path: Path, max_seconds: float, target_fps: float):
    from attention_pipeline.stages.source import iter_frames

    for ts, frame in iter_frames(str(path), target_fps=target_fps):
        if ts > max_seconds:
            break
        yield ts, frame


def _train_fusion(npz_path: Path, epochs: int, out_ckpt: Path) -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from attention_pipeline.models.attention_fusion import AttentionFusionNet
    from attention_pipeline.utils import resolve_device

    class _DS(Dataset):
        def __init__(self, path: Path) -> None:
            data = np.load(path)
            self.x = data["features"].astype(np.float32)
            self.score = data["score"].astype(np.float32)
            self.aux = data["aux"].astype(np.float32)

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, i: int):
            return (
                torch.from_numpy(self.x[i]),
                torch.tensor(self.score[i]),
                torch.from_numpy(self.aux[i]),
            )

    ds = _DS(npz_path)
    n_val = max(1, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=min(4, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(4, len(val_ds)))
    device = resolve_device("auto")
    model = AttentionFusionNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    score_loss = nn.SmoothL1Loss()
    aux_loss = nn.SmoothL1Loss()
    best = 1e9
    print(f"Training fusion on {len(ds)} clip-windows ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        for x, score, aux in train_loader:
            x, score, aux = x.to(device), score.to(device), aux.to(device)
            pred_s, pred_a = model(x)
            loss = score_loss(pred_s, score) + 0.5 * aux_loss(pred_a, aux)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        total = n = 0
        with torch.no_grad():
            for x, score, aux in val_loader:
                x, score = x.to(device), score.to(device)
                pred_s, _ = model(x)
                total += float(score_loss(pred_s, score).item()) * x.size(0)
                n += x.size(0)
        val = total / max(n, 1)
        print(f"  fusion epoch {epoch:02d}  val_smoothl1 {val:.3f}")
        if val <= best:
            best = val
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val": val, "smoke": True}, out_ckpt)
    print(f"Saved {out_ckpt}")
    return out_ckpt


def _write_smoke_config(scb_ckpt: Path, fusion_ckpt: Path, *, use_smoke_scb: bool = False) -> Path:
    """Classroom test config.

    Tiny 20-image YOLO is only a training dry-run — by default the classroom
    overlay keeps the official pretrained SCB weights so reading/writing work.
    Set use_smoke_scb=True to force the smoke checkpoint.
    """
    src = ROOT / "configs" / "pipeline.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    official = ROOT / "checkpoints" / "scb_yolo.pt"
    if use_smoke_scb and scb_ckpt.exists():
        cfg["models"]["scb_weights"] = str(scb_ckpt.relative_to(ROOT)).replace("\\", "/")
    elif official.exists():
        cfg["models"]["scb_weights"] = "checkpoints/scb_yolo.pt"
    else:
        cfg["models"]["scb_weights"] = str(scb_ckpt.relative_to(ROOT)).replace("\\", "/")
    # Smoke fusion on 10 clips overfits; classroom overlay uses the rubric instead.
    cfg["models"]["fusion_weights"] = "checkpoints/fusion_daisee.pt"
    cfg["models"]["gaze_weights"] = "checkpoints/l2cs_gaze360_resnet50.safetensors"
    out = ROOT / "configs" / "pipeline_smoke.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"  scb_weights    = {cfg['models']['scb_weights']}")
    print(f"  fusion_weights = {cfg['models']['fusion_weights']} (missing -> rubric)")
    return out


def _run_classroom_test(config: Path, seconds: float, out_video: Path) -> None:
    if not CLASSROOM.exists():
        raise SystemExit(f"Missing classroom video: {CLASSROOM}")
    cmd = [
        sys.executable,
        "-m",
        "attention_pipeline.cli",
        str(CLASSROOM),
        "--config",
        str(config),
        "--seconds",
        str(seconds),
        "--out-video",
        str(out_video),
        "--out",
        str(ROOT / "runs" / "smoke_inference.jsonl"),
    ]
    print("\n>>>", " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke train (20 images + 10 clips) then test classroom video")
    p.add_argument("--images", type=int, default=20, help="SCB images for mini YOLO train")
    p.add_argument("--clips", type=int, default=10, help="DAiSEE clips for mini fusion train")
    p.add_argument("--scb-epochs", type=int, default=5)
    p.add_argument("--fusion-epochs", type=int, default=8)
    p.add_argument("--clip-seconds", type=float, default=2.0)
    p.add_argument("--test-seconds", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-scb-train", action="store_true")
    p.add_argument("--skip-fusion-train", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument(
        "--use-smoke-scb",
        action="store_true",
        help="Use the 20-image YOLO on the classroom video (weak; for debugging only)",
    )
    args = p.parse_args()

    SMOKE.mkdir(parents=True, exist_ok=True)
    scb_ckpt = ROOT / "checkpoints" / "scb_yolo_smoke.pt"
    fusion_ckpt = ROOT / "checkpoints" / "fusion_smoke.pt"

    if not args.skip_scb_train:
        data_yaml = _copy_scb_subset(n_images=args.images, seed=args.seed)
        _train_scb_yolo(data_yaml, epochs=args.scb_epochs, out_ckpt=scb_ckpt)
    elif not scb_ckpt.exists():
        scb_ckpt = ROOT / "checkpoints" / "scb_yolo.pt"

    if not args.skip_fusion_train:
        npz = _build_fusion_npz(n_clips=args.clips, seconds=args.clip_seconds, seed=args.seed)
        _train_fusion(npz, epochs=args.fusion_epochs, out_ckpt=fusion_ckpt)
    elif not fusion_ckpt.exists():
        fusion_ckpt = ROOT / "checkpoints" / "fusion_daisee.pt"

    if not scb_ckpt.exists():
        scb_ckpt = ROOT / "checkpoints" / "scb_yolo.pt"
    if not fusion_ckpt.exists():
        fusion_ckpt = ROOT / "checkpoints" / "fusion_smoke.pt"

    config = _write_smoke_config(scb_ckpt, fusion_ckpt, use_smoke_scb=args.use_smoke_scb)

    if not args.skip_test:
        out_video = ROOT / "runs" / "attention_overlay_smoke.mp4"
        _run_classroom_test(config, seconds=args.test_seconds, out_video=out_video)
        print("\nSmoke test done.")
        print(f"  Overlay: {out_video}")
        print(f"  JSONL:   {ROOT / 'runs' / 'smoke_inference.jsonl'}")
        print(f"  Mini SCB YOLO: {ROOT / 'checkpoints' / 'scb_yolo_smoke.pt'}")
        print(f"  Mini fusion:   {ROOT / 'checkpoints' / 'fusion_smoke.pt'}")
        print("This is a tiny train only. Full training still uses the full SCB/DAiSEE sets.")


if __name__ == "__main__":
    main()
