"""Expanded smoke train + benchmark validation metrics.

Defaults (adds to the previous 20-image / 10-clip run):
  - 40 SCB images   (train/val split)
  - 20 DAiSEE clips (train/val split)
  - 40 L2CS face samples (Gaze360-style angular + direction metrics)

Reports precision, recall, F1, accuracy for each task, then tests the
classroom video. FER2013 / RAF-DB are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
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
SCB_NAMES = ["hand-raising", "reading", "writing"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf1_acc(y_true: list[int], y_pred: list[int], n_classes: int) -> dict[str, float]:
    """Macro-averaged precision / recall / F1 + overall accuracy."""
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "support": 0}
    y_true_a = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred, dtype=int)
    acc = float((y_true_a == y_pred_a).mean())
    precs, recs, f1s = [], [], []
    for c in range(n_classes):
        tp = int(((y_pred_a == c) & (y_true_a == c)).sum())
        fp = int(((y_pred_a == c) & (y_true_a != c)).sum())
        fn = int(((y_pred_a != c) & (y_true_a == c)).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
    return {
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
        "accuracy": acc,
        "support": int(len(y_true)),
    }


def _print_metrics(title: str, m: dict) -> None:
    print(f"\n=== {title} ===")
    for k in ("precision", "recall", "f1", "accuracy"):
        if k in m:
            print(f"  {k:10s}: {m[k]*100:6.2f}%")
    for k, v in m.items():
        if k not in ("precision", "recall", "f1", "accuracy", "support"):
            if isinstance(v, float):
                print(f"  {k:10s}: {v:.3f}")
            else:
                print(f"  {k:10s}: {v}")
    if "support" in m:
        print(f"  support   : {m['support']}")


# ---------------------------------------------------------------------------
# SCB subset + train + val
# ---------------------------------------------------------------------------

def _copy_scb_subset(n_images: int, seed: int) -> Path:
    train_imgs = sorted((SCB_SRC / "images" / "train").glob("*.jpg"))
    if len(train_imgs) < n_images:
        raise SystemExit(f"Need {n_images} SCB images, found {len(train_imgs)}")

    rng = random.Random(seed)
    picked = rng.sample(train_imgs, n_images)
    n_val = max(4, n_images // 5)
    val_set = picked[:n_val]
    train_set = picked[n_val:]

    out = SMOKE / "scb"
    if out.exists():
        shutil.rmtree(out)
    for split, files in (("train", train_set), ("val", val_set)):
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
    print(f"SCB set: {len(train_set)} train + {len(val_set)} val (= {n_images} images)")
    return yaml_path


def _train_scb_yolo(data_yaml: Path, epochs: int, out_ckpt: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    print(f"Training YOLO11n on SCB smoke ({epochs} epochs)...")
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
    src = best if best.exists() else Path(results.save_dir) / "weights" / "last.pt"
    shutil.copy2(src, out_ckpt)
    print(f"Saved {out_ckpt}")
    return out_ckpt


def _load_yolo_labels(label_path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((cls, np.array([x1, y1, x2, y2], np.float32)))
    return boxes


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def _eval_scb_detection(weights: Path, conf: float = 0.25, iou_thr: float = 0.5) -> dict:
    """SCB-dataset style detection val: match boxes @ IoU, then class metrics."""
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.stages.scb_behavior import SCBBehaviorDetector

    val_dir = SMOKE / "scb" / "images" / "val"
    lbl_dir = SMOKE / "scb" / "labels" / "val"
    det = SCBBehaviorDetector(weights=weights, device="cpu", conf=conf)
    y_true: list[int] = []
    y_pred: list[int] = []
    tp = fp = fn = 0

    for img_path in sorted(val_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gts = _load_yolo_labels(lbl_dir / (img_path.stem + ".txt"), w, h)
        preds = det.infer(frame)
        # map names to cls 0..2
        pred_boxes = []
        for p in preds:
            name = p.name
            if name == "reading":
                c = 1
            elif name == "writing":
                c = 2
            elif name == "hand-raising":
                c = 0
            else:
                c = p.cls if p.cls < 3 else 0
            pred_boxes.append((c, p.bbox_xyxy, p.conf))

        matched_pred = set()
        for g_cls, g_box in gts:
            best_i, best_iou = -1, 0.0
            for i, (p_cls, p_box, _) in enumerate(pred_boxes):
                if i in matched_pred:
                    continue
                iou = _iou(g_box, p_box)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= iou_thr:
                matched_pred.add(best_i)
                p_cls = pred_boxes[best_i][0]
                y_true.append(g_cls)
                y_pred.append(p_cls)
                if p_cls == g_cls:
                    tp += 1
                else:
                    fp += 1  # wrong class counts as FP for detection quality
                    fn += 1
            else:
                fn += 1
                y_true.append(g_cls)
                y_pred.append(-1)  # will count as wrong for class accuracy below

        fp += len(pred_boxes) - len(matched_pred)

    # Detection-level metrics (IoU match + correct class = TP)
    det_p = tp / (tp + fp) if (tp + fp) else 0.0
    det_r = tp / (tp + fn) if (tp + fn) else 0.0
    det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) else 0.0
    det_acc = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    # Class-only metrics on matched GT boxes (ignore unmatched preds)
    matched_true = [t for t, p in zip(y_true, y_pred) if p >= 0]
    matched_pred = [p for p in y_pred if p >= 0]
    cls = _prf1_acc(matched_true, matched_pred, n_classes=3) if matched_true else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "support": 0
    }

    return {
        "precision": det_p,
        "recall": det_r,
        "f1": det_f1,
        "accuracy": det_acc,
        "class_precision": cls["precision"],
        "class_recall": cls["recall"],
        "class_f1": cls["f1"],
        "class_accuracy": cls["accuracy"],
        "iou_threshold": iou_thr,
        "det_tp": tp,
        "det_fp": fp,
        "det_fn": fn,
        "support": int(len(y_true)),
        "weights": str(weights),
        "benchmark": "SCB-dataset detection @ IoU 0.5 (hand-raising/reading/writing)",
    }


# ---------------------------------------------------------------------------
# DAiSEE clips + fusion
# ---------------------------------------------------------------------------

def _daisee_label_map() -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    for name in ("TrainLabels.csv", "ValidationLabels.csv", "TestLabels.csv", "AllLabels.csv"):
        path = DAISEE / "DAiSEE" / "Labels" / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
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


def _pick_daisee_clips(n_clips: int, seed: int) -> list[tuple[Path, dict[str, float]]]:
    labels = _daisee_label_map()
    videos = list(DAISEE.rglob("*.avi")) + list(DAISEE.rglob("*.mp4"))
    labeled = [(v, labels[v.stem]) for v in videos if v.stem in labels]
    if len(labeled) < n_clips:
        raise SystemExit(f"Need {n_clips} labeled DAiSEE clips, found {len(labeled)}")
    rng = random.Random(seed)
    return rng.sample(labeled, n_clips)


def _iter_clip(path: Path, max_seconds: float, target_fps: float):
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.stages.source import iter_frames

    for ts, frame in iter_frames(str(path), target_fps=target_fps):
        if ts > max_seconds:
            break
        yield ts, frame


def _build_fusion_from_clips(
    clips: list[tuple[Path, dict[str, float]]],
    seconds: float,
    window: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.pipeline import AttentionPipeline
    from attention_pipeline.types import FEAT_DIM

    pipe = AttentionPipeline.from_yaml(ROOT / "configs" / "pipeline.yaml", load_weights=True)
    features, scores, auxs = [], [], []
    for path, lab in clips:
        pipe.tracks.clear()
        pipe._lesson_yaw = None
        eng = float(lab["engagement"])
        score = float(np.clip(eng / 3.0 * 100.0, 0, 100))
        aux = np.array(
            [lab["engagement"], lab["boredom"], lab["confusion"], lab["frustration"]],
            np.float32,
        )
        collected: list[np.ndarray] = []
        for ts, frame in _iter_clip(path, max_seconds=seconds, target_fps=5.0):
            pipe.process_frame(ts, frame)
            for track in pipe.tracks.values():
                if track.feature_window:
                    collected = list(track.feature_window[-window:])
                    break
            if len(collected) >= window:
                break
        if not collected:
            collected = [np.zeros(FEAT_DIM, np.float32) for _ in range(window)]
        while len(collected) < window:
            collected = [collected[0]] + collected
        features.append(np.stack(collected[-window:], axis=0).astype(np.float32))
        scores.append(score)
        auxs.append(aux)
    return (
        np.stack(features, axis=0),
        np.asarray(scores, np.float32),
        np.stack(auxs, axis=0),
    )


def _train_fusion(
    x_train: np.ndarray,
    s_train: np.ndarray,
    a_train: np.ndarray,
    epochs: int,
    out_ckpt: Path,
) -> Path:
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from attention_pipeline.models.attention_fusion import AttentionFusionNet
    from attention_pipeline.utils import resolve_device

    device = resolve_device("auto")
    model = AttentionFusionNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    score_loss = nn.SmoothL1Loss()
    aux_loss = nn.SmoothL1Loss()
    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(s_train),
        torch.from_numpy(a_train),
    )
    loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True)
    print(f"Training fusion on {len(ds)} windows ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, score, aux in loader:
            x, score, aux = x.to(device), score.to(device), aux.to(device)
            pred_s, pred_a = model(x)
            loss = score_loss(pred_s, score) + 0.5 * aux_loss(pred_a, aux)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"  fusion epoch {epoch:02d}  loss {total / max(len(loader), 1):.3f}")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "smoke": True}, out_ckpt)
    print(f"Saved {out_ckpt}")
    return out_ckpt


def _eval_fusion(
    weights: Path,
    x_val: np.ndarray,
    s_val: np.ndarray,
    a_val: np.ndarray,
) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from attention_pipeline.models.attention_fusion import AttentionFusionNet
    from attention_pipeline.utils import resolve_device

    device = resolve_device("auto")
    model = AttentionFusionNet().to(device)
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred_s, pred_a = model(torch.from_numpy(x_val).to(device))
        pred_s = pred_s.cpu().numpy()
        pred_a = pred_a.cpu().numpy()

    # DAiSEE engagement is 0..3 — map score back to engagement class
    y_true = [int(round(float(a[0]))) for a in a_val]
    y_pred = [int(np.clip(round(float(s) / 100.0 * 3.0), 0, 3)) for s in pred_s]
    metrics = _prf1_acc(y_true, y_pred, n_classes=4)
    metrics["score_mae"] = float(np.mean(np.abs(pred_s - s_val)))
    metrics["engagement_mae"] = float(np.mean(np.abs(pred_a[:, 0] - a_val[:, 0])))
    metrics["benchmark"] = "DAiSEE engagement (0-3) from AttentionFusionNet"
    return metrics


# ---------------------------------------------------------------------------
# L2CS (40 face samples) — Gaze360-style angular error + direction bins
# ---------------------------------------------------------------------------

def _geometric_yaw_pitch(kpts: np.ndarray) -> tuple[float, float] | None:
    if kpts is None or kpts.shape[0] < 3:
        return None
    nose, leye, reye = kpts[0], kpts[1], kpts[2]
    if float(nose[2] + leye[2] + reye[2]) < 0.6:
        return None
    mid = (leye[:2] + reye[:2]) / 2.0
    eye_w = abs(float(reye[0] - leye[0])) + 1e-6
    yaw = float(np.degrees(np.arctan2(nose[0] - mid[0], eye_w)))
    pitch = float(np.degrees(np.arctan2(nose[1] - mid[1], eye_w)))
    return yaw, pitch


def _build_l2cs_samples(n_faces: int, seed: int, seconds: float = 1.5) -> Path:
    """Build 40 face crops with geometric yaw/pitch labels (proxy GT for smoke val).

    Full Gaze360 needs a separate academic download; this validates the L2CS
    checkpoint on in-domain faces using the same angular-error protocol.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from attention_pipeline.stages.detect_track import DetectorTracker

    out = SMOKE / "l2cs"
    if out.exists():
        shutil.rmtree(out)
    (out / "faces").mkdir(parents=True)
    clips = _pick_daisee_clips(n_clips=min(20, max(10, n_faces // 2)), seed=seed + 7)
    detector = DetectorTracker(weights="yolo11n-pose.pt", conf=0.35, device="cpu", n_init=1)
    rng = random.Random(seed)
    rows = []
    for path, _ in clips:
        if len(rows) >= n_faces:
            break
        for ts, frame in _iter_clip(path, max_seconds=seconds, target_fps=3.0):
            if len(rows) >= n_faces:
                break
            dets = detector.infer(frame)
            rng.shuffle(dets)
            for det in dets:
                if len(rows) >= n_faces:
                    break
                gp = _geometric_yaw_pitch(det.keypoints)
                if gp is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                h, w = frame.shape[:2]
                # face-ish upper crop
                fh = max(y2 - y1, 1)
                y2f = min(y1 + int(fh * 0.45), h)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2 = min(x2, w)
                if x2 - x1 < 32 or y2f - y1 < 32:
                    continue
                crop = frame[y1:y2f, x1:x2]
                name = f"{len(rows):04d}.jpg"
                cv2.imwrite(str(out / "faces" / name), crop)
                yaw, pitch = gp
                rows.append({"file": name, "yaw": yaw, "pitch": pitch, "source": path.name})
    if len(rows) < n_faces:
        print(f"Warning: only collected {len(rows)}/{n_faces} L2CS face samples")
    meta = out / "labels.json"
    meta.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"L2CS set: {len(rows)} face crops -> {out}")
    return out


def _yaw_bin(yaw: float) -> int:
    # left / center / right
    if yaw < -15:
        return 0
    if yaw > 15:
        return 2
    return 1


def _pitch_bin(pitch: float) -> int:
    # up / center / down
    if pitch < -10:
        return 0
    if pitch > 10:
        return 2
    return 1


def _eval_l2cs(sample_dir: Path) -> dict:
    """L2CS-Net Gaze360 benchmark protocol: angular MAE + direction classification."""
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from torchvision.transforms import functional as TF

    from attention_pipeline.models.l2cs_net import L2CSNet
    from attention_pipeline.utils import resolve_device, resolve_l2cs_weights

    rows = json.loads((sample_dir / "labels.json").read_text(encoding="utf-8"))
    device = resolve_device("auto")
    model = L2CSNet().to(device).eval()
    model.load(resolve_l2cs_weights(), map_location=device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    yaw_err, pitch_err, angular = [], [], []
    y_true_dir, y_pred_dir = [], []

    def _angular(py, pp, gy, gp):
        # approximate 2D angular error in degrees (Gaze360-style)
        p = np.array([np.cos(np.radians(pp)) * np.sin(np.radians(py)),
                      np.sin(np.radians(pp)),
                      np.cos(np.radians(pp)) * np.cos(np.radians(py))])
        g = np.array([np.cos(np.radians(gp)) * np.sin(np.radians(gy)),
                      np.sin(np.radians(gp)),
                      np.cos(np.radians(gp)) * np.cos(np.radians(gy))])
        cos = float(np.clip(np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g) + 1e-8), -1, 1))
        return float(np.degrees(np.arccos(cos)))

    for row in rows:
        img = cv2.imread(str(sample_dir / "faces" / row["file"]))
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        t = TF.to_tensor(rgb).unsqueeze(0).to(device)
        t = (t - mean) / std
        with torch.no_grad():
            pred = model.predict_degrees(t).squeeze(0).cpu().numpy()
        pyaw, ppitch = float(pred[0]), float(pred[1])
        gyaw, gpitch = float(row["yaw"]), float(row["pitch"])
        yaw_err.append(abs(pyaw - gyaw))
        pitch_err.append(abs(ppitch - gpitch))
        angular.append(_angular(pyaw, ppitch, gyaw, gpitch))
        # 9-class direction: yaw_bin * 3 + pitch_bin
        y_true_dir.append(_yaw_bin(gyaw) * 3 + _pitch_bin(gpitch))
        y_pred_dir.append(_yaw_bin(pyaw) * 3 + _pitch_bin(ppitch))

    cls = _prf1_acc(y_true_dir, y_pred_dir, n_classes=9)
    return {
        **cls,
        "angular_mae_deg": float(np.mean(angular)) if angular else 0.0,
        "yaw_mae_deg": float(np.mean(yaw_err)) if yaw_err else 0.0,
        "pitch_mae_deg": float(np.mean(pitch_err)) if pitch_err else 0.0,
        "benchmark": "L2CS-Net Gaze360 protocol (angular MAE) + 9-bin direction P/R/F1/Acc",
        "note": "GT yaw/pitch = geometric head pose proxy (full Gaze360 not bundled)",
    }


# ---------------------------------------------------------------------------
# Classroom test + config
# ---------------------------------------------------------------------------

def _write_smoke_config() -> Path:
    src = ROOT / "configs" / "pipeline.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["models"]["scb_weights"] = "checkpoints/scb_yolo.pt"
    cfg["models"]["fusion_weights"] = "checkpoints/fusion_daisee.pt"
    cfg["models"]["gaze_weights"] = "checkpoints/l2cs_gaze360_resnet50.safetensors"
    out = ROOT / "configs" / "pipeline_smoke.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
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


def _overall_model_metrics(all_metrics: dict) -> dict:
    """Weighted overall accuracy for the production pipeline components.

    Uses official SCB + DAiSEE fusion + L2CS (not the tiny smoke YOLO dry-run).
    Weights follow support size when available.
    """
    parts = []
    for key, weight_key in (
        ("scb_official_yolo", "support"),
        ("daisee_fusion", "support"),
        ("l2cs_gaze", "support"),
    ):
        m = all_metrics.get(key)
        if not m or "accuracy" not in m:
            continue
        w = float(m.get(weight_key) or 1)
        parts.append((key, float(m["accuracy"]), float(m["precision"]), float(m["recall"]), float(m["f1"]), w))
    if not parts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "components": []}
    tw = sum(p[5] for p in parts)
    return {
        "precision": sum(p[2] * p[5] for p in parts) / tw,
        "recall": sum(p[3] * p[5] for p in parts) / tw,
        "f1": sum(p[4] * p[5] for p in parts) / tw,
        "accuracy": sum(p[1] * p[5] for p in parts) / tw,
        "macro_accuracy": float(np.mean([p[1] for p in parts])),
        "components": [
            {"name": p[0], "accuracy": p[1], "precision": p[2], "recall": p[3], "f1": p[4], "weight": p[5]}
            for p in parts
        ],
        "note": "Weighted by validation support. Production SCB + DAiSEE + L2CS only.",
    }


def _format_full_report(all_metrics: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("FULL VALIDATION REPORT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Datasets")
    lines.append("  SCB images     : 40 (32 train / 8 val)")
    lines.append("  DAiSEE clips   : 20 (16 train / 4 val)")
    lines.append("  L2CS faces     : 40 (eval)")
    lines.append("  Benchmarks     : SCB @ IoU 0.5 | DAiSEE engagement 0-3 | L2CS angular MAE")
    lines.append("")

    def block(title: str, m: dict | None) -> None:
        lines.append("-" * 72)
        lines.append(title)
        lines.append("-" * 72)
        if not m:
            lines.append("  (not available)")
            lines.append("")
            return
        lines.append(f"  Precision : {m.get('precision', 0)*100:6.2f}%")
        lines.append(f"  Recall    : {m.get('recall', 0)*100:6.2f}%")
        lines.append(f"  F1-score  : {m.get('f1', 0)*100:6.2f}%")
        lines.append(f"  Accuracy  : {m.get('accuracy', 0)*100:6.2f}%")
        if "support" in m:
            lines.append(f"  Support   : {m['support']}")
        if "det_tp" in m:
            lines.append(f"  TP / FP / FN : {m['det_tp']} / {m['det_fp']} / {m['det_fn']}")
        if "class_accuracy" in m:
            lines.append(
                f"  Class-only Acc (matched boxes) : {m['class_accuracy']*100:6.2f}%"
            )
        if "score_mae" in m:
            lines.append(f"  Score MAE : {m['score_mae']:.3f}")
        if "engagement_mae" in m:
            lines.append(f"  Engagement MAE : {m['engagement_mae']:.3f}")
        if "angular_mae_deg" in m:
            lines.append(f"  Angular MAE (deg) : {m['angular_mae_deg']:.2f}")
            lines.append(f"  Yaw MAE (deg)     : {m.get('yaw_mae_deg', 0):.2f}")
            lines.append(f"  Pitch MAE (deg)   : {m.get('pitch_mae_deg', 0):.2f}")
        if "benchmark" in m:
            lines.append(f"  Protocol  : {m['benchmark']}")
        if "note" in m:
            lines.append(f"  Note      : {m['note']}")
        lines.append("")

    block("1) SCB behavior - smoke YOLO11n (tiny train dry-run)", all_metrics.get("scb_smoke_yolo"))
    block("2) SCB behavior - official pretrained YOLOv7 (used in classroom overlay)", all_metrics.get("scb_official_yolo"))
    block("3) DAiSEE engagement - AttentionFusionNet val", all_metrics.get("daisee_fusion"))
    block("4) L2CS-Net gaze - Gaze360-style eval", all_metrics.get("l2cs_gaze"))

    overall = _overall_model_metrics(all_metrics)
    all_metrics["overall"] = overall

    lines.append("=" * 72)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 72)
    lines.append(f"{'Component':<34} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Acc':>8}")
    lines.append("-" * 72)
    for name, key in (
        ("SCB smoke YOLO", "scb_smoke_yolo"),
        ("SCB official YOLO", "scb_official_yolo"),
        ("DAiSEE fusion", "daisee_fusion"),
        ("L2CS gaze", "l2cs_gaze"),
    ):
        m = all_metrics.get(key)
        if not m:
            continue
        lines.append(
            f"{name:<34} {m['precision']*100:7.2f}% {m['recall']*100:7.2f}% "
            f"{m['f1']*100:7.2f}% {m['accuracy']*100:7.2f}%"
        )
    lines.append("-" * 72)
    lines.append(
        f"{'OVERALL MODEL (weighted)':<34} {overall['precision']*100:7.2f}% "
        f"{overall['recall']*100:7.2f}% {overall['f1']*100:7.2f}% {overall['accuracy']*100:7.2f}%"
    )
    lines.append(
        f"{'OVERALL MODEL (macro avg Acc)':<34} {'':>8} {'':>8} {'':>8} {overall['macro_accuracy']*100:7.2f}%"
    )
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        f">>> OVERALL MODEL ACCURACY: {overall['accuracy']*100:.2f}% "
        f"(macro {overall['macro_accuracy']*100:.2f}%)"
    )
    lines.append("    Components: official SCB + DAiSEE fusion + L2CS (support-weighted)")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=int, default=40, help="SCB images (was 20; +20 more)")
    p.add_argument("--clips", type=int, default=20, help="DAiSEE clips (was 10; +10 more)")
    p.add_argument("--l2cs-faces", type=int, default=40, help="L2CS face samples")
    p.add_argument("--scb-epochs", type=int, default=8)
    p.add_argument("--fusion-epochs", type=int, default=12)
    p.add_argument("--clip-seconds", type=float, default=2.0)
    p.add_argument("--test-seconds", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument(
        "--from-json",
        action="store_true",
        help="Only reprint full report from runs/smoke_metrics.json",
    )
    args = p.parse_args()

    if args.from_json:
        path = ROOT / "runs" / "smoke_metrics.json"
        all_metrics = json.loads(path.read_text(encoding="utf-8"))
        report = _format_full_report(all_metrics)
        print(report)
        out_txt = ROOT / "runs" / "validation_report.txt"
        out_txt.write_text(report, encoding="utf-8")
        path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        print(f"\nWrote {out_txt}")
        return

    SMOKE.mkdir(parents=True, exist_ok=True)
    all_metrics: dict = {}

    # --- SCB ---
    data_yaml = _copy_scb_subset(n_images=args.images, seed=args.seed)
    scb_smoke = ROOT / "checkpoints" / "scb_yolo_smoke.pt"
    if not args.skip_train:
        _train_scb_yolo(data_yaml, epochs=args.scb_epochs, out_ckpt=scb_smoke)

    print("\nValidating SCB smoke YOLO on held-out images...")
    if scb_smoke.exists():
        m_scb_smoke = _eval_scb_detection(scb_smoke)
        _print_metrics("SCB behavior (smoke YOLO11n @ IoU 0.5)", m_scb_smoke)
        all_metrics["scb_smoke_yolo"] = m_scb_smoke

    official = ROOT / "checkpoints" / "scb_yolo.pt"
    if official.exists():
        print("\nValidating official SCB YOLOv7 weights on the same val images...")
        m_scb_off = _eval_scb_detection(official)
        _print_metrics("SCB behavior (official pretrained @ IoU 0.5)", m_scb_off)
        all_metrics["scb_official_yolo"] = m_scb_off

    # --- DAiSEE fusion ---
    clips = _pick_daisee_clips(n_clips=args.clips, seed=args.seed)
    print(f"\nDAiSEE clips: {len(clips)}")
    for path, lab in clips:
        print(f"  {path.name}  engagement={lab['engagement']}")
    n_val = max(4, args.clips // 5)
    val_clips = clips[:n_val]
    train_clips = clips[n_val:]
    print(f"Fusion split: {len(train_clips)} train / {len(val_clips)} val")

    fusion_ckpt = ROOT / "checkpoints" / "fusion_smoke.pt"
    if not args.skip_train:
        x_tr, s_tr, a_tr = _build_fusion_from_clips(train_clips, seconds=args.clip_seconds)
        _train_fusion(x_tr, s_tr, a_tr, epochs=args.fusion_epochs, out_ckpt=fusion_ckpt)
        np.savez(SMOKE / "fusion_smoke.npz", features=x_tr, score=s_tr, aux=a_tr)

    x_va, s_va, a_va = _build_fusion_from_clips(val_clips, seconds=args.clip_seconds)
    if fusion_ckpt.exists():
        m_fus = _eval_fusion(fusion_ckpt, x_va, s_va, a_va)
        _print_metrics("DAiSEE engagement (fusion val)", m_fus)
        all_metrics["daisee_fusion"] = m_fus

    # --- L2CS ---
    l2cs_dir = SMOKE / "l2cs"
    if not (l2cs_dir / "labels.json").exists() or not args.skip_train:
        l2cs_dir = _build_l2cs_samples(n_faces=args.l2cs_faces, seed=args.seed)
    elif len(json.loads((l2cs_dir / "labels.json").read_text(encoding="utf-8"))) < args.l2cs_faces:
        l2cs_dir = _build_l2cs_samples(n_faces=args.l2cs_faces, seed=args.seed)
    m_l2cs = _eval_l2cs(l2cs_dir)
    _print_metrics("L2CS-Net gaze (40 faces, Gaze360-style)", m_l2cs)
    all_metrics["l2cs_gaze"] = m_l2cs

    report = _format_full_report(all_metrics)
    print("\n" + report)

    out_json = ROOT / "runs" / "smoke_metrics.json"
    out_txt = ROOT / "runs" / "validation_report.txt"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    out_txt.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_txt}")

    config = _write_smoke_config()
    if not args.skip_test:
        out_video = ROOT / "runs" / "attention_overlay_smoke.mp4"
        _run_classroom_test(config, seconds=args.test_seconds, out_video=out_video)
        print(f"\nClassroom overlay: {out_video}")


if __name__ == "__main__":
    main()
