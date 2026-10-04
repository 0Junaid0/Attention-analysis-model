"""Train PostureMLP on YOLO keypoints extracted from SCB-Dataset images.

CSV format (data/processed/scb_keypoints.csv):
  path,label,x1,y1,...,x17,y17
label is one of the SCB classes:
  hand-raising, reading, writing, using-phone, bowing-head, leaning-over-table

The overlay prefers the SCB YOLO detector (scripts/setup_scb_l2cs.py).
This MLP is only a keypoint fallback. FER2013 / RAF-DB are not used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attention_pipeline.models.posture_mlp import PostureMLP
from attention_pipeline.types import POSTURE_LABELS
from attention_pipeline.utils import resolve_device

LABEL_TO_IDX = {n: i for i, n in enumerate(POSTURE_LABELS)}


class KeypointCsv(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                xs = np.array([float(row[f"x{i}"]) for i in range(1, 18)], np.float32)
                ys = np.array([float(row[f"y{i}"]) for i in range(1, 18)], np.float32)
                feat = np.stack([xs, ys], axis=1).reshape(-1)
                self.rows.append((feat, LABEL_TO_IDX[row["label"]]))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x, y = self.rows[i]
        return torch.from_numpy(x), torch.tensor(y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/processed/scb_keypoints.csv")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="checkpoints/posture_scb.pt")
    args = p.parse_args()
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Extract YOLO11-Pose keypoints from SCB images first."
        )
    ds = KeypointCsv(path)
    n_val = max(1, len(ds) // 5)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    device = resolve_device("auto")
    model = PostureMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        correct = n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += int((pred == y).sum().item())
                n += y.size(0)
        acc = correct / max(n, 1)
        print(f"epoch {epoch:03d}  val_acc {acc:.3f}")
        if acc >= best:
            best = acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val_acc": acc}, args.out)


if __name__ == "__main__":
    main()
