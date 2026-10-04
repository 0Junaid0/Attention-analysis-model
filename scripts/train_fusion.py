"""Train AttentionFusionNet on packed (T, 32) sequences.

NPZ keys:
  features: (N, T, 32) float32
  score: (N,) float32 in 0–100   (optional)
  aux: (N, 4) float32 DAiSEE 0–3  (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.utils import resolve_device


class SequenceNpz(Dataset):
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.x = data["features"].astype(np.float32)
        self.score = data["score"].astype(np.float32) if "score" in data else None
        self.aux = data["aux"].astype(np.float32) if "aux" in data else None
        if self.score is None and self.aux is None:
            raise ValueError("NPZ must contain 'score' and/or 'aux'")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        item = {"x": torch.from_numpy(self.x[i])}
        if self.score is not None:
            item["score"] = torch.tensor(self.score[i])
        if self.aux is not None:
            item["aux"] = torch.from_numpy(self.aux[i])
        return item


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/processed/fusion_daisee.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", default="checkpoints/fusion_daisee.pt")
    args = p.parse_args()
    path = Path(args.npz)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Build DAiSEE/UAP sequences with scripts/build_fusion_npz.py"
        )
    ds = SequenceNpz(path)
    n_val = max(1, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    device = resolve_device("auto")
    model = AttentionFusionNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    score_loss = nn.SmoothL1Loss()
    aux_loss = nn.SmoothL1Loss()
    best = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            pred_s, pred_a = model(x)
            loss = 0.0
            if "score" in batch:
                loss = loss + score_loss(pred_s, batch["score"].to(device))
            if "aux" in batch:
                loss = loss + 0.5 * aux_loss(pred_a, batch["aux"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        total = n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                pred_s, pred_a = model(x)
                if "score" in batch:
                    total += float(score_loss(pred_s, batch["score"].to(device)).item()) * x.size(0)
                    n += x.size(0)
        val = total / max(n, 1)
        print(f"epoch {epoch:03d}  val_smoothl1 {val:.3f}")
        if val <= best:
            best = val
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val": val}, args.out)


if __name__ == "__main__":
    main()
