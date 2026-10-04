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
from attention_pipeline.timed_checkpoint import TimedCheckpoint, resume_path_for
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
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--checkpoint-minutes", type=float, default=30)
    p.add_argument("--fresh", action="store_true", help="Ignore a saved resume checkpoint")
    args = p.parse_args()
    path = Path(args.npz)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run: python scripts/train_all.py"
        )
    ds = SequenceNpz(path)
    n_val = max(1, len(ds) // 5)
    split_gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val], generator=split_gen
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    device = resolve_device("auto")
    model = AttentionFusionNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    score_loss = nn.SmoothL1Loss()
    aux_loss = nn.SmoothL1Loss()
    best = 1e9
    start_epoch = 1
    out = Path(args.out)
    ckpt = TimedCheckpoint(resume_path_for(out), minutes=args.checkpoint_minutes)
    saved = None if args.fresh else ckpt.load(map_location=device)
    if saved:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        best = float(saved.get("best", best))
        start_epoch = int(saved["epoch"]) + (1 if saved.get("epoch_finished") else 0)
        print(f"Resuming fusion training at epoch {start_epoch} from {ckpt.path}")
    else:
        print(f"Fusion checkpoint every {args.checkpoint_minutes:g} min -> {ckpt.path}")

    def _payload(epoch: int, finished: bool) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "epoch_finished": finished,
            "best": best,
        }

    for epoch in range(start_epoch, args.epochs + 1):
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
            ckpt.maybe_save(lambda epoch=epoch: _payload(epoch, False))
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
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "val": val}, out)
        ckpt.save(_payload(epoch, True), reason=f"epoch {epoch}")


if __name__ == "__main__":
    main()
