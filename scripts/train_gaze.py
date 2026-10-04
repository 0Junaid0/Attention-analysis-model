"""Optional L2CS-Net fine-tune. Default inference uses the Gaze360 checkpoint.

NPZ keys: images (N, 3, 224, 224) float32 in [0,1], pitch (N,) deg, yaw (N,) deg

Does not use FER2013 or RAF-DB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from attention_pipeline.models.l2cs_net import L2CSNet
from attention_pipeline.utils import resolve_device


def deg_to_bin(deg: torch.Tensor) -> torch.Tensor:
    return ((deg + 180.0) / 4.0).clamp(0, 89).long()


class GazeNpz(Dataset):
    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self.x = data["images"].astype(np.float32)
        self.pitch = data["pitch"].astype(np.float32)
        self.yaw = data["yaw"].astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.x[i]),
            torch.tensor(self.pitch[i]),
            torch.tensor(self.yaw[i]),
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="data/processed/gaze_l2cs.npz")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--init", default="checkpoints/l2cs_gaze360_resnet50.safetensors")
    p.add_argument("--out", default="checkpoints/l2cs_finetuned.pt")
    args = p.parse_args()
    path = Path(args.npz)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}\n"
            "You do not need this for the classroom demo. "
            "Gaze uses checkpoints/l2cs_gaze360_resnet50.safetensors from setup_scb_l2cs.py."
        )
    ds = GazeNpz(path)
    n_val = max(1, len(ds) // 5)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    device = resolve_device("auto")
    model = L2CSNet().to(device)
    init = Path(args.init)
    if init.exists():
        model.load(init, map_location=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    best = 1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, pitch, yaw in train_loader:
            x = x.to(device)
            if x.max() > 1.5:
                x = x / 255.0
            x = (x - mean) / std
            y_log, p_log = model(x)
            loss = loss_fn(p_log, deg_to_bin(pitch.to(device))) + loss_fn(
                y_log, deg_to_bin(yaw.to(device))
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        err = n = 0
        with torch.no_grad():
            for x, pitch, yaw in val_loader:
                x = x.to(device)
                if x.max() > 1.5:
                    x = x / 255.0
                x = (x - mean) / std
                deg = model.predict_degrees(x).cpu()
                err += float((deg[:, 1] - pitch).abs().sum() + (deg[:, 0] - yaw).abs().sum())
                n += x.size(0)
        mae = err / max(n, 1) / 2.0
        print(f"epoch {epoch:03d}  val MAE {mae:.2f} deg")
        if mae <= best:
            best = mae
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "mae": mae}, args.out)


if __name__ == "__main__":
    main()
