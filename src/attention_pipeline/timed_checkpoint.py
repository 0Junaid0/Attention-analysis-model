"""Write a resumable training checkpoint on a wall-clock interval."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch


def resume_path_for(out: Path) -> Path:
    out = Path(out)
    return out.with_name(out.stem + ".resume.pt")


class TimedCheckpoint:
    """Save optimizer state every N minutes so a power cut can resume."""

    def __init__(self, path: Path, minutes: float = 30) -> None:
        self.path = Path(path)
        self.interval = max(float(minutes), 0.0) * 60.0
        self._last = time.monotonic()

    def due(self) -> bool:
        if self.interval <= 0:
            return True
        return (time.monotonic() - self._last) >= self.interval

    def save(self, payload: dict, reason: str | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(self.path)
        self.mark()
        label = reason or f"{self.interval / 60:.0f} min"
        print(f"Checkpoint saved ({label}) -> {self.path}")

    def mark(self) -> None:
        self._last = time.monotonic()

    def maybe_save(self, payload_fn: Callable[[], dict]) -> None:
        if self.due():
            self.save(payload_fn())

    def load(self, map_location: str | torch.device = "cpu") -> dict | None:
        if not self.path.exists():
            return None
        try:
            return torch.load(self.path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(self.path, map_location=map_location)
