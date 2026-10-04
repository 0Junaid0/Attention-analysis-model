from __future__ import annotations

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(path: str | Path | None) -> Path | None:
    """Resolve a path relative to the thesis project root when it is not absolute."""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def resolve_scb_root() -> Path:
    """SCB-dataset images/labels under data/raw/scb (Whiffe/SCB-dataset)."""
    base = PROJECT_ROOT / "data" / "raw" / "scb"
    for cand in (
        base / "SCB5-Handrise-Read-write" / "SCB5-Handrise-Read-write-2024-9-17",
        base / "SCB5-Handrise-Read-write",
        base,
    ):
        if (cand / "images" / "train").exists() or (cand / "images").exists():
            return cand
    matches = list(base.rglob("images/train")) if base.exists() else []
    if matches:
        return matches[0].parent.parent
    raise FileNotFoundError("SCB not found. Run: python scripts/setup_scb_l2cs.py")


def resolve_scb_yaml() -> Path:
    root = resolve_scb_root()
    yaml_path = root / "scb.yaml"
    if yaml_path.exists():
        return yaml_path
    found = list(root.rglob("scb.yaml"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Missing scb.yaml under {root}. Run: python scripts/setup_scb_l2cs.py")


def resolve_l2cs_weights() -> Path:
    p = PROJECT_ROOT / "checkpoints" / "l2cs_gaze360_resnet50.safetensors"
    if p.exists():
        return p
    raise FileNotFoundError("L2CS weights missing. Run: python scripts/setup_scb_l2cs.py")


def resolve_scb_weights() -> Path:
    for name in ("scb_yolo.pt", "scb_yolo11n.pt"):
        p = PROJECT_ROOT / "checkpoints" / name
        if p.exists():
            return p
    raise FileNotFoundError("SCB weights missing. Run: python scripts/setup_scb_l2cs.py")


def resolve_daisee_root() -> Path:
    """Return the DAiSEE folder: project data/raw/daisee, or the KaggleHub cache path."""
    dest = PROJECT_ROOT / "data" / "raw" / "daisee"
    marker = dest / "KAGGLE_SOURCE.txt"
    if dest.exists() and any(dest.rglob("*")):
        return dest
    if marker.exists():
        cached = Path(marker.read_text(encoding="utf-8").strip())
        if cached.exists():
            return cached
    raise FileNotFoundError(
        "DAiSEE not found. Run: python scripts/download_daisee_kaggle.py"
    )
