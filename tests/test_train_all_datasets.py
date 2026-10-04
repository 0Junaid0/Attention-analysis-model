"""The full training entry point sees SCB, L2CS weights, and the DAiSEE train split."""

import importlib.util
from pathlib import Path

from attention_pipeline.utils import resolve_l2cs_weights


def _train_all():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_all.py"
    spec = importlib.util.spec_from_file_location("train_all", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_three_datasets_are_the_full_sets() -> None:
    train_all = _train_all()
    images = train_all.scb_all_images()
    train_only = train_all.daisee_split("Train")
    clips = train_all.daisee_all_clips()
    assert len(images) > 40
    assert any("images/val" in p.as_posix() or "images\\val" in str(p) for p in images)
    assert len(clips) > len(train_only)
    assert resolve_l2cs_weights().exists()
    yaml_path = train_all.write_scb_all_yaml()
    listing = (yaml_path.parent / "all_images.txt").read_text(encoding="utf-8").splitlines()
    assert len(listing) == len(images)
