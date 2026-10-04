"""Resume checkpoints survive an interrupted training save."""

from pathlib import Path

import torch

from attention_pipeline.timed_checkpoint import TimedCheckpoint, resume_path_for


def test_resume_path_sits_next_to_the_weights() -> None:
    assert resume_path_for(Path("checkpoints/fusion_daisee.pt")).name == "fusion_daisee.resume.pt"


def test_saves_only_after_the_interval_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "model.resume.pt"
    ckpt = TimedCheckpoint(path, minutes=30)
    assert ckpt.due() is False
    ckpt._last -= 30 * 60
    assert ckpt.due() is True
    ckpt.save({"epoch": 4, "epoch_finished": False, "w": torch.tensor([1.5])})
    loaded = ckpt.load()
    assert loaded is not None
    assert loaded["epoch"] == 4
    assert loaded["epoch_finished"] is False
    assert float(loaded["w"][0]) == 1.5
    assert ckpt.due() is False
