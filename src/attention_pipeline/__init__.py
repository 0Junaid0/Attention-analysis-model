"""Classroom multimodal attention pipeline."""

from attention_pipeline.pipeline import AttentionPipeline
from attention_pipeline.types import AttentionEvent, StudentFrame, TrackState

__all__ = [
    "AttentionPipeline",
    "AttentionEvent",
    "StudentFrame",
    "TrackState",
]
__version__ = "0.1.0"
