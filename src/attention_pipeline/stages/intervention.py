"""Stage 6: structured event → LLM recommendation payload."""

from __future__ import annotations

import json
from typing import Any

from attention_pipeline.types import AttentionEvent, EMOTION_LABELS, POSTURE_LABELS


SYSTEM_PROMPT = """You are an instructional coach sitting beside a teacher.
You receive a structured attention-drop event from a classroom vision system.
Give three short fields:
1) explanation — what the signals suggest, without naming the student publicly
2) teaching_recommendation — one in-the-moment pedagogical move
3) intervention_suggestion — whether to wait, prompt the class, or check privately
Do not invent identity, grades, or medical claims. Stay under 120 words total.
"""


def event_to_prompt(event: AttentionEvent, class_mean_score: float | None = None) -> str:
    payload = {
        "student_slot": event.student_id,
        "timestamp_s": round(event.timestamp, 2),
        "score": round(event.score, 1),
        "drop_kind": event.kind,
        "dominant_modality": event.dominant_modality,
        "features": event.explanation_features,
        "class_mean_score": class_mean_score,
        "emotion_labels": list(EMOTION_LABELS),
        "posture_labels": list(POSTURE_LABELS),
    }
    return (
        SYSTEM_PROMPT
        + "\n\nEvent JSON:\n"
        + json.dumps(payload, indent=2)
    )


def recommend(event: AttentionEvent, class_mean_score: float | None = None) -> dict[str, Any]:
    """Deterministic fallback so the pipeline runs without an API key."""
    modality = event.dominant_modality.replace("_", " ")
    wait = event.kind == "sudden" and event.score > 30
    return {
        "explanation": (
            f"Attention dropped ({event.kind}) with {modality} as the strongest cue "
            f"at score {event.score:.0f}/100."
        ),
        "teaching_recommendation": (
            "Pause for a 15-second retrieval prompt aimed at the whole class, "
            "then cold-call a nearby volunteer rather than the flagged seat."
        ),
        "intervention_suggestion": (
            "Wait one more cycle; brief dips are common."
            if wait
            else "Circulate toward that side of the room and use a quiet check-in."
        ),
        "prompt": event_to_prompt(event, class_mean_score),
    }
