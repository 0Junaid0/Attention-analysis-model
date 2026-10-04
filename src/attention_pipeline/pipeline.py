"""End-to-end runner for the six-stage attention pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from attention_pipeline.features import to_student_frame
from attention_pipeline.rubric import apply_peer_cues, classify_attention
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.drop_detector import AttentionDropDetector, DropConfig
from attention_pipeline.models.l2cs_net import L2CSNet
from attention_pipeline.models.posture_mlp import PostureMLP
from attention_pipeline.stages.attention import AttentionScorer
from attention_pipeline.stages.detect_track import DetectorTracker
from attention_pipeline.stages.intervention import recommend
from attention_pipeline.stages.modalities import FaceModalityExtractor, PostureExtractor
from attention_pipeline.stages.scb_behavior import SCBBehaviorDetector
from attention_pipeline.stages.source import iter_frames
from attention_pipeline.types import POSTURE_LABELS, TrackState
from attention_pipeline.utils import load_yaml, project_path, resolve_device


class AttentionPipeline:
    def __init__(self, cfg: dict[str, Any], load_weights: bool = True) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", "auto"))
        models_cfg = cfg.get("models", {})
        data_cfg = cfg.get("data", {})

        self.gaze = L2CSNet().to(self.device).eval()
        self.posture = PostureMLP().to(self.device).eval()
        fusion_cfg = cfg.get("fusion", {})
        self.fusion = AttentionFusionNet(
            feat_dim=fusion_cfg.get("feat_dim", 32),
            hidden=fusion_cfg.get("hidden", 64),
        ).to(self.device).eval()

        loaded_fusion = False
        loaded_gaze = loaded_posture = False
        if load_weights:
            loaded_gaze = self._maybe_load(self.gaze, models_cfg.get("gaze_weights"))
            loaded_posture = self._maybe_load(self.posture, models_cfg.get("posture_weights"))
            loaded_fusion = self._maybe_load(self.fusion, models_cfg.get("fusion_weights"))

        detect_cfg = cfg.get("detect", {})
        track_cfg = cfg.get("track", {})
        self.detector = DetectorTracker(
            weights=detect_cfg.get("weights", "yolo11n-pose.pt"),
            person_weights=detect_cfg.get("person_weights", "yolo11n.pt"),
            conf=detect_cfg.get("conf", 0.08),
            iou=detect_cfg.get("iou", 0.45),
            device=str(self.device),
            max_age=track_cfg.get("max_age", 45),
            n_init=track_cfg.get("n_init", 1),
            imgsz=int(detect_cfg.get("imgsz", 960)),
            tiles=bool(detect_cfg.get("tiles", True)),
        )
        face_cfg = cfg.get("face", {})
        self.faces = FaceModalityExtractor(
            gaze=self.gaze,
            device=self.device,
            backend=face_cfg.get("backend", "stub"),
            min_face_px=face_cfg.get("min_face_px", 32),
            use_l2cs=loaded_gaze,
        )
        self.posture_x = PostureExtractor(self.posture, self.device, use_learned=loaded_posture)
        scb_weights = project_path(models_cfg.get("scb_weights"))
        self.scb = SCBBehaviorDetector(
            weights=scb_weights,
            device=str(self.device),
            conf=float(cfg.get("scb", {}).get("conf", 0.25)),
        )
        self.scb_data_yaml = project_path(data_cfg.get("scb_yaml"))
        self.scorer = AttentionScorer(
            self.fusion,
            self.device,
            window=fusion_cfg.get("window", 16),
        )
        self.scorer.use_heuristic = not loaded_fusion
        drop_cfg = cfg.get("drop", {})
        self.drop = AttentionDropDetector(DropConfig(**drop_cfg) if drop_cfg else DropConfig())
        self.tracks: dict[int, TrackState] = {}
        self._lesson_yaw: float | None = None
        self.dataset_status = {
            "l2cs_gaze": loaded_gaze,
            "scb_behavior": self.scb.enabled,
            "scb_yaml": bool(self.scb_data_yaml and self.scb_data_yaml.exists()),
            "fusion": loaded_fusion,
        }

    @classmethod
    def from_yaml(cls, path: str | Path, load_weights: bool = True) -> "AttentionPipeline":
        return cls(load_yaml(str(path)), load_weights=load_weights)

    def _maybe_load(self, module, path: str | None) -> bool:
        p = project_path(path)
        if p is None or not p.exists():
            return False
        module.load(p, map_location=self.device)
        return True

    def process_frame(self, timestamp: float, frame: np.ndarray) -> dict[str, Any]:
        detections = self.detector.infer(frame)
        scb_dets = self.scb.infer(frame)
        scb_cfg = self.cfg.get("scb", {})
        min_iou = float(scb_cfg.get("match_iou", 0.4))
        blend = float(scb_cfg.get("blend", 0.65))
        student_boxes = [det.bbox_xyxy for det in detections]
        scb_matches = self.scb.match_all(student_boxes, scb_dets, min_iou=min_iou)

        observed: list[tuple[TrackState, Any]] = []
        for det, scb_probs in zip(detections, scb_matches):
            track = self.tracks.setdefault(det.track_id, TrackState(student_id=det.track_id))
            track.last_bbox = det.bbox_xyxy
            face = self.faces.extract(frame, det, det.track_id)
            posture = self.posture_x.predict(det.keypoints, det.bbox_xyxy)
            if scb_probs is not None:
                # Blend SCB with geometric pose so one wrong SCB box cannot dominate.
                posture = (blend * scb_probs + (1.0 - blend) * posture).astype(np.float32)
                posture = posture / max(float(posture.sum()), 1e-6)
            obs = to_student_frame(
                student_id=det.track_id,
                timestamp=timestamp,
                bbox=det.bbox_xyxy,
                keypoints=det.keypoints,
                head_pose=face.head_pose,
                gaze=face.gaze,
                emotion_probs=face.emotion_probs,
                posture_probs=posture,
                motion=face.motion,
                eye_aspect=face.eye_aspect,
                mouth_aspect=face.mouth_aspect,
            )
            observed.append((track, obs))

        if observed:
            med = float(np.median([obs.head_pose[0] for _, obs in observed]))
            if self._lesson_yaw is None:
                self._lesson_yaw = med
            else:
                self._lesson_yaw = 0.2 * med + 0.8 * self._lesson_yaw

        per_student = []
        events = []
        pending = []
        for track, obs in observed:
            behavior = classify_attention(obs, lesson_yaw=self._lesson_yaw)
            score, rubric = self.scorer.push(
                track, obs.features, timestamp, obs=obs, lesson_yaw=self._lesson_yaw
            )
            if rubric is None:
                rubric = behavior
            if behavior.band == "LOW" and rubric.score > behavior.score:
                rubric = behavior
                score = behavior.score
                if track.scores:
                    track.scores[-1] = score
            pending.append((track, obs, score, rubric))

        peer = apply_peer_cues(
            [obs for _track, obs, _score, _rubric in pending],
            [rubric for _track, _obs, _score, rubric in pending],
            self._lesson_yaw,
        )
        for (track, obs, score, _old), rubric in zip(pending, peer):
            if rubric.score < score:
                score = rubric.score
                if track.scores:
                    track.scores[-1] = score
            band, cue = rubric.band, rubric.cue
            event = self.drop.update(track, obs.features)
            rec = recommend(event, class_mean_score=_mean_score(self.tracks)) if event else None
            if event:
                events.append({"event": event, "recommendation": rec})
            per_student.append(
                {
                    "student_id": obs.student_id,
                    "bbox": obs.bbox.tolist(),
                    "score": score,
                    "band": band,
                    "cue": cue,
                    "head_pose": obs.head_pose.tolist(),
                    "gaze": obs.gaze.tolist(),
                    "emotion": obs.emotion_probs.tolist(),
                    "posture": obs.posture_probs.tolist(),
                    "behavior": POSTURE_LABELS[int(np.argmax(obs.posture_probs))],
                }
            )
        return {
            "timestamp": timestamp,
            "students": per_student,
            "events": events,
        }

    def run_video(self, source: str | int, max_seconds: float | None = None):
        fps = float(self.cfg.get("fps", 10))
        for ts, frame in iter_frames(source, target_fps=fps):
            if max_seconds is not None and ts > max_seconds:
                break
            yield self.process_frame(ts, frame)


def _mean_score(tracks: dict[int, TrackState]) -> float | None:
    vals = [t.scores[-1] for t in tracks.values() if t.scores]
    if not vals:
        return None
    return float(np.mean(vals))
