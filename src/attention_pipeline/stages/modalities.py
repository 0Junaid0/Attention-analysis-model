"""Stage 3: face crop + landmarks / pose / gaze / emotion / motion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as TF

from attention_pipeline.models.l2cs_net import L2CSNet
from attention_pipeline.models.posture_mlp import PostureMLP, normalize_keypoints
from attention_pipeline.types import EMOTION_LABELS, POSTURE_LABELS


@dataclass
class FaceSignals:
    head_pose: np.ndarray
    gaze: np.ndarray
    emotion_probs: np.ndarray
    motion: float
    eye_aspect: float
    mouth_aspect: float
    landmarks: np.ndarray | None = None


class FaceModalityExtractor:
    """Geometric head pose from YOLO keypoints, L2CS-Net gaze on the face crop.

    FER2013 / RAF-DB are not used; emotion stays a neutral placeholder.
    """

    def __init__(
        self,
        gaze: L2CSNet | None,
        device: torch.device,
        backend: str = "stub",
        min_face_px: int = 32,
        use_l2cs: bool = False,
    ) -> None:
        self.gaze = gaze
        self.device = device
        self.backend = backend
        self.min_face_px = min_face_px
        self.use_l2cs = use_l2cs and gaze is not None
        self._prev_landmarks: dict[int, np.ndarray] = {}
        self._osf = None
        if backend == "openseeface":
            self._try_openseeface()

    def _try_openseeface(self) -> None:
        try:
            import importlib

            importlib.import_module("OpenSeeFace")
            self._osf = True
        except Exception:
            self._osf = None

    def extract(self, frame: np.ndarray, det, student_id: int) -> FaceSignals:
        crop, face_box = self._crop_face(frame, det.bbox_xyxy, det.keypoints)
        landmarks = self._landmarks(crop, det.keypoints, face_box)
        head_pose = self._head_pose(landmarks, det.keypoints)
        eye_aspect, mouth_aspect = self._aspects(landmarks)
        motion = self._motion(student_id, landmarks)

        emotion = np.zeros(len(EMOTION_LABELS), np.float32)
        emotion[-1] = 1.0  # FER2013/RAF-DB are not used
        gaze = head_pose[:2].copy()
        if self.use_l2cs and crop is not None:
            tensor = self._to_tensor(crop)
            gaze = self.gaze.predict_degrees(tensor).squeeze(0).cpu().numpy()

        return FaceSignals(
            head_pose=head_pose,
            gaze=gaze.astype(np.float32),
            emotion_probs=emotion.astype(np.float32),
            motion=motion,
            eye_aspect=eye_aspect,
            mouth_aspect=mouth_aspect,
            landmarks=landmarks,
        )

    def _crop_face(self, frame, bbox, kpts):
        h, w = frame.shape[:2]
        if kpts is not None and kpts.shape[0] >= 5 and kpts[0, 2] + kpts[1, 2] > 0.2:
            eyes = kpts[[0, 1, 2, 3, 4], :2]
            cx, cy = eyes.mean(axis=0)
            scale = np.linalg.norm(kpts[1, :2] - kpts[2, :2]) + 1.0
            side = max(int(scale * 3.5), self.min_face_px)
            x1, y1 = int(cx - side / 2), int(cy - side / 2)
            x2, y2 = x1 + side, y1 + side
        else:
            x1, y1, x2, y2 = bbox.astype(int)
            bh = max(y2 - y1, 1)
            y2 = y1 + int(bh * 0.45)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 - x1 < self.min_face_px or y2 - y1 < self.min_face_px:
            return None, np.array([x1, y1, x2, y2], np.float32)
        crop = frame[y1:y2, x1:x2]
        return crop, np.array([x1, y1, x2, y2], np.float32)

    def _landmarks(self, crop, kpts, face_box) -> np.ndarray:
        if kpts is None:
            return np.zeros((5, 2), np.float32)
        return kpts[:5, :2].astype(np.float32)

    def _head_pose(self, landmarks: np.ndarray, kpts: np.ndarray) -> np.ndarray:
        if kpts is None or kpts.shape[0] < 5:
            return np.zeros(3, np.float32)
        nose, leye, reye, lear, rear = kpts[0], kpts[1], kpts[2], kpts[3], kpts[4]
        mid_eye = (leye[:2] + reye[:2]) / 2.0
        yaw = np.degrees(np.arctan2(nose[0] - mid_eye[0], abs(reye[0] - leye[0]) + 1e-6))
        pitch = np.degrees(np.arctan2(nose[1] - mid_eye[1], abs(reye[0] - leye[0]) + 1e-6))
        roll = np.degrees(np.arctan2(reye[1] - leye[1], reye[0] - leye[0] + 1e-6))
        # Small profile bias only. Side-classroom views normally hide one ear.
        if lear[2] > 0.3 and rear[2] < 0.15:
            yaw = np.clip(yaw - 12, -90, 90)
        elif rear[2] > 0.3 and lear[2] < 0.15:
            yaw = np.clip(yaw + 12, -90, 90)
        return np.array([yaw, pitch, roll], dtype=np.float32)

    def _aspects(self, landmarks: np.ndarray) -> tuple[float, float]:
        if landmarks is None or len(landmarks) < 3:
            return 0.3, 0.3
        return 0.3, 0.3

    def _motion(self, student_id: int, landmarks: np.ndarray) -> float:
        prev = self._prev_landmarks.get(student_id)
        self._prev_landmarks[student_id] = landmarks.copy() if landmarks is not None else np.zeros((5, 2))
        if prev is None or landmarks is None or prev.shape != landmarks.shape:
            return 0.0
        delta = np.linalg.norm(landmarks - prev, axis=1).mean()
        return float(np.clip(delta / 15.0, 0, 1))

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        t = TF.to_tensor(rgb).unsqueeze(0).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        return (t - mean) / std


class PostureExtractor:
    def __init__(self, model: PostureMLP, device: torch.device, use_learned: bool = False) -> None:
        self.model = model
        self.device = device
        self.use_learned = use_learned

    def predict(self, keypoints: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        if not self.use_learned:
            return geometric_posture(keypoints)
        k = torch.from_numpy(keypoints[:, :2]).float().unsqueeze(0).to(self.device)
        b = torch.from_numpy(bbox).float().unsqueeze(0).to(self.device)
        x = normalize_keypoints(k, b)
        probs = self.model.predict_proba(x).squeeze(0).cpu().numpy()
        if probs.shape[0] != len(POSTURE_LABELS):
            out = np.zeros(len(POSTURE_LABELS), np.float32)
            out[: probs.shape[0]] = probs
            return out
        return probs.astype(np.float32)


def geometric_posture(keypoints: np.ndarray) -> np.ndarray:
    """6-class posture from COCO-17 keypoints when the SCB MLP is untrained."""
    probs = np.zeros(len(POSTURE_LABELS), np.float32)
    if keypoints is None or keypoints.shape[0] < 13:
        probs[1] = 1.0
        return probs
    if keypoints.shape[1] < 3:
        padded = np.zeros((keypoints.shape[0], 3), np.float32)
        padded[:, : keypoints.shape[1]] = keypoints
        padded[:, 2] = 1.0
        keypoints = padded
    nose, leye, reye = keypoints[0], keypoints[1], keypoints[2]
    l_sh, r_sh = keypoints[5], keypoints[6]
    l_wr, r_wr = keypoints[9], keypoints[10]
    l_hip, r_hip = keypoints[11], keypoints[12]
    sh_y = (l_sh[1] + r_sh[1]) / 2.0
    hip_y = (l_hip[1] + r_hip[1]) / 2.0
    head_gap = sh_y - nose[1]
    looking_down = nose[2] > 0.2 and head_gap < 12
    resting = nose[2] > 0.2 and nose[1] >= sh_y - 4
    hand_raised = (l_wr[2] > 0.2 and l_wr[1] < sh_y - 10) or (
        r_wr[2] > 0.2 and r_wr[1] < sh_y - 10
    )
    leaned = abs(((l_sh[0] + r_sh[0]) / 2) - ((l_hip[0] + r_hip[0]) / 2)) > 0.25 * max(
        abs(hip_y - sh_y), 1.0
    )
    if hand_raised:
        probs[0] = 1.0
    elif resting:
        probs[5] = 1.0
    elif looking_down:
        probs[1] = 1.0
    elif leaned:
        probs[5] = 1.0
    else:
        # Neutral seated / watching — do not force "reading".
        probs[:] = 1.0 / len(POSTURE_LABELS)
    return probs
