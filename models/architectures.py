"""
Model Architectures Module  —  MediaPipe Face Mesh Edition
===========================================================

Pipeline
--------
    VideoFrame
        │
        ▼
    FaceMeshExtractor          ← MediaPipe Face Mesh (CPU-optimised)
        │  478 3-D landmarks
        ▼
    LandmarkGeometry           ← EAR · MAR · Head-pose · Blink-rate
        │  per-frame scalar features
        ▼
    PerclosTracker             ← Rolling-window PERCLOS + microsleep detector
        │  temporal features
        ▼
    FusionScorer               ← Weighted, thresholded fusion → [0,1] drowsiness
        │
        ▼
    DrowsinessState            ← Enum: ALERT / DROWSY / MICROSLEEP

Design goals
------------
* Zero GPU dependency — fully CPU-bound; MediaPipe is already SIMD-optimised.
* No model training required — geometry is the signal; thresholds are calibrated.
* Sub-20 ms per frame on a modern laptop CPU at 640×480.
* Self-calibrating EAR/MAR baselines via exponential moving average.
* Structured, typed, fully docstring-annotated — ready for IEEE-style publication.

Landmark index references
-------------------------
MediaPipe Face Mesh uses the canonical 468-landmark topology.
Indices used here follow the official map:
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

Author : Asad
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, NamedTuple, Optional, Sequence, Tuple
# add alongside the existing torch/torchvision imports
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency in this environment
    mp = None

# ──────────────────────────────────────────────────────────────────────────────
# Public surface
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "DrowsinessState",
    "FrameFeatures",
    "FaceMeshExtractor",
    "LandmarkGeometry",
    "PerclosTracker",
    "FusionScorer",
    "DrowsinessDetector",
    "DrowsinessModel",
    "EyeModel",
    "EnsembleModel",
]


class _SimpleConvModel(nn.Module):
    """Small convolutional model used as a training fallback."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DrowsinessModel(nn.Module):
    """Training-compatible face model wrapper."""

    def __init__(self, backbone: Optional[str] = None, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone_name = backbone or "simple_cnn"
        self.network = _SimpleConvModel(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EyeModel(nn.Module):
    """Training-compatible eye-crop model wrapper."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        backbone.classifier[3] = nn.Linear(
            backbone.classifier[3].in_features, num_classes
        )
        self.network = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EnsembleModel(nn.Module):
    """
    Weighted ensemble of DrowsinessModel (face crop) and EyeModel (eye crop).

    Parameters
    ----------
    face_model : DrowsinessModel
        Trained face-level CNN.
    eye_model : EyeModel or None
        Trained eye-crop CNN.  Pass None to run face model only.
    face_weight : float
        Relative weight applied to face model logits.
    eye_weight : float
        Relative weight applied to eye model logits.
        Weights are normalised internally so they need not sum to 1.
    """

    def __init__(
        self,
        face_model:  DrowsinessModel,
        eye_model:   Optional[EyeModel] = None,
        face_weight: float = 0.6,
        eye_weight:  float = 0.4,
    ) -> None:
        super().__init__()
        self.face_model  = face_model
        self.eye_model   = eye_model
        self.face_weight = face_weight
        self.eye_weight  = eye_weight

    def forward(
        self,
        face_tensor: torch.Tensor,
        eye_tensor:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return weighted ensemble logits, shape (B, num_classes)."""
        face_logits = self.face_model(face_tensor)

        if self.eye_model is not None and eye_tensor is not None:
            eye_logits = self.eye_model(eye_tensor)
            total_w    = self.face_weight + self.eye_weight
            return (
                self.face_weight * face_logits
                + self.eye_weight * eye_logits
            ) / max(total_w, 1e-6)

        return face_logits

    @torch.no_grad()
    def predict(
        self,
        face_tensor: torch.Tensor,
        eye_tensor:  Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """
        Run inference and return (drowsy_probability, raw_logits).

        Parameters
        ----------
        face_tensor : torch.Tensor
            Shape (1, 3, H, W) — preprocessed face crop.
        eye_tensor : torch.Tensor or None
            Shape (1, 3, H, W) — preprocessed eye crop.  Optional.

        Returns
        -------
        drowsy_prob : float
            Probability of the drowsy class in [0, 1].
        logits : torch.Tensor
            Raw ensemble logits, shape (1, num_classes).
        """
        logits      = self.forward(face_tensor, eye_tensor)
        probs       = torch.softmax(logits, dim=1)
        # Convention: class 0 = awake, class 1 = drowsy
        drowsy_prob = float(probs[0, 1].item())
        return drowsy_prob, logits


# ──────────────────────────────────────────────────────────────────────────────
# Landmark index constants  (MediaPipe Face Mesh 468-point topology)
# ──────────────────────────────────────────────────────────────────────────────

# Eye landmarks  — 6 points per eye (P1…P6, clockwise from outer canthus)
# P1=outer, P2=upper-outer, P3=upper-inner, P4=inner, P5=lower-inner, P6=lower-outer
_LEFT_EYE_IDX:  Tuple[int, ...] = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE_IDX: Tuple[int, ...] = (33,  160, 158, 133, 153, 144)

# Iris centres (used for gaze-deviation, optional refinement)
_LEFT_IRIS_IDX:  int = 468   # only present when refine_landmarks=True
_RIGHT_IRIS_IDX: int = 473

# Mouth landmarks — 8 points: 4 horizontal + 4 vertical
# Outer lip: 61(left), 291(right), 0(top-centre), 17(bottom-centre)
# Inner lip: 78(left-inner), 308(right-inner), 13(top-inner), 14(bottom-inner)
_MOUTH_IDX: Tuple[int, ...] = (61, 291, 0, 17, 78, 308, 13, 14)

# Head-pose reference points (6-point PnP)
_POSE_IDX: Tuple[int, ...] = (
    1,    # Nose tip
    152,  # Chin
    226,  # Left eye left corner
    446,  # Right eye right corner
    57,   # Left mouth corner
    287,  # Right mouth corner
)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

class DrowsinessState(Enum):
    """Discrete driver state label."""
    ALERT       = auto()
    DROWSY      = auto()
    MICROSLEEP  = auto()


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    """
    All per-frame scalar features extracted from a single video frame.

    Attributes
    ----------
    ear : float
        Eye Aspect Ratio — average of both eyes.  Range ≈ [0.0, 0.5].
        Values below ~0.20 indicate closed/closing eyes.
    ear_left : float
        Per-eye EAR for the left eye.
    ear_right : float
        Per-eye EAR for the right eye.
    mar : float
        Mouth Aspect Ratio.  Range ≈ [0.0, 1.0].
        Values above ~0.50 indicate an open/yawning mouth.
    perclos : float
        Percentage of Eye Closure over the current rolling window.
        Range [0.0, 1.0].  Values above 0.15 (15 %) indicate drowsiness.
    blink_rate : float
        Blinks per minute estimated over the rolling window.
    yaw : float
        Head yaw angle in degrees (positive = right turn).
    pitch : float
        Head pitch angle in degrees (positive = nod down).
    roll : float
        Head roll angle in degrees.
    microsleep_flag : bool
        True when eyes have been continuously closed for ≥ threshold.
    timestamp : float
        unix timestamp (seconds) of this frame.
    face_detected : bool
        Whether a face was found in the frame.
    """
    ear:             float
    ear_left:        float
    ear_right:       float
    mar:             float
    perclos:         float
    blink_rate:      float
    yaw:             float
    pitch:           float
    roll:            float
    microsleep_flag: bool
    timestamp:       float
    face_detected:   bool = True


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Final per-frame output returned to the application layer."""
    state:           DrowsinessState
    drowsiness_prob: float           # [0, 1]
    features:        FrameFeatures
    breakdown: Dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Fast 2-D or 3-D Euclidean distance."""
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(landmarks: np.ndarray, eye_idx: Sequence[int]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for one eye.

    EAR = (‖P2-P6‖ + ‖P3-P5‖) / (2 · ‖P1-P4‖)

    References
    ----------
    Soukupová & Čech (2016). "Real-Time Eye Blink Detection using Facial
    Landmarks." CVWW.

    Parameters
    ----------
    landmarks : np.ndarray, shape (N, 2) or (N, 3)
        All face landmark coordinates (pixel or normalised).
    eye_idx : sequence of 6 ints
        Indices into `landmarks` in the canonical P1…P6 order.

    Returns
    -------
    float
        EAR value in [0, ∞).  Typical open-eye ≈ 0.28–0.35.
    """
    p = landmarks[list(eye_idx)]  # (6, 2or3)
    # Vertical distances
    v1 = _euclidean(p[1], p[5])   # P2–P6
    v2 = _euclidean(p[2], p[4])   # P3–P5
    # Horizontal distance
    h  = _euclidean(p[0], p[3])   # P1–P4
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def _mouth_aspect_ratio(landmarks: np.ndarray, mouth_idx: Sequence[int]) -> float:
    """
    Compute Mouth Aspect Ratio (MAR).

    MAR = (‖P3-P8‖ + ‖P4-P7‖) / (2 · ‖P1-P2‖)

    where P1/P2 are outer horizontal corners, P3-P8 are vertical pairs
    on inner/outer lip.

    References
    ----------
    Noureddin et al. (2012). "A non-intrusive driver drowsiness monitoring
    system based on real-time detection of yawning."

    Parameters
    ----------
    landmarks : np.ndarray
        Full face landmark array.
    mouth_idx : sequence of 8 ints
        [left-outer, right-outer, top-centre-outer, bottom-centre-outer,
         left-inner, right-inner, top-centre-inner, bottom-centre-inner]

    Returns
    -------
    float
        MAR value.  Yawning ≈ > 0.50.
    """
    p = landmarks[list(mouth_idx)]  # (8, 2or3)
    # Horizontal
    h  = _euclidean(p[0], p[1])    # left-outer → right-outer
    # Vertical (outer pair + inner pair)
    v1 = _euclidean(p[2], p[3])    # top → bottom outer
    v2 = _euclidean(p[6], p[7])    # top → bottom inner
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Face Mesh Extractor
# ──────────────────────────────────────────────────────────────────────────────

class FaceMeshExtractor:
    """
    Thin, resource-managed wrapper around MediaPipe FaceMesh.

    CPU-optimisation notes
    ----------------------
    * ``static_image_mode=False``  — enables tracking between frames, which
      skips the expensive face-detection step on most frames.
    * ``refine_landmarks=True``    — adds 10 iris landmarks (468…477) for
      optional gaze / blink-precision improvements.
    * ``min_detection_confidence`` and ``min_tracking_confidence`` are tuned
      for speed vs. accuracy on CPU.

    Usage
    -----
    Use as a context manager to guarantee MediaPipe resource cleanup::

        with FaceMeshExtractor() as extractor:
            coords = extractor.process(bgr_frame)
    """

    def __init__(
        self,
        max_num_faces:           int   = 1,
        refine_landmarks:        bool  = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence:  float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._refine = refine_landmarks

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "FaceMeshExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release MediaPipe FaceMesh resources."""
        self._face_mesh.close()

    # ── public API ───────────────────────────────────────────────────────────

    def process(
        self,
        bgr_frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Run MediaPipe FaceMesh on one BGR frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            H×W×3 uint8 OpenCV frame (BGR colour order).

        Returns
        -------
        np.ndarray or None
            Float32 array of shape (N_landmarks, 3) with (x_px, y_px, z_px)
            for the first detected face, or ``None`` if no face found.
            ``z`` is the depth relative to the face centre (not in pixels).
        """
        h, w = bgr_frame.shape[:2]
        # MediaPipe requires RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        # Convert normalised [0,1] coords to pixel coords for the first face
        lm = results.multi_face_landmarks[0].landmark
        coords = np.array(
            [[p.x * w, p.y * h, p.z * w] for p in lm],
            dtype=np.float32,
        )
        return coords  # (468 or 478, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Landmark Geometry  (EAR, MAR, Head-pose)
# ──────────────────────────────────────────────────────────────────────────────

class LandmarkGeometry:
    """
    Stateful geometry engine: computes EAR, MAR, and head-pose from landmarks.

    Self-calibration
    ----------------
    The first ``calibration_frames`` frames with a detected face are used to
    compute a personal EAR baseline via Exponential Moving Average (EMA).
    This makes the EAR threshold robust to between-subject variability
    (e.g., eye shape, glasses, ethnicity).

    Head-pose via solvePnP
    ----------------------
    Uses OpenCV's ``SOLVEPNP_ITERATIVE`` with a 3-D reference face model
    to extract yaw/pitch/roll Euler angles in degrees.  No GPU required.
    """

    # 3-D reference model for the 6 pose landmarks (in mm, canonical face)
    _FACE_3D = np.array(
        [
            [0.0,    0.0,    0.0   ],   # Nose tip
            [0.0,   -63.6, -12.5  ],   # Chin
            [-43.3,  32.7, -26.0  ],   # Left eye left corner
            [ 43.3,  32.7, -26.0  ],   # Right eye right corner
            [-28.9, -28.9, -24.1  ],   # Left mouth corner
            [ 28.9, -28.9, -24.1  ],   # Right mouth corner
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        frame_width:         int   = 640,
        frame_height:        int   = 480,
        calibration_frames:  int   = 60,
        ema_alpha:           float = 0.05,
    ) -> None:
        self._w = frame_width
        self._h = frame_height
        self._calib_n    = calibration_frames
        self._alpha      = ema_alpha

        # Running EMA baseline for EAR (used for adaptive threshold)
        self._ear_baseline: Optional[float] = None
        self._calib_count: int = 0

        # Camera intrinsic matrix (pinhole approximation)
        focal  = frame_width
        cx, cy = frame_width / 2.0, frame_height / 2.0
        self._cam_matrix = np.array(
            [[focal, 0,  cx],
             [0, focal,  cy],
             [0,     0,  1.0]],
            dtype=np.float64,
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ── EAR calibration ──────────────────────────────────────────────────────

    def _update_ear_baseline(self, ear: float) -> None:
        """EMA update of the open-eye EAR baseline during calibration phase."""
        if self._calib_count < self._calib_n:
            if self._ear_baseline is None:
                self._ear_baseline = ear
            else:
                self._ear_baseline = (
                    self._alpha * ear + (1.0 - self._alpha) * self._ear_baseline
                )
            self._calib_count += 1

    @property
    def ear_threshold(self) -> float:
        """
        Adaptive EAR closure threshold.

        Returns 80 % of the calibrated open-eye baseline.
        Falls back to 0.20 (literature default) before calibration completes.
        """
        if self._ear_baseline is not None and self._calib_count >= self._calib_n:
            return 0.80 * self._ear_baseline
        return 0.20  # literature default (Soukupová & Čech 2016)

    # ── head pose ────────────────────────────────────────────────────────────

    def _head_pose(
        self,
        landmarks: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Estimate yaw, pitch, roll via solvePnP.

        Returns
        -------
        Tuple[float, float, float]
            (yaw_deg, pitch_deg, roll_deg).
            Returns (0, 0, 0) if solvePnP fails.
        """
        pts_2d = landmarks[list(_POSE_IDX), :2].astype(np.float64)
        ok, rvec, _ = cv2.solvePnP(
            self._FACE_3D,
            pts_2d,
            self._cam_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix to Euler angles
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.degrees(math.atan2( rmat[2, 1], rmat[2, 2]))
            yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll  = math.degrees(math.atan2( rmat[1, 0], rmat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
            yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
            roll  = 0.0

        return yaw, pitch, roll

    # ── main API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        landmarks: np.ndarray,
    ) -> Dict[str, float]:
        """
        Extract all geometry features from a landmark array.

        Parameters
        ----------
        landmarks : np.ndarray, shape (N, 3)
            Pixel-space landmark coordinates from FaceMeshExtractor.

        Returns
        -------
        dict with keys:
            ear, ear_left, ear_right, mar, yaw, pitch, roll
        """
        ear_l = _eye_aspect_ratio(landmarks, _LEFT_EYE_IDX)
        ear_r = _eye_aspect_ratio(landmarks, _RIGHT_EYE_IDX)
        ear   = (ear_l + ear_r) / 2.0
        mar   = _mouth_aspect_ratio(landmarks, _MOUTH_IDX)
        yaw, pitch, roll = self._head_pose(landmarks)

        self._update_ear_baseline(ear)

        return {
            "ear":       ear,
            "ear_left":  ear_l,
            "ear_right": ear_r,
            "mar":       mar,
            "yaw":       yaw,
            "pitch":     pitch,
            "roll":      roll,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  PERCLOS Tracker
# ──────────────────────────────────────────────────────────────────────────────

class PerclosTracker:
    """
    Rolling-window PERCLOS and microsleep detector.

    PERCLOS Definition
    ------------------
    PERCLOS (PERcentage of eye CLOSure) is defined as the proportion of
    frames in the past ``window_seconds`` during which the eyes are ≥ 80 %
    closed (i.e. EAR < threshold).

    References
    ----------
    Wierwille & Ellsworth (1994). "Evaluation of driver drowsiness by
    trained raters." Accident Analysis & Prevention, 26(5), 571–581.

    Microsleep Detection
    --------------------
    A microsleep event is flagged when the eyes remain continuously closed
    for ``microsleep_min_seconds`` (default 0.5 s).

    Parameters
    ----------
    fps : float
        Nominal camera frame rate.  Used to convert seconds to frame counts.
    window_seconds : float
        Duration of the rolling window for PERCLOS.  Default 60 s.
    microsleep_min_seconds : float
        Minimum continuous eye-closure duration to count as microsleep.
    """

    def __init__(
        self,
        fps:                    float = 30.0,
        window_seconds:         float = 60.0,
        microsleep_min_seconds: float = 0.50,
    ) -> None:
        self._fps        = max(fps, 1.0)
        self._win_frames = int(window_seconds * fps)
        self._micro_thresh = int(microsleep_min_seconds * fps)

        # Circular buffer: 1 = eye closed, 0 = eye open
        self._closure_buf: Deque[int] = deque(maxlen=self._win_frames)
        # Blink event timestamps (unix seconds)
        self._blink_times: Deque[float] = deque(maxlen=self._win_frames)

        self._consecutive_closed: int  = 0
        self._prev_closed:        bool = False
        self._in_blink:           bool = False

    def update(
        self,
        ear:       float,
        threshold: float,
        timestamp: float,
    ) -> Tuple[float, bool, float]:
        """
        Update internal state with one new frame.

        Parameters
        ----------
        ear : float
            Current frame EAR.
        threshold : float
            EAR closure threshold (from LandmarkGeometry.ear_threshold).
        timestamp : float
            Unix timestamp of this frame.

        Returns
        -------
        perclos : float
            PERCLOS over the rolling window [0, 1].
        microsleep : bool
            True if a microsleep event is currently active.
        blink_rate : float
            Estimated blinks-per-minute over the rolling window.
        """
        closed = ear < threshold

        # ── consecutive closure counter (microsleep) ──────────────────────
        if closed:
            self._consecutive_closed += 1
        else:
            self._consecutive_closed = 0

        microsleep = self._consecutive_closed >= self._micro_thresh

        # ── blink detection (rising edge of closure) ──────────────────────
        if closed and not self._prev_closed:
            self._blink_times.append(timestamp)

        self._prev_closed = closed

        # ── PERCLOS buffer ────────────────────────────────────────────────
        self._closure_buf.append(int(closed))
        perclos = (
            sum(self._closure_buf) / len(self._closure_buf)
            if self._closure_buf
            else 0.0
        )

        # ── blink rate ────────────────────────────────────────────────────
        cutoff = timestamp - 60.0
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        blink_rate = float(len(self._blink_times))  # blinks in last 60 s ≡ BPM

        return perclos, microsleep, blink_rate


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Fusion Scorer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FusionWeights:
    """
    Tuneable weights and thresholds for the fusion scorer.

    All weights are normalised internally; they only define relative importance.
    """
    # Feature weights
    w_perclos:  float = 0.35
    w_ear:      float = 0.25
    w_mar:      float = 0.15
    w_blink:    float = 0.10
    w_pose:     float = 0.10
    w_micro:    float = 0.05

    # Thresholds
    ear_threshold:     float = 0.20  # overridden by LandmarkGeometry adaptive value
    mar_threshold:     float = 0.50  # yawning
    perclos_drowsy:    float = 0.15  # 15 % → DROWSY boundary (NHTSA standard)
    perclos_alert:     float = 0.08  # < 8 % → ALERT
    pose_yaw_limit:    float = 25.0  # degrees
    pose_pitch_limit:  float = 15.0  # degrees
    # Normal blink rate range
    blink_min_bpm:     float = 8.0
    blink_max_bpm:     float = 21.0


class FusionScorer:
    """
    Multi-cue drowsiness fusion engine.

    Combines PERCLOS, EAR deviation, MAR (yawning), blink rate anomaly,
    head-pose deviation, and microsleep flag into a single [0,1] probability.

    The fusion is a weighted sum of individual normalised sub-scores,
    clamped to [0, 1] and mapped to a ``DrowsinessState``.

    State transition rules
    ----------------------
    * MICROSLEEP overrides all other states immediately.
    * DROWSY when ``drowsiness_prob ≥ 0.45``.
    * ALERT otherwise.
    """

    def __init__(self, weights: Optional[FusionWeights] = None) -> None:
        self._w = weights or FusionWeights()

    def score(
        self,
        features: FrameFeatures,
        ear_threshold: float,
    ) -> Tuple[DrowsinessState, float, Dict[str, float]]:
        """
        Compute drowsiness probability and state for one frame.

        Parameters
        ----------
        features : FrameFeatures
            Extracted per-frame features.
        ear_threshold : float
            Adaptive EAR threshold from LandmarkGeometry.

        Returns
        -------
        state : DrowsinessState
        prob  : float in [0, 1]
        breakdown : dict mapping sub-score names to their [0,1] values
        """
        w = self._w

        # ── sub-scores ────────────────────────────────────────────────────

        # 1. PERCLOS score
        s_perclos = _norm(features.perclos, w.perclos_alert, w.perclos_drowsy)

        # 2. EAR score (how far below the open-eye baseline)
        ear_delta = (ear_threshold - features.ear) / max(ear_threshold, 1e-6)
        s_ear = float(np.clip(ear_delta, 0.0, 1.0))

        # 3. MAR score (yawning)
        s_mar = _norm(features.mar, 0.30, w.mar_threshold)

        # 4. Blink rate anomaly (too slow or too fast)
        bpm = features.blink_rate
        if w.blink_min_bpm <= bpm <= w.blink_max_bpm:
            s_blink = 0.0
        elif bpm < w.blink_min_bpm:
            # Too slow: approaching 0 BPM is maximally drowsy
            s_blink = _norm(bpm, w.blink_min_bpm, 0.0, invert=True)
        else:
            # Too fast (micro-blinks): moderately concerning
            s_blink = _norm(bpm, w.blink_max_bpm, 40.0) * 0.5

        # 5. Head pose score (large deviations → inattention / nodding)
        yaw_score   = _norm(abs(features.yaw),   0.0, w.pose_yaw_limit)
        pitch_score = _norm(abs(features.pitch), 0.0, w.pose_pitch_limit)
        s_pose = float(np.clip((yaw_score + pitch_score) / 2.0, 0.0, 1.0))

        # 6. Microsleep flag
        s_micro = 1.0 if features.microsleep_flag else 0.0

        # ── weighted fusion ───────────────────────────────────────────────
        ws = w.w_perclos + w.w_ear + w.w_mar + w.w_blink + w.w_pose + w.w_micro
        prob = (
            w.w_perclos * s_perclos
            + w.w_ear   * s_ear
            + w.w_mar   * s_mar
            + w.w_blink * s_blink
            + w.w_pose  * s_pose
            + w.w_micro * s_micro
        ) / ws
        prob = float(np.clip(prob, 0.0, 1.0))

        # ── state mapping ─────────────────────────────────────────────────
        if features.microsleep_flag:
            state = DrowsinessState.MICROSLEEP
        elif prob >= 0.45:
            state = DrowsinessState.DROWSY
        else:
            state = DrowsinessState.ALERT

        breakdown = {
            "s_perclos": round(s_perclos, 4),
            "s_ear":     round(s_ear, 4),
            "s_mar":     round(s_mar, 4),
            "s_blink":   round(s_blink, 4),
            "s_pose":    round(s_pose, 4),
            "s_micro":   round(s_micro, 4),
        }
        return state, prob, breakdown


def _norm(
    x:     float,
    lo:    float,
    hi:    float,
    invert: bool = False,
) -> float:
    """
    Linearly normalise x from [lo, hi] → [0, 1], clamped.
    If ``invert=True``, the mapping is reversed (lo→1, hi→0).
    """
    if abs(hi - lo) < 1e-9:
        return 1.0 if x >= hi else 0.0
    val = (x - lo) / (hi - lo)
    val = float(np.clip(val, 0.0, 1.0))
    return 1.0 - val if invert else val


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Top-level DrowsinessDetector  (replaces old EnsembleModel)
# ──────────────────────────────────────────────────────────────────────────────

class DrowsinessDetector:
    """
    End-to-end drowsiness detection pipeline — CPU-only, no training required.

    Replaces the original CNN-based ``EnsembleModel`` with a fully
    geometry-driven pipeline using MediaPipe Face Mesh.

    Advantages over CNN ensemble on CPU
    ------------------------------------
    * MediaPipe FaceMesh with tracking runs at 25–35 FPS on a modern laptop
      CPU, versus 4–8 FPS for EfficientNetV2-S inference.
    * No dataset or training needed; geometry is a direct physiological signal.
    * Self-calibrating to each driver's anatomy via EMA baseline.
    * Interpretable — every sub-score maps to a named physiological feature.

    Usage
    -----
    ::

        detector = DrowsinessDetector(frame_width=640, frame_height=480, fps=30)

        # Optionally call as context manager for guaranteed cleanup:
        with DrowsinessDetector(...) as det:
            cap = cv2.VideoCapture(0)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = det.process_frame(frame)
                print(result.state, f"{result.drowsiness_prob:.2f}")

    Parameters
    ----------
    frame_width : int
        Camera frame width in pixels.
    frame_height : int
        Camera frame height in pixels.
    fps : float
        Camera frame rate.
    perclos_window_seconds : float
        Duration of the PERCLOS rolling window (default 60 s).
    microsleep_min_seconds : float
        Minimum continuous eye-closure for a microsleep event (default 0.5 s).
    calibration_frames : int
        Number of frames for EAR personal baseline calibration (default 60).
    fusion_weights : FusionWeights or None
        Custom fusion weights; pass ``None`` for sensible defaults.
    mediapipe_detection_conf : float
        MediaPipe face detection confidence.
    mediapipe_tracking_conf : float
        MediaPipe landmark tracking confidence.
    """

    def __init__(
        self,
        frame_width:             int   = 640,
        frame_height:            int   = 480,
        fps:                     float = 30.0,
        perclos_window_seconds:  float = 60.0,
        microsleep_min_seconds:  float = 0.50,
        calibration_frames:      int   = 60,
        fusion_weights:          Optional[FusionWeights] = None,
        mediapipe_detection_conf:  float = 0.5,
        mediapipe_tracking_conf:   float = 0.5,
    ) -> None:
        self._extractor = FaceMeshExtractor(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=mediapipe_detection_conf,
            min_tracking_confidence=mediapipe_tracking_conf,
        )
        self._geometry = LandmarkGeometry(
            frame_width=frame_width,
            frame_height=frame_height,
            calibration_frames=calibration_frames,
        )
        self._perclos = PerclosTracker(
            fps=fps,
            window_seconds=perclos_window_seconds,
            microsleep_min_seconds=microsleep_min_seconds,
        )
        self._scorer = FusionScorer(weights=fusion_weights)

        # Null FrameFeatures returned when no face is detected
        self._null_features = FrameFeatures(
            ear=0.0, ear_left=0.0, ear_right=0.0,
            mar=0.0, perclos=0.0, blink_rate=0.0,
            yaw=0.0, pitch=0.0, roll=0.0,
            microsleep_flag=False,
            timestamp=0.0,
            face_detected=False,
        )

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "DrowsinessDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release all resources."""
        self._extractor.close()

    # ── main API ─────────────────────────────────────────────────────────────

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> DetectionResult:
        """
        Process one BGR video frame and return a ``DetectionResult``.

        Parameters
        ----------
        bgr_frame : np.ndarray
            H×W×3 uint8 OpenCV frame in BGR colour order.
        timestamp : float or None
            Frame timestamp in seconds.  Uses ``time.monotonic()`` if None.

        Returns
        -------
        DetectionResult
            Contains state, drowsiness_prob, features, and sub-score breakdown.
            If no face is detected, state=ALERT with prob=0.0.
        """
        ts = timestamp if timestamp is not None else time.monotonic()

        # ── landmark extraction ───────────────────────────────────────────
        landmarks = self._extractor.process(bgr_frame)

        if landmarks is None:
            # No face detected: treat as ALERT, carry over PERCLOS silently
            return DetectionResult(
                state=DrowsinessState.ALERT,
                drowsiness_prob=0.0,
                features=self._null_features,
                breakdown={},
            )

        # ── geometry features ─────────────────────────────────────────────
        geo = self._geometry.compute(landmarks)

        # ── temporal features ─────────────────────────────────────────────
        perclos, microsleep, blink_rate = self._perclos.update(
            ear=geo["ear"],
            threshold=self._geometry.ear_threshold,
            timestamp=ts,
        )

        features = FrameFeatures(
            ear=geo["ear"],
            ear_left=geo["ear_left"],
            ear_right=geo["ear_right"],
            mar=geo["mar"],
            perclos=perclos,
            blink_rate=blink_rate,
            yaw=geo["yaw"],
            pitch=geo["pitch"],
            roll=geo["roll"],
            microsleep_flag=microsleep,
            timestamp=ts,
            face_detected=True,
        )

        # ── fusion scoring ────────────────────────────────────────────────
        state, prob, breakdown = self._scorer.score(
            features=features,
            ear_threshold=self._geometry.ear_threshold,
        )

        return DetectionResult(
            state=state,
            drowsiness_prob=prob,
            features=features,
            breakdown=breakdown,
        )

    # ── diagnostics / calibration ─────────────────────────────────────────────

    @property
    def ear_threshold(self) -> float:
        """Current adaptive EAR closure threshold."""
        return self._geometry.ear_threshold

    @property
    def calibration_complete(self) -> bool:
        """True once the EAR personal baseline has been established."""
        return self._geometry._calib_count >= self._geometry._calib_n

    def reset(self) -> None:
        """Clear the EAR calibration baseline and PERCLOS/blink/microsleep state."""
        self._geometry._ear_baseline = None
        self._geometry._calib_count = 0
        self._perclos._closure_buf.clear()
        self._perclos._blink_times.clear()
        self._perclos._consecutive_closed = 0
        self._perclos._prev_closed = False