"""
Face & Landmark Detection Module

Primary:  MediaPipe FaceMesh (468 landmarks, GPU/CPU, real-time)
Fallback: dlib 68-point predictor

Returns standardised landmark arrays regardless of backend.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── Optional imports (handled gracefully) ─────────────────────────────────────
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

try:
    import dlib
    _HAS_DLIB = True
except ImportError:
    _HAS_DLIB = False

from config.config import infer_cfg
from utils.logger import get_logger
from utils.geometry import (
    MP_LEFT_EYE_EAR, MP_RIGHT_EYE_EAR, MP_MOUTH_EAR,
    DLIB_LEFT_EYE, DLIB_RIGHT_EYE, DLIB_MOUTH,
)

log = get_logger(__name__)


@dataclass
class FaceResult:
    """Everything extracted from one face per frame."""
    face_bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    landmarks: np.ndarray                  # (N, 2)  pixel coords
    left_eye_pts: np.ndarray               # (6, 2)  EAR 6-point
    right_eye_pts: np.ndarray              # (6, 2)
    mouth_pts: np.ndarray                  # (8, 2)  MAR 8-point
    left_eye_crop: np.ndarray              # (H, W, 3) BGR
    right_eye_crop: np.ndarray
    face_crop: np.ndarray                  # Full face crop (H, W, 3)
    backend: str                           # "mediapipe" | "dlib"


class FaceDetector:
    """
    Wraps MediaPipe FaceMesh with optional dlib fallback.

    Usage
    -----
    detector = FaceDetector()
    results  = detector.detect(bgr_frame)   # List[FaceResult]
    """

    def __init__(self) -> None:
        self._mp_detector = None
        self._dlib_detector = None
        self._dlib_predictor = None
        self._active_backend: str = "none"

        self._init_mediapipe()
        if not _HAS_MEDIAPIPE or self._mp_detector is None:
            self._init_dlib()

    # ─── Backend initialisation ────────────────────────────────────────────────

    def _init_mediapipe(self) -> None:
        if not _HAS_MEDIAPIPE:
            log.warning("MediaPipe not installed. Run: pip install mediapipe")
            return
        try:
            face_mesh = mp.solutions.face_mesh
            self._mp_detector = face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=infer_cfg.face_detection_confidence,
                min_tracking_confidence=infer_cfg.landmark_confidence,
            )
            self._active_backend = "mediapipe"
            log.info("MediaPipe FaceMesh initialised (468 landmarks).")
        except Exception as exc:
            log.error("MediaPipe init failed: %s", exc)
            self._mp_detector = None

    def _init_dlib(self) -> None:
        if not _HAS_DLIB:
            log.warning("dlib not installed. Run: pip install dlib")
            return
        # Predictor path — standard location or next to this file
        predictor_candidates = [
            Path("shape_predictor_68_face_landmarks.dat"),
            Path(__file__).parent.parent / "models" / "shape_predictor_68_face_landmarks.dat",
        ]
        predictor_path = next((p for p in predictor_candidates if p.exists()), None)

        if predictor_path is None:
            log.warning(
                "dlib 68-point predictor not found. "
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
            return
        try:
            self._dlib_detector = dlib.get_frontal_face_detector()
            self._dlib_predictor = dlib.shape_predictor(str(predictor_path))
            self._active_backend = "dlib"
            log.info("dlib face detector + 68-point predictor loaded.")
        except Exception as exc:
            log.error("dlib init failed: %s", exc)

    # ─── Main detection call ───────────────────────────────────────────────────

    def detect(self, bgr_frame: np.ndarray) -> List[FaceResult]:
        """
        Detect faces and extract all landmarks in one call.

        Parameters
        ----------
        bgr_frame : np.ndarray  (H, W, 3) BGR frame from OpenCV.

        Returns
        -------
        List[FaceResult]  — one entry per detected face (usually 0 or 1).
        """
        if self._active_backend == "mediapipe":
            return self._detect_mediapipe(bgr_frame)
        elif self._active_backend == "dlib":
            return self._detect_dlib(bgr_frame)
        return []

    # ─── MediaPipe backend ────────────────────────────────────────────────────

    def _detect_mediapipe(self, bgr: np.ndarray) -> List[FaceResult]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._mp_detector.process(rgb)

        if not results.multi_face_landmarks:
            return []

        faces: List[FaceResult] = []
        for face_lm in results.multi_face_landmarks:
            # Convert normalised → pixel
            lm = np.array(
                [[lm.x * w, lm.y * h] for lm in face_lm.landmark],
                dtype=np.float32,
            )  # (468, 2)

            left_pts  = lm[MP_LEFT_EYE_EAR]   # (6, 2)
            right_pts = lm[MP_RIGHT_EYE_EAR]   # (6, 2)
            mouth_pts = lm[MP_MOUTH_EAR]        # (8, 2)

            # Bounding box from landmarks
            x1, y1 = lm[:, 0].min(), lm[:, 1].min()
            x2, y2 = lm[:, 0].max(), lm[:, 1].max()
            pad = 20
            x1, y1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
            x2, y2 = min(w, int(x2) + pad), min(h, int(y2) + pad)

            face_crop  = bgr[y1:y2, x1:x2]
            l_eye_crop = self._crop_eye(bgr, left_pts)
            r_eye_crop = self._crop_eye(bgr, right_pts)

            faces.append(FaceResult(
                face_bbox=(x1, y1, x2, y2),
                landmarks=lm,
                left_eye_pts=left_pts,
                right_eye_pts=right_pts,
                mouth_pts=mouth_pts,
                left_eye_crop=l_eye_crop,
                right_eye_crop=r_eye_crop,
                face_crop=face_crop,
                backend="mediapipe",
            ))
        return faces

    # ─── dlib backend ─────────────────────────────────────────────────────────

    def _detect_dlib(self, bgr: np.ndarray) -> List[FaceResult]:
        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self._dlib_detector(gray, 0)

        faces: List[FaceResult] = []
        for rect in rects:
            shape = self._dlib_predictor(gray, rect)
            lm = np.array([[shape.part(i).x, shape.part(i).y]
                           for i in range(68)], dtype=np.float32)

            left_pts  = lm[DLIB_LEFT_EYE]  # 6 pts
            right_pts = lm[DLIB_RIGHT_EYE]
            mouth_pts = lm[DLIB_MOUTH[:8]] # first 8 mouth points

            x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
            x1, y1 = max(0, x1), max(0, y1)
            face_crop  = bgr[y1:y2, x1:x2]
            l_eye_crop = self._crop_eye(bgr, left_pts)
            r_eye_crop = self._crop_eye(bgr, right_pts)

            faces.append(FaceResult(
                face_bbox=(x1, y1, x2, y2),
                landmarks=lm,
                left_eye_pts=left_pts,
                right_eye_pts=right_pts,
                mouth_pts=mouth_pts,
                left_eye_crop=l_eye_crop,
                right_eye_crop=r_eye_crop,
                face_crop=face_crop,
                backend="dlib",
            ))
        return faces

    # ─── Helper ───────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_eye(bgr: np.ndarray, pts: np.ndarray, pad: int = 10) -> np.ndarray:
        x1, y1 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
        x2, y2 = pts[:, 0].max() + pad, pts[:, 1].max() + pad
        h, w = bgr.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return cv2.resize(crop, (64, 64))

    @property
    def backend(self) -> str:
        return self._active_backend

    def close(self) -> None:
        if self._mp_detector is not None:
            self._mp_detector.close()
