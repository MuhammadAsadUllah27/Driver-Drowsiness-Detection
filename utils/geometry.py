"""
Geometric feature extraction from facial landmarks.

Implements:
  • EAR  — Eye Aspect Ratio  (Soukupová & Čech, 2016)
  • MAR  — Mouth Aspect Ratio (yawn detection)
  • Head Pose estimation via solvePnP
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─── MediaPipe Landmark Indices ────────────────────────────────────────────────
# 468-point FaceMesh indices for key regions

MP_LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MP_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 6-point EAR landmarks per eye (top, bottom, left corner, right corner)
MP_LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
MP_RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

MP_MOUTH_EAR = [61, 291, 0, 17, 39, 181, 269, 405]

# dlib 68-point indices
DLIB_LEFT_EYE = list(range(42, 48))
DLIB_RIGHT_EYE = list(range(36, 42))
DLIB_MOUTH = list(range(48, 68))

# 3-D model points for head pose (generic face model, in mm)
MODEL_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip
    [0.0, -330.0, -65.0],     # Chin
    [-225.0, 170.0, -135.0],  # Left eye left corner
    [225.0, 170.0, -135.0],   # Right eye right corner
    [-150.0, -150.0, -125.0], # Left mouth corner
    [150.0, -150.0, -125.0],  # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
MODEL_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]


# ─── EAR ──────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(landmarks_6pt: np.ndarray) -> float:
    """
    Compute Eye Aspect Ratio from 6 landmarks.

    Points order: [p1 (left corner), p2 (top-left), p3 (top-right),
                   p4 (right corner), p5 (bottom-right), p6 (bottom-left)]

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    landmarks_6pt : np.ndarray, shape (6, 2)  (x, y) pixel coords

    Returns
    -------
    float  EAR value (lower → more closed)
    """
    p1, p2, p3, p4, p5, p6 = landmarks_6pt
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)
    return float(ear)


def mean_ear(left_pts: np.ndarray, right_pts: np.ndarray) -> float:
    """Average EAR across both eyes."""
    return (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0


# ─── MAR ──────────────────────────────────────────────────────────────────────

def mouth_aspect_ratio(landmarks_8pt: np.ndarray) -> float:
    """
    Compute Mouth Aspect Ratio from 8 landmarks for yawn detection.

    MAR = (||A-G|| + ||B-F|| + ||C-E||) / (2 * ||D-H||)

    Parameters
    ----------
    landmarks_8pt : np.ndarray, shape (8, 2)

    Returns
    -------
    float  MAR value (higher → more open)
    """
    A, B, C, D, E, F, G, H = landmarks_8pt
    vert1 = np.linalg.norm(A - G)
    vert2 = np.linalg.norm(B - F)
    vert3 = np.linalg.norm(C - E)
    horiz = np.linalg.norm(D - H)
    mar = (vert1 + vert2 + vert3) / (2.0 * horiz + 1e-6)
    return float(mar)


# ─── Head Pose ────────────────────────────────────────────────────────────────

def estimate_head_pose(
    landmarks_2d: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[float, float, float]:
    """
    Estimate head pose (pitch, yaw, roll) using solvePnP.

    Parameters
    ----------
    landmarks_2d : np.ndarray  Shape (N, 2) — all face landmarks in pixel coords.
    image_shape  : (height, width)

    Returns
    -------
    pitch, yaw, roll  (degrees)
      pitch > 0 → nodding down
      yaw   > 0 → turning right
      roll  > 0 → tilting right
    """
    h, w = image_shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    image_points = landmarks_2d[MODEL_LANDMARK_IDS].astype(np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_3D_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = math.degrees(angles[0])
    yaw   = math.degrees(angles[1])
    roll  = math.degrees(angles[2])
    return pitch, yaw, roll


# ─── Drawing Helpers ──────────────────────────────────────────────────────────

def draw_eye_contour(
    image: np.ndarray,
    pts: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> None:
    pts_int = pts.astype(int)
    cv2.polylines(image, [pts_int], isClosed=True, color=color, thickness=thickness)


def draw_metrics_overlay(
    image: np.ndarray,
    ear: float,
    mar: float,
    pitch: float,
    yaw: float,
    cnn_prob: float,
    state: str,
    fps: float,
) -> np.ndarray:
    """
    Renders a semi-transparent HUD with all metrics on the frame.
    """
    overlay = image.copy()
    panel_h, panel_w = 200, 270
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    state_color = (0, 60, 255) if state == "DROWSY" else (0, 200, 60)

    lines = [
        (f"State : {state}", state_color, 0.65, 2),
        (f"CNN   : {cnn_prob:.2f}", (255, 255, 255), 0.52, 1),
        (f"EAR   : {ear:.3f}", (255, 255, 255), 0.52, 1),
        (f"MAR   : {mar:.3f}", (255, 255, 255), 0.52, 1),
        (f"Pitch : {pitch:+.1f}°", (255, 255, 255), 0.52, 1),
        (f"Yaw   : {yaw:+.1f}°", (255, 255, 255), 0.52, 1),
        (f"FPS   : {fps:.1f}", (200, 200, 200), 0.48, 1),
    ]
    y = 38
    for text, color, scale, thickness in lines:
        cv2.putText(image, text, (18, y), font, scale, color, thickness, cv2.LINE_AA)
        y += 27

    return image
