"""
Drowsiness State Machine

Combines CNN probability + EAR + MAR + head-pose into a single
robust drowsiness decision using temporal smoothing and multi-cue fusion.

States: AWAKE  →  DROWSY
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from config.config import infer_cfg
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FrameFeatures:
    """All per-frame cues passed to the state machine."""
    cnn_prob: float          # CNN drowsiness probability [0, 1]
    ear: float               # Eye Aspect Ratio
    mar: float               # Mouth Aspect Ratio (yawn)
    pitch: float             # Head pitch (degrees)
    yaw: float               # Head yaw (degrees)
    face_detected: bool = True


@dataclass
class DriveState:
    """Output state for one frame."""
    label: str               # "AWAKE" | "DROWSY"
    confidence: float        # Fused confidence [0, 1]
    cnn_prob: float
    ear: float
    mar: float
    pitch: float
    yaw: float
    ear_alert: bool          # EAR triggered independently
    mar_alert: bool          # MAR triggered independently
    pose_alert: bool         # Head pose triggered
    consecutive_drowsy: int  # Frames in current drowsy streak


class DrowsinessStateMachine:
    """
    Fuses multiple cues into a stable drowsiness decision.

    Algorithm
    ---------
    1. Collect CNN prob, EAR, MAR, head pose per frame.
    2. Compute a weighted fused_score in [0, 1].
    3. Smooth score over a rolling window.
    4. Trigger DROWSY if smoothed_score > threshold for N consecutive frames.
    5. Also trigger on raw EAR / MAR streaks (geometric backup).
    """

    def __init__(self) -> None:
        self._prob_window: Deque[float]   = deque(maxlen=infer_cfg.smoothing_window)
        self._ear_streak:  int            = 0
        self._mar_streak:  int            = 0
        self._drowsy_streak: int          = 0

        # Weights for fusion (must sum to 1)
        self._W_CNN   = 0.55
        self._W_EAR   = 0.25
        self._W_MAR   = 0.10
        self._W_POSE  = 0.10

        log.debug("DrowsinessStateMachine initialised.")

    def update(self, feat: FrameFeatures) -> DriveState:
        """
        Process one frame of features and return the current drive state.

        Parameters
        ----------
        feat : FrameFeatures

        Returns
        -------
        DriveState
        """
        if not feat.face_detected:
            self._prob_window.append(0.0)
            self._drowsy_streak = max(0, self._drowsy_streak - 1)
            return DriveState(
                label="AWAKE", confidence=0.0,
                cnn_prob=0.0, ear=0.0, mar=0.0, pitch=0.0, yaw=0.0,
                ear_alert=False, mar_alert=False, pose_alert=False,
                consecutive_drowsy=self._drowsy_streak,
            )

        # ── EAR score: normalise so 0 = open, 1 = closed ──────────────────────
        ear_score = max(0.0, 1.0 - (feat.ear / infer_cfg.ear_threshold))
        ear_score = min(ear_score, 1.0)

        # ── MAR score ─────────────────────────────────────────────────────────
        mar_score = min(feat.mar / infer_cfg.mar_threshold, 1.0)

        # ── Pose score ────────────────────────────────────────────────────────
        pitch_score = min(abs(feat.pitch) / infer_cfg.pitch_threshold, 1.0)
        yaw_score   = min(abs(feat.yaw)   / infer_cfg.yaw_threshold,   1.0)
        pose_score  = max(pitch_score, yaw_score)

        # ── Fused score ───────────────────────────────────────────────────────
        fused = (
            self._W_CNN  * feat.cnn_prob +
            self._W_EAR  * ear_score     +
            self._W_MAR  * mar_score     +
            self._W_POSE * pose_score
        )

        self._prob_window.append(fused)
        smoothed = sum(self._prob_window) / len(self._prob_window)

        # ── EAR streak (geometric) ────────────────────────────────────────────
        if feat.ear < infer_cfg.ear_threshold:
            self._ear_streak += 1
        else:
            self._ear_streak = 0
        ear_alert = self._ear_streak >= infer_cfg.ear_consec_frames

        # ── MAR streak ────────────────────────────────────────────────────────
        if feat.mar > infer_cfg.mar_threshold:
            self._mar_streak += 1
        else:
            self._mar_streak = 0
        mar_alert = self._mar_streak >= infer_cfg.mar_consec_frames

        # ── Pose alert ────────────────────────────────────────────────────────
        pose_alert = (
            abs(feat.pitch) > infer_cfg.pitch_threshold or
            abs(feat.yaw)   > infer_cfg.yaw_threshold
        )

        # ── Final decision ────────────────────────────────────────────────────
        is_drowsy = (
            smoothed >= infer_cfg.drowsy_threshold or
            ear_alert or
            mar_alert
        )

        if is_drowsy:
            self._drowsy_streak += 1
        else:
            self._drowsy_streak = max(0, self._drowsy_streak - 1)

        # Require sustained evidence before triggering
        label = (
            "DROWSY"
            if self._drowsy_streak >= infer_cfg.alert_trigger_frames
            else "AWAKE"
        )

        return DriveState(
            label=label,
            confidence=smoothed,
            cnn_prob=feat.cnn_prob,
            ear=feat.ear,
            mar=feat.mar,
            pitch=feat.pitch,
            yaw=feat.yaw,
            ear_alert=ear_alert,
            mar_alert=mar_alert,
            pose_alert=pose_alert,
            consecutive_drowsy=self._drowsy_streak,
        )

    def reset(self) -> None:
        self._prob_window.clear()
        self._ear_streak    = 0
        self._mar_streak    = 0
        self._drowsy_streak = 0
