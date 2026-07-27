"""
Alert System Module

Handles:
    • Auditory alert  — pygame beep or system bell fallback
    • Visual overlay  — red flashing banner on frame
    • Cooldown logic  — prevents alert spam
"""

from __future__ import annotations

import time
import threading
from pathlib import Path

import cv2
import numpy as np

from config.config import infer_cfg
from utils.logger import get_logger

log = get_logger(__name__)

# ── Optional pygame for sound ──────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False
    log.warning("pygame not available — falling back to terminal bell for alerts.")


def _beep_thread(freq: int = 880, duration_ms: int = 600) -> None:
    """Play a beep in a background thread so it doesn't block the frame loop."""
    if _HAS_PYGAME:
        try:
            sample_rate = 44100
            n_samples   = int(sample_rate * duration_ms / 1000)
            t           = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
            wave        = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            sound       = pygame.sndarray.make_sound(wave)
            sound.play()
            pygame.time.wait(duration_ms)
        except Exception as exc:
            log.debug("pygame beep error: %s", exc)
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)


class AlertManager:
    """
    Manages drowsiness alerts with cooldown and clip saving.

    Parameters
    ----------
    fps : expected camera FPS (used for clip length calculation)
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps             = fps
        self._last_alert_t   = 0.0
        self._alert_active   = False
        self._flash_state    = False
        self._flash_counter  = 0

        # No on-disk clip saving: alerts are visual/audio only.
        self._recording = False
        self._record_frames_left = 0
        log.info("AlertManager ready | clip saving disabled")

    # ─── Main update call ─────────────────────────────────────────────────────

    def update(self, drowsy: bool, frame: np.ndarray) -> bool:
        """
        Call every frame.

        Parameters
        ----------
        drowsy : True if drowsiness detected this frame
        frame  : current BGR frame (used for clip saving)

        Returns
        -------
        bool  — True if an alert fired this frame
        """
        fired = False
        if drowsy:
            now     = time.time()
            elapsed = now - self._last_alert_t
            if elapsed >= infer_cfg.alert_cooldown_sec:
                self._fire_alert(frame)
                self._last_alert_t = now
                fired = True
            self._alert_active = True
        else:
            self._alert_active = False

        # (No on-disk clip recording in this build)

        return fired

    # ─── Overlay ──────────────────────────────────────────────────────────────

    def draw_alert_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Adds a flashing red danger banner when alert is active.
        Call once per frame after update().
        """
        if not self._alert_active:
            return frame

        self._flash_counter += 1
        if self._flash_counter % 10 == 0:
            self._flash_state = not self._flash_state

        if self._flash_state:
            overlay = frame.copy()
            h, w    = frame.shape[:2]
            cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            cv2.putText(
                frame,
                "⚠  DROWSINESS DETECTED — PLEASE PULL OVER  ⚠",
                (w // 2 - 380, 52),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _fire_alert(self, frame: np.ndarray) -> None:
        """Play sound (no on-disk saving)."""
        log.warning("DROWSINESS ALERT FIRED")

        if infer_cfg.alert_sound:
            t = threading.Thread(target=_beep_thread, args=(880, 800), daemon=True)
            t.start()
    def release(self) -> None:
        return
