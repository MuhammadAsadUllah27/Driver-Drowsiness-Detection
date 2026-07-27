"""
Real-Time Inference Pipeline — MediaPipe Face Mesh Edition
============================================================

Thin camera-loop wrapper around `DrowsinessDetector` (models.architectures),
which does all the real work: MediaPipe Face Mesh -> EAR/MAR/head-pose ->
PERCLOS/microsleep -> weighted fusion -> DrowsinessState.

This file replaces the old CNN-checkpoint-based Pipeline (FaceDetector +
EnsembleModel + DrowsinessStateMachine). Those modules are no longer used
here; DrowsinessDetector supersedes them.

Entry point: Pipeline.run()  — opens camera and loops until 'q' pressed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from config.config import infer_cfg, data_cfg
from inference.alert import AlertManager
from models.architectures import DrowsinessDetector, DrowsinessState, FusionWeights
from utils.logger import get_logger

log = get_logger(__name__, log_file=data_cfg.logs_dir / "inference.log")

_DEFAULT_SNAPSHOT_DIR = Path("snapshots")

_STATE_COLOR = {
    DrowsinessState.ALERT:      (0, 200, 60),    # green (BGR)
    DrowsinessState.DROWSY:     (0, 140, 255),   # orange
    DrowsinessState.MICROSLEEP: (0, 0, 255),     # red
}


class Pipeline:
    """
    Full real-time drowsiness detection pipeline (MediaPipe Face Mesh).

    Usage
    -----
    ::

        pipeline = Pipeline(
            fusion_weights=FusionWeights(w_perclos=0.40, w_ear=0.30),
            fps=30.0,
            perclos_window_seconds=60.0,
            microsleep_min_seconds=0.5,
            calibration_frames=60,
            mediapipe_detection_conf=0.5,
            mediapipe_tracking_conf=0.5,
            snapshot_dir=Path("snapshots"),
        )
        pipeline.run(camera_id=0)

    Parameters
    ----------
    fusion_weights : FusionWeights or None
        Relative importance of each drowsiness cue. Defaults applied if None.
    fps : float
        Requested/nominal camera FPS — used as a fallback and to seed the
        PERCLOS window-to-frame-count conversion before the actual camera
        FPS is known.
    perclos_window_seconds : float
        Rolling window duration for PERCLOS computation.
    microsleep_min_seconds : float
        Minimum continuous eye-closure duration to flag a microsleep.
    calibration_frames : int
        Number of frames used to calibrate the personal EAR baseline.
    mediapipe_detection_conf : float
        MediaPipe face detection confidence threshold.
    mediapipe_tracking_conf : float
        MediaPipe landmark tracking confidence threshold.
    snapshot_dir : Path or None
        Directory for manual frame snapshots (key: s). Defaults to
        ./snapshots if not provided.
    """

    def __init__(
        self,
        fusion_weights: Optional[FusionWeights] = None,
        fps: float = 30.0,
        perclos_window_seconds: float = 60.0,
        microsleep_min_seconds: float = 0.50,
        calibration_frames: int = 60,
        mediapipe_detection_conf: float = 0.5,
        mediapipe_tracking_conf: float = 0.5,
        snapshot_dir: Optional[Path] = None,
    ) -> None:
        self.fusion_weights = fusion_weights or FusionWeights()
        self.fps_hint = fps
        self.perclos_window_seconds = perclos_window_seconds
        self.microsleep_min_seconds = microsleep_min_seconds
        self.calibration_frames = calibration_frames
        self.mediapipe_detection_conf = mediapipe_detection_conf
        self.mediapipe_tracking_conf = mediapipe_tracking_conf

        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else _DEFAULT_SNAPSHOT_DIR
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Built lazily in run() once the actual camera resolution/FPS is
        # known — LandmarkGeometry's solvePnP camera matrix depends on it.
        self.detector: Optional[DrowsinessDetector] = None
        self.alert_mgr: Optional[AlertManager] = None
        self._paused = False

        log.info("Pipeline configured (MediaPipe Face Mesh, CPU-only).")

    # ─── Main camera loop ─────────────────────────────────────────────────────

    def run(self, camera_id: Optional[int] = None) -> None:
        """
        Open camera and run the detection loop.
        Press 'q' to quit, 'r' to reset calibration, 's' to snapshot,
        'p' to pause/resume.
        """
        cam_id = camera_id if camera_id is not None else infer_cfg.camera_id
        cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            log.error("Cannot open camera %d. Check 'camera_id' in config.", cam_id)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  infer_cfg.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, infer_cfg.camera_height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps_hint)

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or infer_cfg.camera_width
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or infer_cfg.camera_height
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps_hint

        # ── build detector + alert manager now that frame size/FPS are known ──
        self.detector = DrowsinessDetector(
            frame_width=actual_w,
            frame_height=actual_h,
            fps=actual_fps,
            perclos_window_seconds=self.perclos_window_seconds,
            microsleep_min_seconds=self.microsleep_min_seconds,
            calibration_frames=self.calibration_frames,
            fusion_weights=self.fusion_weights,
            mediapipe_detection_conf=self.mediapipe_detection_conf,
            mediapipe_tracking_conf=self.mediapipe_tracking_conf,
        )
        self.alert_mgr = AlertManager(fps=actual_fps)

        log.info(
            "Camera %d opened at %dx%d @ %.0f FPS — press 'q' to quit",
            cam_id, actual_w, actual_h, actual_fps,
        )
        print(
            "\n[Pipeline] Running. Press 'q' to quit, 'r' to reset, "
            "'s' to snapshot, 'p' to pause.\n"
        )

        fps_meter   = _FPSMeter()
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.error("Failed to grab frame. Camera disconnected?")
                    break

                frame_count += 1
                fps = fps_meter.tick()

                if not self._paused:
                    result = self.detector.process_frame(frame)
                    is_drowsy = result.state in (
                        DrowsinessState.DROWSY, DrowsinessState.MICROSLEEP
                    )
                    self.alert_mgr.update(drowsy=is_drowsy, frame=frame)
                    self._draw_hud(frame, result, fps)
                    self.alert_mgr.draw_alert_overlay(frame)
                else:
                    self._draw_paused(frame)

                if not self._show_and_handle_keys(frame):
                    break

        finally:
            cap.release()
            if self.alert_mgr is not None:
                self.alert_mgr.release()
            if self.detector is not None:
                self.detector.close()
            cv2.destroyAllWindows()
            log.info("Pipeline stopped after %d frames.", frame_count)

    # ─── Drawing helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _draw_hud(frame, result, fps: float) -> None:
        f = result.features
        color = _STATE_COLOR.get(result.state, (255, 255, 255))

        if not f.face_detected:
            h, w = frame.shape[:2]
            cv2.putText(
                frame, "No face detected", (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
            return

        lines = [
            f"STATE: {result.state.name}  ({result.drowsiness_prob:.2f})",
            f"EAR: {f.ear:.3f}   MAR: {f.mar:.3f}",
            f"PERCLOS: {f.perclos * 100:.1f}%   Blink: {f.blink_rate:.0f}/min",
            f"Yaw: {f.yaw:.1f} deg   Pitch: {f.pitch:.1f} deg",
            f"FPS: {fps:.1f}",
        ]
        y = 30
        for line in lines:
            cv2.putText(
                frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA,
            )
            y += 26

        if f.microsleep_flag:
            h, w = frame.shape[:2]
            cv2.putText(
                frame, "MICROSLEEP!", (w // 2 - 140, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA,
            )

    @staticmethod
    def _draw_paused(frame) -> None:
        h, w = frame.shape[:2]
        cv2.putText(
            frame, "PAUSED", (w // 2 - 90, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 3, cv2.LINE_AA,
        )

    def _show_and_handle_keys(self, frame) -> bool:
        cv2.imshow("Driver Drowsiness Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

        if key == ord("r"):
            if self.detector is not None:
                self.detector.reset()
            log.info("Detector reset — recalibrating EAR baseline.")

        if key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = self.snapshot_dir / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            log.info("Snapshot saved: %s", path)

        if key == ord("p"):
            self._paused = not self._paused
            log.info("Pipeline %s.", "paused" if self._paused else "resumed")

        return True


# ─── FPS Meter ────────────────────────────────────────────────────────────────

class _FPSMeter:
    def __init__(self, window: int = 30) -> None:
        from collections import deque
        self._times: "deque" = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / max(elapsed, 1e-6)