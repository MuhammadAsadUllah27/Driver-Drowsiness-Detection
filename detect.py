"""
detect.py — Real-time driver drowsiness detection entry point.
==============================================================

Pipeline: MediaPipe Face Mesh → EAR · MAR · PERCLOS → FusionScorer
No model weights, no GPU, no checkpoint required.

Usage
-----
::

    python detect.py                          # webcam 0, all defaults
    python detect.py --camera 1              # secondary camera
    python detect.py --no-sound              # silent (visual alert only)
    python detect.py --no-clips              # do not save alert video clips
    python detect.py --perclos-window 30     # shorter PERCLOS window (seconds)
    python detect.py --microsleep 0.4        # tighten microsleep threshold

Fusion weight tuning (all weights are relative; normalised internally)::

    python detect.py --w-perclos 0.40 --w-ear 0.30   # emphasise PERCLOS + EAR

Controls (keyboard, while the HUD window is focused)
-----------------------------------------------------
    q      Quit
    r      Reset detector state (clears PERCLOS buffer + EMA baseline)
    s      Save a snapshot of the current frame to snapshots/
    p      Pause / resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── resolve project root so local imports work regardless of CWD ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import infer_cfg
from models.architectures import FusionWeights
from utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-Time Driver Drowsiness Detection  "
                    "(MediaPipe Face Mesh + EAR + MAR + PERCLOS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── camera ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--camera", type=int, default=infer_cfg.camera_id,
        metavar="ID",
        help="Camera device index (0 = built-in webcam, 1 = external).",
    )
    p.add_argument(
        "--width",  type=int, default=infer_cfg.camera_width,
        metavar="PX",
        help="Requested capture width in pixels.",
    )
    p.add_argument(
        "--height", type=int, default=infer_cfg.camera_height,
        metavar="PX",
        help="Requested capture height in pixels.",
    )
    p.add_argument(
        "--fps", type=float, default=getattr(infer_cfg, "camera_fps", 30.0),
        metavar="N",
        help="Camera frame rate — used for PERCLOS timing calculations.",
    )

    # ── detector parameters ───────────────────────────────────────────────────
    p.add_argument(
        "--perclos-window", type=float,
        default=getattr(infer_cfg, "perclos_window_seconds", 60.0),
        metavar="SEC",
        help="Rolling window duration for PERCLOS computation (seconds).",
    )
    p.add_argument(
        "--microsleep", type=float,
        default=getattr(infer_cfg, "microsleep_min_seconds", 0.50),
        metavar="SEC",
        help="Minimum continuous eye-closure to trigger a microsleep alarm.",
    )
    p.add_argument(
        "--calib-frames", type=int,
        default=getattr(infer_cfg, "calibration_frames", 60),
        metavar="N",
        help="Number of frames used to calibrate the personal EAR baseline.",
    )
    p.add_argument(
        "--detect-conf", type=float, default=0.5,
        metavar="[0-1]",
        help="MediaPipe face detection confidence threshold.",
    )
    p.add_argument(
        "--track-conf", type=float, default=0.5,
        metavar="[0-1]",
        help="MediaPipe landmark tracking confidence threshold.",
    )

    # ── fusion weights ────────────────────────────────────────────────────────
    fw = p.add_argument_group(
        "Fusion weights",
        "Relative importance of each cue — normalised internally. "
        "Increase a weight to make that signal dominate the drowsiness score.",
    )
    fw.add_argument("--w-perclos", type=float, default=0.35, metavar="W",
                    help="Weight for PERCLOS.")
    fw.add_argument("--w-ear",     type=float, default=0.25, metavar="W",
                    help="Weight for EAR deviation from personal baseline.")
    fw.add_argument("--w-mar",     type=float, default=0.15, metavar="W",
                    help="Weight for MAR (yawning).")
    fw.add_argument("--w-blink",   type=float, default=0.10, metavar="W",
                    help="Weight for blink-rate anomaly.")
    fw.add_argument("--w-pose",    type=float, default=0.10, metavar="W",
                    help="Weight for head-pose deviation (nod / look-away).")
    fw.add_argument("--w-micro",   type=float, default=0.05, metavar="W",
                    help="Weight for microsleep flag.")

    # ── runtime options ───────────────────────────────────────────────────────
    p.add_argument(
        "--no-sound", action="store_true",
        help="Disable audio alarm (visual HUD warning is always shown).",
    )
    p.add_argument(
        "--no-clips", action="store_true",
        help="Do not save short video clips around alert events.",
    )
    p.add_argument(
        "--snapshot-dir", type=Path,
        default=Path("snapshots"),
        metavar="DIR",
        help="Directory for manual frame snapshots (key: s).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Startup banner
# ──────────────────────────────────────────────────────────────────────────────

def _print_banner(args: argparse.Namespace) -> None:
    sep = "═" * 60
    print(f"\n{sep}")
    print("  Driver Drowsiness Detection  |  Real-Time  |  CPU-Only")
    print(f"{sep}")
    print(f"  Pipeline    : MediaPipe Face Mesh + EAR + MAR + PERCLOS")
    print(f"  Camera ID   : {args.camera}  ({args.width}×{args.height} @ {args.fps:.0f} fps)")
    print(f"  PERCLOS win : {args.perclos_window} s")
    print(f"  Microsleep  : ≥ {args.microsleep} s continuous closure")
    print(f"  Calib frames: {args.calib_frames}")
    print(f"  Sound alert : {'ON'  if not args.no_sound else 'OFF'}")
    print(f"  Save clips  : {'ON'  if not args.no_clips else 'OFF'}")
    print(f"  Fusion wts  : PERCLOS={args.w_perclos}  EAR={args.w_ear}  "
          f"MAR={args.w_mar}  Blink={args.w_blink}  "
          f"Pose={args.w_pose}  Micro={args.w_micro}")
    print(f"{sep}")
    print("  Keyboard: q = Quit | r = Reset | s = Snapshot | p = Pause")
    print(f"{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # ── apply CLI overrides to infer_cfg ──────────────────────────────────────
    # Only mutate fields the config object actually owns; new fields are passed
    # directly to DrowsinessDetector below without touching the config.
    infer_cfg.camera_id        = args.camera
    infer_cfg.camera_width     = args.width
    infer_cfg.camera_height    = args.height
    infer_cfg.alert_sound      = not args.no_sound
    infer_cfg.save_alert_clips = not args.no_clips

    # ── build FusionWeights from CLI args ─────────────────────────────────────
    fusion_weights = FusionWeights(
        w_perclos = args.w_perclos,
        w_ear     = args.w_ear,
        w_mar     = args.w_mar,
        w_blink   = args.w_blink,
        w_pose    = args.w_pose,
        w_micro   = args.w_micro,
    )

    _print_banner(args)

    # ── launch pipeline ───────────────────────────────────────────────────────
    # Pipeline is imported here (not at module level) so that the banner and
    # any config mutations are applied before MediaPipe initialises.
    from inference.pipeline import Pipeline

    pipeline = Pipeline(
        fusion_weights           = fusion_weights,
        fps                      = args.fps,
        perclos_window_seconds   = args.perclos_window,
        microsleep_min_seconds   = args.microsleep,
        calibration_frames       = args.calib_frames,
        mediapipe_detection_conf = args.detect_conf,
        mediapipe_tracking_conf  = args.track_conf,
        snapshot_dir             = args.snapshot_dir,
    )
    pipeline.run(camera_id=args.camera)


if __name__ == "__main__":
    main()