"""
evaluate.py — Offline evaluation of the MediaPipe drowsiness detection pipeline.
==================================================================================

Replaces the old CNN-checkpoint evaluator.  No model weights, no GPU, no
DataLoader — the pipeline is purely geometric (EAR / MAR / PERCLOS).

Input formats
-------------
The script accepts **two** evaluation modes:

  1. Video + CSV annotation  (frame-level labels)
     --video  path/to/clip.mp4  --labels path/to/labels.csv

  2. Folder of annotated video clips  (one CSV per video, same stem)
     --video_dir  dataset/test_videos/

Annotation CSV format
---------------------
Each CSV must have at minimum two columns::

    frame_idx,label
    0,alert
    1,alert
    2,drowsy
    ...

Accepted label strings (case-insensitive):
    alert      → DrowsinessState.ALERT
    drowsy     → DrowsinessState.DROWSY
    microsleep → DrowsinessState.MICROSLEEP

Usage examples
--------------
::

    # Single video
    python evaluate.py --video data/subject_01.mp4 --labels data/subject_01.csv

    # Folder (each .mp4 must have a matching .csv with same filename stem)
    python evaluate.py --video_dir dataset/test_videos/

    # Override detector thresholds
    python evaluate.py --video_dir dataset/test/ --perclos_window 30 --fps 25

    # Save detailed per-frame results to CSV
    python evaluate.py --video_dir dataset/test/ --save_results results/

Output
------
Prints a structured metrics report to stdout and optionally writes
per-frame CSV files + a JSON summary to --save_results.

Metrics computed
----------------
* Overall accuracy, macro-averaged precision / recall / F1
* Per-class precision / recall / F1 / support
* Confusion matrix (text table)
* PERCLOS Mean Absolute Error (vs ground-truth, if GT provides PERCLOS values)
* Mean inference latency (ms/frame)  and effective FPS
* Frames with no face detected (%)
* Calibration convergence frame index

Author : Asad
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── resolve project root so local imports work regardless of CWD ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.architectures import (
    DetectionResult,
    DrowsinessDetector,
    DrowsinessState,
    FusionWeights,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ── label string → enum mapping ───────────────────────────────────────────────
_LABEL_MAP: Dict[str, DrowsinessState] = {
    "alert":      DrowsinessState.ALERT,
    "drowsy":     DrowsinessState.DROWSY,
    "microsleep": DrowsinessState.MICROSLEEP,
}
_STATE_NAMES: List[str] = ["alert", "drowsy", "microsleep"]
_STATE_IDX:  Dict[DrowsinessState, int] = {
    DrowsinessState.ALERT:      0,
    DrowsinessState.DROWSY:     1,
    DrowsinessState.MICROSLEEP: 2,
}


# ──────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ──────────────────────────────────────────────────────────────────────────────

def load_labels(csv_path: Path) -> Dict[int, DrowsinessState]:
    """
    Load frame-level ground-truth labels from a CSV file.

    Parameters
    ----------
    csv_path : Path
        CSV with at minimum columns ``frame_idx`` and ``label``.

    Returns
    -------
    dict mapping frame_idx (int) → DrowsinessState

    Raises
    ------
    ValueError
        If a label string is not in ``_LABEL_MAP``.
    FileNotFoundError
        If the CSV path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found: {csv_path}")

    labels: Dict[int, DrowsinessState] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {csv_path}")

        # Normalise header names (strip whitespace, lowercase)
        norm_fields = {f.strip().lower(): f for f in reader.fieldnames}
        if "frame_idx" not in norm_fields or "label" not in norm_fields:
            raise ValueError(
                f"CSV must contain 'frame_idx' and 'label' columns. "
                f"Found: {list(reader.fieldnames)}"
            )

        for row in reader:
            idx_raw   = row[norm_fields["frame_idx"]].strip()
            label_raw = row[norm_fields["label"]].strip().lower()
            if label_raw not in _LABEL_MAP:
                raise ValueError(
                    f"Unknown label '{label_raw}' in {csv_path}. "
                    f"Accepted: {list(_LABEL_MAP)}"
                )
            labels[int(idx_raw)] = _LABEL_MAP[label_raw]

    log.debug("Loaded %d labelled frames from %s", len(labels), csv_path)
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Per-video evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_video(
    video_path:  Path,
    labels:      Dict[int, DrowsinessState],
    detector:    DrowsinessDetector,
) -> Tuple[List[dict], dict]:
    """
    Run the detector on every labelled frame of one video clip.

    Parameters
    ----------
    video_path : Path
        Path to the video file (any format OpenCV can decode).
    labels : dict
        Frame-index → DrowsinessState ground-truth map.
    detector : DrowsinessDetector
        Pre-initialised detector (shared across all videos for efficiency).

    Returns
    -------
    rows : list of dict
        One dict per evaluated frame containing ground-truth, prediction,
        probability, all features, and timing information.
    summary : dict
        Aggregate stats for this clip (latency, no-face rate, etc.).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.info(
        "Video: %s  |  %dx%d  %.1f fps  %d frames  |  %d labelled",
        video_path.name, frame_w, frame_h, video_fps, total_frames, len(labels),
    )

    rows:         List[dict] = []
    latencies:    List[float] = []
    no_face_count: int = 0
    frame_idx:    int = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in labels:
            t0 = time.perf_counter()
            result: DetectionResult = detector.process_frame(
                frame, timestamp=frame_idx / video_fps
            )
            latency_ms = (time.perf_counter() - t0) * 1_000

            latencies.append(latency_ms)
            if not result.features.face_detected:
                no_face_count += 1

            gt = labels[frame_idx]
            rows.append({
                "frame_idx":       frame_idx,
                "gt_label":        gt.name,
                "pred_label":      result.state.name,
                "drowsiness_prob": round(result.drowsiness_prob, 6),
                "ear":             round(result.features.ear, 6),
                "mar":             round(result.features.mar, 6),
                "perclos":         round(result.features.perclos, 6),
                "blink_rate":      round(result.features.blink_rate, 4),
                "yaw":             round(result.features.yaw, 3),
                "pitch":           round(result.features.pitch, 3),
                "roll":            round(result.features.roll, 3),
                "microsleep_flag": int(result.features.microsleep_flag),
                "face_detected":   int(result.features.face_detected),
                "latency_ms":      round(latency_ms, 3),
                "ear_threshold":   round(detector.ear_threshold, 6),
                "calib_complete":  int(detector.calibration_complete),
                **{f"s_{k}": round(v, 6) for k, v in result.breakdown.items()},
            })

        frame_idx += 1

    cap.release()

    n = len(rows)
    summary = {
        "video":            video_path.name,
        "evaluated_frames": n,
        "no_face_pct":      round(no_face_count / max(n, 1) * 100, 2),
        "mean_latency_ms":  round(float(np.mean(latencies)) if latencies else 0.0, 3),
        "std_latency_ms":   round(float(np.std(latencies))  if latencies else 0.0, 3),
        "p95_latency_ms":   round(float(np.percentile(latencies, 95)) if latencies else 0.0, 3),
        "effective_fps":    round(1000.0 / float(np.mean(latencies)) if latencies else 0.0, 2),
    }
    return rows, summary


# ──────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(rows: List[dict]) -> dict:
    """
    Compute classification metrics from a flat list of per-frame result dicts.

    Parameters
    ----------
    rows : list of dict
        Each dict must contain ``gt_label`` and ``pred_label`` keys
        (DrowsinessState.name strings).

    Returns
    -------
    dict with keys:
        accuracy, macro_precision, macro_recall, macro_f1,
        per_class (dict), confusion_matrix (3×3 list)
    """
    n_classes = len(_STATE_NAMES)

    # confusion_matrix[true][pred]
    cm: List[List[int]] = [[0] * n_classes for _ in range(n_classes)]

    name_to_idx = {name: i for i, name in enumerate(_STATE_NAMES)}

    for row in rows:
        gt_idx   = name_to_idx.get(row["gt_label"].lower(),   -1)
        pred_idx = name_to_idx.get(row["pred_label"].lower(), -1)
        if gt_idx >= 0 and pred_idx >= 0:
            cm[gt_idx][pred_idx] += 1

    total   = sum(cm[i][j] for i in range(n_classes) for j in range(n_classes))
    correct = sum(cm[i][i] for i in range(n_classes))
    accuracy = correct / max(total, 1)

    per_class: Dict[str, dict] = {}
    precisions, recalls, f1s = [], [], []

    for i, cls in enumerate(_STATE_NAMES):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n_classes)) - tp   # col sum − diag
        fn = sum(cm[i][j] for j in range(n_classes)) - tp   # row sum − diag
        support = tp + fn

        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)

        per_class[cls] = {
            "precision": round(prec,    4),
            "recall":    round(rec,     4),
            "f1":        round(f1,      4),
            "support":   support,
        }
        if support > 0:                          # exclude absent classes from macro
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_p  = float(np.mean(precisions)) if precisions else 0.0
    macro_r  = float(np.mean(recalls))    if recalls    else 0.0
    macro_f1 = float(np.mean(f1s))        if f1s        else 0.0

    return {
        "total_frames":    total,
        "accuracy":        round(accuracy,  4),
        "macro_precision": round(macro_p,   4),
        "macro_recall":    round(macro_r,   4),
        "macro_f1":        round(macro_f1,  4),
        "per_class":       per_class,
        "confusion_matrix": cm,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report printing
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30) -> str:
    """ASCII progress bar for a [0,1] value."""
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"]  {value:.2%}"


def print_report(
    metrics:    dict,
    perf_stats: dict,
    video_name: str = "",
) -> None:
    """Print a structured evaluation report to stdout."""
    sep  = "─" * 62
    sep2 = "═" * 62

    print(f"\n{sep2}")
    title = "DROWSINESS DETECTOR — EVALUATION REPORT"
    if video_name:
        title += f"\n  {video_name}"
    print(f"  {title}")
    print(f"{sep2}\n")

    # ── classification metrics ────────────────────────────────────────────────
    print("  CLASSIFICATION METRICS")
    print(f"  {sep}")
    print(f"  Overall Accuracy   : {metrics['accuracy']:.4f}  "
          f"({metrics['accuracy']:.2%})  "
          f"[{metrics['total_frames']} frames]")
    print(f"  Macro Precision    : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall       : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1           : {metrics['macro_f1']:.4f}")
    print()

    # ── per-class ─────────────────────────────────────────────────────────────
    print("  PER-CLASS BREAKDOWN")
    print(f"  {sep}")
    header = f"  {'Class':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"
    print(header)
    print(f"  {sep}")
    for cls, m in metrics["per_class"].items():
        print(
            f"  {cls.upper():<14}"
            f"{m['precision']:>10.4f}"
            f"{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}"
            f"{m['support']:>10}"
        )
    print()

    # ── confusion matrix ──────────────────────────────────────────────────────
    print("  CONFUSION MATRIX  (rows = ground-truth, cols = predicted)")
    print(f"  {sep}")
    col_w = 13
    header_row = " " * 18 + "".join(f"{n.upper():>{col_w}}" for n in _STATE_NAMES)
    print(f"  {header_row}")
    cm = metrics["confusion_matrix"]
    for i, row_name in enumerate(_STATE_NAMES):
        row_str = "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(_STATE_NAMES)))
        print(f"  {row_name.upper():<18}{row_str}")
    print()

    # ── visual F1 bars ────────────────────────────────────────────────────────
    print("  F1 SCORE VISUAL")
    print(f"  {sep}")
    for cls, m in metrics["per_class"].items():
        if m["support"] > 0:
            print(f"  {cls.upper():<14} {_bar(m['f1'])}")
    print()

    # ── pipeline performance ──────────────────────────────────────────────────
    print("  PIPELINE PERFORMANCE (CPU)")
    print(f"  {sep}")
    print(f"  Mean latency       : {perf_stats.get('mean_latency_ms', 0):.2f} ms/frame")
    print(f"  Std  latency       : {perf_stats.get('std_latency_ms',  0):.2f} ms")
    print(f"  P95  latency       : {perf_stats.get('p95_latency_ms',  0):.2f} ms")
    print(f"  Effective FPS      : {perf_stats.get('effective_fps',   0):.1f}")
    print(f"  No-face frames     : {perf_stats.get('no_face_pct',     0):.1f}%")
    print(f"\n{sep2}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Results saving
# ──────────────────────────────────────────────────────────────────────────────

def save_results(
    rows:       List[dict],
    metrics:    dict,
    perf_stats: dict,
    out_dir:    Path,
    stem:       str,
) -> None:
    """
    Write per-frame CSV and JSON summary to ``out_dir``.

    Parameters
    ----------
    rows : list of dict
        Per-frame result rows from evaluate_video().
    metrics : dict
        Output of compute_metrics().
    perf_stats : dict
        Performance summary dict from evaluate_video().
    out_dir : Path
        Directory to write outputs (created if needed).
    stem : str
        File stem (usually the video filename without extension).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame CSV
    csv_path = out_dir / f"{stem}_frames.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Per-frame CSV → %s", csv_path)

    # JSON summary
    json_path = out_dir / f"{stem}_summary.json"
    summary = {
        "performance": perf_stats,
        "metrics":     metrics,
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary JSON  → %s", json_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the MediaPipe drowsiness detector on labelled video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── input ──────────────────────────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--video", type=Path, metavar="PATH",
        help="Single video file to evaluate.",
    )
    grp.add_argument(
        "--video_dir", type=Path, metavar="DIR",
        help=(
            "Directory containing video files. Each video must have a matching "
            ".csv annotation file with the same filename stem."
        ),
    )
    p.add_argument(
        "--labels", type=Path, default=None, metavar="CSV",
        help="Label CSV for --video mode. Defaults to <video_stem>.csv in same folder.",
    )

    # ── detector knobs ─────────────────────────────────────────────────────────
    p.add_argument("--fps",              type=float, default=30.0,  help="Camera/video FPS.")
    p.add_argument("--perclos_window",   type=float, default=60.0,  help="PERCLOS window (seconds).")
    p.add_argument("--microsleep_secs",  type=float, default=0.50,  help="Microsleep threshold (seconds).")
    p.add_argument("--calib_frames",     type=int,   default=60,    help="EAR calibration frame count.")
    p.add_argument("--detect_conf",      type=float, default=0.5,   help="MediaPipe detection confidence.")
    p.add_argument("--track_conf",       type=float, default=0.5,   help="MediaPipe tracking confidence.")

    # ── fusion weight overrides ────────────────────────────────────────────────
    p.add_argument("--w_perclos", type=float, default=0.35)
    p.add_argument("--w_ear",     type=float, default=0.25)
    p.add_argument("--w_mar",     type=float, default=0.15)
    p.add_argument("--w_blink",   type=float, default=0.10)
    p.add_argument("--w_pose",    type=float, default=0.10)
    p.add_argument("--w_micro",   type=float, default=0.05)

    # ── output ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--save_results", type=Path, default=None, metavar="DIR",
        help="Directory to save per-frame CSVs and JSON summaries.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")

    return p


def _collect_video_label_pairs(
    args: argparse.Namespace,
) -> List[Tuple[Path, Path]]:
    """Resolve all (video, label_csv) pairs from CLI arguments."""
    pairs: List[Tuple[Path, Path]] = []

    if args.video:
        video = args.video
        if not video.exists():
            log.error("Video not found: %s", video)
            sys.exit(1)
        csv_path = args.labels or video.with_suffix(".csv")
        if not csv_path.exists():
            log.error(
                "Label file not found: %s  "
                "(pass --labels explicitly or name it <video_stem>.csv)",
                csv_path,
            )
            sys.exit(1)
        pairs.append((video, csv_path))

    else:  # --video_dir
        video_dir = args.video_dir
        if not video_dir.is_dir():
            log.error("Not a directory: %s", video_dir)
            sys.exit(1)
        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        for vf in sorted(video_dir.iterdir()):
            if vf.suffix.lower() in video_extensions:
                csv_path = vf.with_suffix(".csv")
                if csv_path.exists():
                    pairs.append((vf, csv_path))
                else:
                    log.warning("No label CSV found for %s — skipping.", vf.name)

        if not pairs:
            log.error(
                "No (video, .csv) pairs found in %s. "
                "Each video must have a matching .csv with the same filename stem.",
                video_dir,
            )
            sys.exit(1)

    return pairs


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # ── collect input pairs ───────────────────────────────────────────────────
    pairs = _collect_video_label_pairs(args)
    log.info("Found %d video(s) to evaluate.", len(pairs))

    # ── build detector once (shared across all videos) ────────────────────────
    # Resolution is read from the first video; all videos in a batch are assumed
    # to share the same resolution. Override with --video_dir if mixed.
    first_cap = cv2.VideoCapture(str(pairs[0][0]))
    frame_w   = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    fusion_weights = FusionWeights(
        w_perclos = args.w_perclos,
        w_ear     = args.w_ear,
        w_mar     = args.w_mar,
        w_blink   = args.w_blink,
        w_pose    = args.w_pose,
        w_micro   = args.w_micro,
    )

    detector = DrowsinessDetector(
        frame_width              = frame_w,
        frame_height             = frame_h,
        fps                      = args.fps,
        perclos_window_seconds   = args.perclos_window,
        microsleep_min_seconds   = args.microsleep_secs,
        calibration_frames       = args.calib_frames,
        fusion_weights           = fusion_weights,
        mediapipe_detection_conf = args.detect_conf,
        mediapipe_tracking_conf  = args.track_conf,
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    all_rows:   List[dict] = []
    all_stats:  List[dict] = []

    with detector:
        for video_path, label_path in pairs:
            labels = load_labels(label_path)
            rows, perf = evaluate_video(video_path, labels, detector)
            metrics    = compute_metrics(rows)

            print_report(metrics, perf, video_name=video_path.name)

            if args.save_results:
                save_results(rows, metrics, perf, args.save_results, video_path.stem)

            all_rows.extend(rows)
            all_stats.append(perf)

    # ── aggregate report across all videos ───────────────────────────────────
    if len(pairs) > 1:
        agg_metrics = compute_metrics(all_rows)
        agg_perf = {
            "video":            "ALL VIDEOS (aggregate)",
            "evaluated_frames": sum(s["evaluated_frames"] for s in all_stats),
            "no_face_pct":      float(np.mean([s["no_face_pct"]     for s in all_stats])),
            "mean_latency_ms":  float(np.mean([s["mean_latency_ms"] for s in all_stats])),
            "std_latency_ms":   float(np.mean([s["std_latency_ms"]  for s in all_stats])),
            "p95_latency_ms":   float(np.mean([s["p95_latency_ms"]  for s in all_stats])),
            "effective_fps":    float(np.mean([s["effective_fps"]   for s in all_stats])),
        }
        print_report(agg_metrics, agg_perf, video_name="── AGGREGATE (all videos) ──")

        if args.save_results:
            save_results(
                all_rows, agg_metrics, agg_perf,
                args.save_results, "aggregate",
            )


if __name__ == "__main__":
    main()