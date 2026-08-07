# 🚗 Driver Drowsiness Detection

> **Real-time, CPU-only driver fatigue monitoring using MediaPipe Face Mesh geometry — no deep-learning weights, no GPU required.**

---

## Overview

This system detects driver drowsiness in real time by analysing facial geometry extracted from a standard webcam. Rather than running CNN inference, the pipeline computes four physiologically grounded signals per frame and fuses them into a single drowsiness score:

| Signal | Description |
|---|---|
| **EAR** | Eye Aspect Ratio — geometric measure of eye openness |
| **MAR** | Mouth Aspect Ratio — detects yawning |
| **PERCLOS** | Percentage of Eye Closure over a rolling time window |
| **Head Pose** | Yaw / pitch / roll via `solvePnP` — detects nodding and look-away |

An additional **microsleep detector** fires when eyes remain continuously closed for a configurable minimum duration (default: 0.5 s). All six signals are combined by a `FusionScorer` with tunable relative weights to produce a `DrowsinessState`: **ALERT**, **DROWSY**, or **MICROSLEEP**.

**No PyTorch. No checkpoints. No GPU.** The entire inference path runs on CPU with a dependency footprint of just four packages.

---

## Architecture

```
Webcam frame
    │
    ▼
MediaPipe Face Mesh  (478 landmarks, TFLite runtime, bundled)
    │
    ├─▶ EAR calculator        → eye-openness ratio
    ├─▶ MAR calculator        → mouth-openness ratio
    ├─▶ PERCLOS tracker       → rolling eye-closure %
    ├─▶ Blink rate monitor    → blink anomaly score
    ├─▶ Head-pose estimator   → yaw · pitch · roll (solvePnP)
    └─▶ Microsleep detector   → continuous-closure flag
             │
             ▼
        FusionScorer          (weighted sum → [0, 1])
             │
             ▼
        DrowsinessState       ALERT / DROWSY / MICROSLEEP
             │
             ▼
        HUD overlay + audio alarm
```

A **personal EAR baseline** is computed during a short calibration window (default: 60 frames) to adapt thresholds to the individual driver, improving robustness across ethnicities, glasses, and lighting conditions. An optional Butterworth low-pass filter (via SciPy) smooths EAR/MAR time series before feeding the PERCLOS tracker.

---

## Repository Structure

```
Driver-Drowsiness-Detection/
├── config/                 # Centralised configuration dataclasses
│   ├── __init__.py
│   ├── config.py
│   └── __pycache__/
│   │   ├── __init__.cpython-310
│   │   └── config.cpython-310
├── models/                 # DrowsinessDetector, FusionWeights, architectures
│   ├── __init__.py
│   ├── architectures.py
│   └── __pycache__/
│   │   ├── __init__.cpython-310
│   │   └── architectures.cpython-310
├── inference/              # Pipeline class — frame loop, HUD, alert logic
│   ├── __init__.py
│   ├── alert.py
│   ├── face_detector.py
│   ├── pipeline.py
│   ├── preprocessor.py
│   ├── state_machine.py
│   └── __pycache__/
│   │   ├── __init__.cpython-310
│   │   ├── alert.cpython-310
│   │   ├── face_detector.cpython-310
│   │   ├── pipeline.cpython-310
│   │   ├── preprocessor.cpython-310
│   │   └── state_machine.cpython-310
├── training/               # Dataset loaders, loss functions, trainer
│   ├── __init__.py
│   ├── trainer.py
│   └── __pycache__/
│   │   ├── __init__.cpython-310
│   │   └── trainer.cpython-310
├── utils/                  # Logger, geometry helpers, signal processing
│   ├── __init__.py
│   ├── geometry.py
│   ├── logger.py
│   └── __pycache__/
│   │   ├── __init__.cpython-310
│   │   ├── geometry.cpython-310
│   │   └── logger.cpython-310
├── dashboard/              # (Optional) monitoring dashboard
│   ├── __init__.py
├── data/                   # Dataset storage
│   ├── __init__.py
│   ├── dataset.py
│   └── __pycache__/
│   │   ├── __init__.cpython-310
│   │   └── dataset.cpython-310
├── checkpoints/            # Saved model/calibration checkpoints
│   └── best_face_model.pth
├── logs/                   # Runtime and evaluation logs
│   ├── train.py
│   └── inference.py
├── tests/
│   ├── __init__.py
│   ├── test_dataset_layout.py
│   ├── test_state_machine.py
│   ├── test_training_defaults.py
│   └── __pycache__/
│   │   ├── __init__.cpython-310
│   │   ├── test_dataset_layout.cpython-310-pytest-8.1.1
│   │   ├── test_dataset_layout.cpython-310-pytest-9.1.1
│   │   ├── test_state_machine.cpython-310-pytest-8.1.1
│   │   ├── test_state_machine.cpython-310-pytest-9.1.1
│   │   ├── test_training_defaults.cpython-310-pytest-8.1.1
│   │   └── test_training_defaults.cpython-310-pytest-9.1.1
├── __pycache__/
│   ├── __init__.cpython-310
│   └── train.cpython-310
├── detect.py           # Entry point — real-time webcam detection
├── __init__.py
├── evaluate.py             # Offline evaluation against annotated video
├── train.py                # (Optional) supervised calibration trainer
├── requirements.txt        # Minimal CPU-only dependency list
├── setup_ubuntu.sh         # One-shot Ubuntu setup script
├── dataset.zip             # Bundled sample dataset
└── readme.md                  # Unit and integration tests
```

---

## Installation

### Prerequisites

- Python ≥ 3.9
- A webcam (index 0 by default)
- No GPU required

### Quick setup (Ubuntu / Debian)

```bash
git clone https://github.com/MuhammadAsadUllah27/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
bash setup_ubuntu.sh
```

### Manual setup (all platforms)

```bash
git clone https://github.com/MuhammadAsadUllah27/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
source venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** Do **not** install `opencv-python` alongside `opencv-contrib-python`. MediaPipe vendors its own OpenCV build; the contrib variant is a strict superset and avoids `cv2` namespace conflicts that can cause segfaults.

---

## Usage

### Real-time detection (webcam)

```bash
# Built-in webcam, all defaults
python detect.py

# Secondary camera
python detect.py --camera 1

# Silent mode (visual HUD only, no audio alarm)
python detect.py --no-sound

# Tighten microsleep threshold and shorten PERCLOS window
python detect.py --microsleep 0.4 --perclos-window 30

# Emphasise PERCLOS and EAR in the fusion score
python detect.py --w-perclos 0.40 --w-ear 0.30
```

#### Keyboard controls (while the HUD window is focused)

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset detector state (clears PERCLOS buffer + EMA baseline) |
| `s` | Save a snapshot of the current frame to `snapshots/` |
| `p` | Pause / resume |

### Offline evaluation

```bash
# Single annotated video
python evaluate.py --video data/subject_01.mp4 --labels data/subject_01.csv

# Entire folder (each .mp4 must have a matching .csv with the same filename stem)
python evaluate.py --video_dir dataset/test_videos/

# Save per-frame CSVs and JSON summary
python evaluate.py --video_dir dataset/test_videos/ --save_results results/

# Override detector parameters
python evaluate.py --video_dir dataset/test/ --perclos_window 30 --fps 25
```

#### Annotation CSV format

```csv
frame_idx,label
0,alert
1,alert
2,drowsy
150,microsleep
```

Accepted label strings (case-insensitive): `alert`, `drowsy`, `microsleep`.

#### Evaluation output

- Overall accuracy, macro-averaged precision / recall / F1
- Per-class breakdown (precision, recall, F1, support)
- Confusion matrix
- Mean/P95 inference latency and effective FPS
- No-face frame percentage
- Per-frame CSV + JSON summary (with `--save_results`)

---

## Configuration Reference

All parameters can be overridden at the CLI. Key options:

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--width` / `--height` | `640` / `480` | Capture resolution |
| `--fps` | `30` | Frame rate (used for PERCLOS timing) |
| `--perclos-window` | `60` | Rolling PERCLOS window (seconds) |
| `--microsleep` | `0.5` | Minimum continuous eye closure for microsleep (seconds) |
| `--calib-frames` | `60` | Frames used to calibrate personal EAR baseline |
| `--detect-conf` | `0.5` | MediaPipe face detection confidence threshold |
| `--track-conf` | `0.5` | MediaPipe landmark tracking confidence threshold |
| `--no-sound` | — | Disable audio alarm |
| `--no-clips` | — | Do not save alert video clips |

#### Fusion weight flags

All weights are relative and normalised internally. Increasing a weight makes that signal dominate the final drowsiness score.

| Flag | Default | Signal |
|---|---|---|
| `--w-perclos` | `0.35` | PERCLOS (eye closure over time) |
| `--w-ear` | `0.25` | EAR deviation from personal baseline |
| `--w-mar` | `0.15` | MAR (yawning) |
| `--w-blink` | `0.10` | Blink-rate anomaly |
| `--w-pose` | `0.10` | Head-pose deviation |
| `--w-micro` | `0.05` | Microsleep flag |

---

## Dependencies

```
mediapipe>=0.10.14          # Face Mesh landmark extraction (bundles TFLite)
opencv-contrib-python>=4.8  # Video capture, BGR↔RGB, solvePnP head-pose
numpy>=1.24                 # Landmark coordinate math
pygame>=2.5                 # Cross-platform audio alarm
tqdm>=4.65                  # Progress bars for batch evaluation
scipy>=1.11                 # Optional Butterworth low-pass EAR/MAR smoothing
```

PyTorch, torchvision, and Pillow are **not** required — the pipeline contains zero CNN inference.

---

## Project Background

This project went through a significant architectural evolution. The original system used an ensemble of deep-learning models (EfficientNetV2-S + MobileNetV3) requiring GPU inference and checkpoint files. The current version replaces that entirely with a geometry-based MediaPipe pipeline, reducing the dependency footprint from ~8 GB (PyTorch + weights) to ~150 MB, enabling deployment on embedded hardware, edge devices, and laptops without a discrete GPU.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Author

**Muhammad Asad Ullah**
Robotics Software Engineer & Researcher
