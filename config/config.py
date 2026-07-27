"""
Driver Drowsiness Detection System
Configuration Module — all hyperparameters, paths, and constants live here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Dict
import os


# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Dataset paths and preprocessing settings."""
    # ── Paths ──────────────────────────────────────────────────────────────────
    dataset_root: Path = ROOT / "dataset"
    # Expected layout:
    #   dataset/
    #     train/
    #       awake/   (open eyes)
    #       drowsy/  (closed/half-open eyes)
    #     val/
    #       awake/
    #       drowsy/
    #     test/      (optional)

    processed_dir: Path = ROOT / "data" / "processed"
    checkpoints_dir: Path = ROOT / "checkpoints"
    logs_dir: Path = ROOT / "logs"

    # ── Image ──────────────────────────────────────────────────────────────────
    image_size: Tuple[int, int] = (224, 224)
    eye_crop_size: Tuple[int, int] = (64, 64)

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug_brightness: float = 0.3
    aug_contrast: float = 0.3
    aug_saturation: float = 0.2
    aug_hue: float = 0.1
    aug_rotation_degrees: float = 15.0
    aug_horizontal_flip: bool = True

    # ── Classes ───────────────────────────────────────────────────────────────
    class_names: List[str] = field(default_factory=lambda: ["awake", "drowsy"])
    num_classes: int = 2

    # ── DataLoader ────────────────────────────────────────────────────────────
    num_workers: int = 2
    pin_memory: bool = False
    prefetch_factor: int = 1


@dataclass
class ModelConfig:
    """Model architecture and selection settings."""
    # Options: "efficientnet_v2_s", "mobilenet_v3_large", "resnet50", "convnext_tiny"
    backbone: str = "mobilenet_v3_large"
    pretrained: bool = True
    dropout_rate: float = 0.25
    freeze_bn: bool = False

    # Eye-based secondary model (lightweight CNN)
    use_eye_model: bool = True
    eye_model_name: str = "mobilenet_v3_small"

    # Ensemble
    ensemble_weights: Tuple[float, float] = (0.6, 0.4)  # (face_model, eye_model)


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 5e-4
    label_smoothing: float = 0.05

    # Scheduler
    scheduler: str = "cosine_warmup"  # "cosine_warmup" | "plateau" | "step"
    warmup_epochs: int = 3
    min_lr: float = 1e-5

    # Regularisation
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    use_tta: bool = True                # Test-time augmentation at inference

    # Early stopping
    patience: int = 6
    min_delta: float = 1e-4

    # Mixed precision
    use_amp: bool = True

    # Gradient clipping
    grad_clip: float = 1.0

    # Class imbalance
    use_weighted_sampler: bool = True
    focal_loss_gamma: float = 1.5


@dataclass
class InferenceConfig:
    """Real-time inference settings."""
    # ── Thresholds ────────────────────────────────────────────────────────────
    drowsy_threshold: float = 0.35          # Fused score at/above this → drowsy

    # EAR (Eye Aspect Ratio) — geometric backup
    ear_threshold: float = 0.25
    ear_consec_frames: int = 20             # Frames below threshold → alert

    # MAR (Mouth Aspect Ratio) — yawn detection
    mar_threshold: float = 0.75
    mar_consec_frames: int = 15

    # Head pose thresholds (degrees)
    pitch_threshold: float = 20.0           # Nodding
    yaw_threshold: float = 35.0             # Looking away

    # ── Temporal Smoothing ────────────────────────────────────────────────────
    smoothing_window: int = 10              # Rolling avg over N frames
    alert_trigger_frames: int = 15         # Consecutive drowsy frames → alarm

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    # ── Alert ─────────────────────────────────────────────────────────────────
    alert_sound: bool = True
    alert_visual: bool = True
    alert_cooldown_sec: float = 3.0         # Min seconds between alerts

    # ── Face / Landmark Detection ─────────────────────────────────────────────
    face_detection_confidence: float = 0.7
    landmark_confidence: float = 0.6
    use_mediapipe: bool = True              # MediaPipe for 468 landmarks
    use_dlib_fallback: bool = True          # dlib 68-point as fallback

    # ── Logging ───────────────────────────────────────────────────────────────
    log_fps_interval: int = 30
    save_alert_clips: bool = False
    clips_dir: Path = ROOT / "alert_clips"

    # ── Model Path ────────────────────────────────────────────────────────────
    model_checkpoint: Path = ROOT / "checkpoints" / "best_model.pth"
    eye_model_checkpoint: Path = ROOT / "checkpoints" / "best_eye_model.pth"


# ── Singleton Instances ────────────────────────────────────────────────────────
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
infer_cfg = InferenceConfig()

# ── Auto-create directories ────────────────────────────────────────────────────
for _dir in [
    data_cfg.processed_dir,
    data_cfg.checkpoints_dir,
    data_cfg.logs_dir,
]:
    _dir.mkdir(parents=True, exist_ok=True)
