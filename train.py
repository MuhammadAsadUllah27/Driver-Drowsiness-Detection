"""
train.py — Entry point for training the drowsiness detection models.

Usage
-----
# Train with your dataset (default paths from config)
python train.py

# Override dataset path
python train.py --data /path/to/your/dataset

# Choose backbone
python train.py --backbone convnext_tiny

# Train both face model AND eye model
python train.py --train-eye

Run  python train.py --help  for all options.
"""

import argparse
import shutil
import sys
from pathlib import Path

# ── Allow running from project root ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from config.config import data_cfg, model_cfg, train_cfg
from data.dataset import build_dataloaders
from models.architectures import DrowsinessModel, EyeModel
from training.trainer import Trainer
from utils.logger import get_logger

log = get_logger(__name__, log_file=data_cfg.logs_dir / "train.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Drowsiness Detection Model")
    p.add_argument("--data",       type=Path, default=data_cfg.dataset_root,
                   help="Root of dataset (must contain train/ and val/ subfolders)")
    p.add_argument("--backbone",   type=str,  default=model_cfg.backbone,
                   choices=["efficientnet_v2_s","efficientnet_v2_m",
                            "mobilenet_v3_large","resnet50","convnext_tiny"],
                   help="Backbone architecture")
    p.add_argument("--epochs",     type=int,  default=train_cfg.num_epochs)
    p.add_argument("--batch-size", type=int,  default=train_cfg.batch_size)
    p.add_argument("--lr",         type=float,default=train_cfg.learning_rate)
    p.add_argument("--no-amp",     action="store_true",
                   help="Disable mixed precision training")
    p.add_argument("--train-eye",  action="store_true",
                   help="Also train the lightweight eye-crop model")
    p.add_argument("--no-mixup",   action="store_true",
                   help="Disable MixUp / CutMix augmentation")
    return p.parse_args()


def ensure_dataset_layout(root: Path) -> None:
    """Validate that `train/` and `val/` class folders exist.

    This function intentionally does NOT attempt to create or populate
    the dataset from raw captures. If the expected layout is missing,
    it prints an error and exits so the user can prepare the dataset
    explicitly outside of this tool.
    """
    train_dir = root / "train"
    val_dir = root / "val"
    class_names = ["awake", "drowsy"]

    def _has_images(split_dir: Path) -> bool:
        return any((split_dir / cls).exists() and any((split_dir / cls).iterdir()) for cls in class_names)

    if train_dir.exists() and val_dir.exists() and _has_images(train_dir) and _has_images(val_dir):
        return

    print(
        f"\n[ERROR] Dataset layout not found under {root}.\n"
        "Expected: <root>/train/awake, <root>/train/drowsy, <root>/val/awake, <root>/val/drowsy."
    )
    sys.exit(1)


def check_dataset(root: Path) -> None:
    """Validate that the dataset structure is correct."""
    ensure_dataset_layout(root)
    train_dir = root / "train"
    val_dir = root / "val"

    errors = []
    for split_dir in [train_dir, val_dir]:
        classes = [d for d in split_dir.iterdir() if d.is_dir()]
        if len(classes) < 2:
            errors.append(
                f"{split_dir} must contain at least 2 class folders "
                f"(e.g. 'awake/' and 'drowsy/').  Found: {[c.name for c in classes]}"
            )

    if errors:
        print("\n[ERROR] Dataset structure problems:")
        for e in errors:
            print(f"  • {e}")
        print("\nExpected layout:")
        print("  dataset/")
        print("    train/")
        print("      awake/    ← images of alert driver")
        print("      drowsy/   ← images of drowsy driver")
        print("    val/")
        print("      awake/")
        print("      drowsy/")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config
    model_cfg.backbone       = args.backbone
    train_cfg.num_epochs     = args.epochs
    train_cfg.batch_size     = args.batch_size
    train_cfg.learning_rate  = args.lr
    train_cfg.use_amp        = not args.no_amp

    log.info("=" * 60)
    log.info("Driver Drowsiness Detection — Training")
    log.info("=" * 60)
    log.info("Device        : %s", "CUDA" if torch.cuda.is_available() else "CPU")
    log.info("Backbone      : %s", model_cfg.backbone)
    log.info("Dataset root  : %s", args.data)
    log.info("Epochs        : %d", train_cfg.num_epochs)
    log.info("Batch size    : %d", train_cfg.batch_size)
    log.info("Learning rate : %g", train_cfg.learning_rate)
    log.info("AMP           : %s", train_cfg.use_amp)
    log.info("=" * 60)

    # ── Validate dataset ──────────────────────────────────────────────────────
    check_dataset(args.data)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    loaders = build_dataloaders(
        train_root=args.data / "train",
        val_root=args.data / "val",
        use_mixup=not args.no_mixup,
    )

    # ── Train Face Model ──────────────────────────────────────────────────────
    log.info("\n--- Training Face / Full-Frame Model ---")
    face_model = DrowsinessModel(
        backbone=model_cfg.backbone,
        num_classes=data_cfg.num_classes,
        pretrained=model_cfg.pretrained,
    )

    face_trainer = Trainer(
        model=face_model,
        loaders=loaders,
        model_tag="face_model",
    )
    face_result = face_trainer.fit()
    log.info("Face model done — best val_acc: %.4f", face_result["best_val_acc"])
    log.info("Checkpoint: %s", face_result["checkpoint"])

    # ── Train Eye Model (optional) ────────────────────────────────────────────
    if args.train_eye:
        log.info("\n--- Training Eye-Crop Model ---")
        # Reuse same loaders (eye crops are extracted online in the pipeline,
        # here we train on full images as proxy — swap for eye-crop dataset if available)
        eye_model = EyeModel(num_classes=data_cfg.num_classes)
        eye_trainer = Trainer(
            model=eye_model,
            loaders=loaders,
            model_tag="eye_model",
        )
        eye_result = eye_trainer.fit()
        log.info("Eye model done — best val_acc: %.4f", eye_result["best_val_acc"])
        log.info("Checkpoint: %s", eye_result["checkpoint"])

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Face model accuracy : {face_result['best_val_acc']:.2%}")
    print(f"  Checkpoint saved to : checkpoints/")
    print("\n  To run real-time detection:")
    print("    python detect.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
