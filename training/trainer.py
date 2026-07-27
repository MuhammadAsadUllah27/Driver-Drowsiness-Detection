"""
Training Engine

Features:
  • Mixed precision (AMP)
  • Cosine LR + warmup scheduler
  • Early stopping
  • MixUp / CutMix support (soft labels)
  • Class-weighted focal loss
  • Best checkpoint saving
  • TensorBoard logging
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

try:
    import triton.backends.compiler as _triton_backend_compiler  # type: ignore
except Exception:
    try:
        import triton.compiler as _triton_backend_compiler  # type: ignore
    except Exception:
        _triton_backend_compiler = None

if _triton_backend_compiler is not None:
    triton_backends = ModuleType("triton.backends")
    triton_backends.compiler = _triton_backend_compiler
    sys.modules.setdefault("triton.backends", triton_backends)
    sys.modules["triton.backends.compiler"] = _triton_backend_compiler
    try:
        import triton as _triton_pkg  # type: ignore
    except Exception:
        _triton_pkg = None
    if _triton_pkg is not None:
        setattr(_triton_pkg, "backends", triton_backends)

from torch.optim import AdamW

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency in this environment
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def add_scalars(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

from config.config import data_cfg, train_cfg
from utils.logger import get_logger

log = get_logger(__name__, log_file=data_cfg.logs_dir / "train.log")


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    Supports both hard (long) and soft (float) labels.
    """

    def __init__(
        self,
        gamma: float = train_cfg.focal_loss_gamma,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = train_cfg.label_smoothing,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma          = gamma
        self.alpha          = alpha
        self.label_smoothing = label_smoothing
        self.reduction      = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        num_classes = logits.size(1)

        # Soft labels (MixUp / CutMix)
        if targets.dim() == 2:
            log_prob = F.log_softmax(logits, dim=1)
            prob     = log_prob.exp()
            pt       = (prob * targets).sum(dim=1)
            focal_w  = (1 - pt) ** self.gamma
            loss     = -(targets * log_prob).sum(dim=1)
            loss     = focal_w * loss
        else:
            # Hard labels with optional smoothing
            if self.label_smoothing > 0:
                smooth_targets = torch.full_like(
                    logits, self.label_smoothing / (num_classes - 1)
                )
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            else:
                smooth_targets = F.one_hot(targets, num_classes).float()

            log_prob = F.log_softmax(logits, dim=1)
            prob     = log_prob.exp()
            pt       = (prob * smooth_targets).sum(dim=1)
            focal_w  = (1 - pt) ** self.gamma
            loss     = -(smooth_targets * log_prob).sum(dim=1)
            loss     = focal_w * loss

        if self.alpha is not None:
            if targets.dim() == 2:
                at = (self.alpha.to(logits.device) * targets).sum(dim=1)
            else:
                at = self.alpha.to(logits.device)[targets]
            loss = at * loss

        return loss.mean() if self.reduction == "mean" else loss.sum()


# ─── Metrics ──────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Works with both hard and soft targets."""
    preds = logits.argmax(dim=1)
    if targets.dim() == 2:
        hard = targets.argmax(dim=1)
    else:
        hard = targets
    return (preds == hard).float().mean().item()


# ─── Trainer ──────────────────────────────────────────────────────────────────

class Trainer:
    """
    Full training loop with early stopping and checkpointing.

    Parameters
    ----------
    model      : nn.Module to train
    loaders    : {"train": DataLoader, "val": DataLoader}
    save_dir   : directory for checkpoints
    model_tag  : used in checkpoint filename
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        save_dir: Path = data_cfg.checkpoints_dir,
        model_tag: str = "face_model",
    ) -> None:
        self.model     = model
        self.loaders   = loaders
        self.save_dir  = Path(save_dir)
        self.model_tag = model_tag
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        log.info("Training device: %s", self.device)
        self.model.to(self.device)

        # ── Optimiser ─────────────────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        # ── Scheduler: linear warmup → cosine ─────────────────────────────────
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=train_cfg.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.num_epochs - train_cfg.warmup_epochs,
            eta_min=train_cfg.min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[train_cfg.warmup_epochs],
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = FocalLoss()

        # ── AMP ───────────────────────────────────────────────────────────────
        self.use_amp = train_cfg.use_amp and self.device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        # ── TensorBoard ───────────────────────────────────────────────────────
        tb_dir = data_cfg.logs_dir / "tensorboard" / model_tag
        self.writer = SummaryWriter(str(tb_dir))

        # ── State ─────────────────────────────────────────────────────────────
        self.best_val_acc  = 0.0
        self.no_improve    = 0
        self.global_step   = 0

    # ─── One epoch ────────────────────────────────────────────────────────────

    def _run_epoch(self, split: str) -> Tuple[float, float]:
        is_train = split == "train"
        self.model.train(is_train)
        loader = self.loaders[split]

        loss_m = AverageMeter()
        acc_m  = AverageMeter()

        with torch.set_grad_enabled(is_train):
            for images, targets in loader:
                images  = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss   = self.criterion(logits, targets)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), train_cfg.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
                    self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)

                loss_m.update(loss.item(), images.size(0))
                acc_m.update(accuracy(logits.detach(), targets.detach()), images.size(0))

        return loss_m.avg, acc_m.avg

    # ─── Main train loop ──────────────────────────────────────────────────────

    def fit(self) -> Dict[str, float]:
        """
        Run full training loop.

        Returns
        -------
        dict with best_val_acc and best checkpoint path.
        """
        log.info("Starting training: %d epochs | %s", train_cfg.num_epochs, self.model_tag)
        best_ckpt_path = self.save_dir / f"best_{self.model_tag}.pth"

        for epoch in range(1, train_cfg.num_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch("train")
            val_loss, val_acc     = self._run_epoch("val")
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            log.info(
                "Epoch %3d/%d | TrainLoss=%.4f TrainAcc=%.4f | "
                "ValLoss=%.4f ValAcc=%.4f | LR=%.2e | %.1fs",
                epoch, train_cfg.num_epochs,
                train_loss, train_acc, val_loss, val_acc, lr, elapsed,
            )

            # TensorBoard
            self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("acc",  {"train": train_acc,  "val": val_acc},  epoch)
            self.writer.add_scalar("lr", lr, epoch)

            # Checkpoint
            if val_acc > self.best_val_acc + train_cfg.min_delta:
                self.best_val_acc = val_acc
                self.no_improve   = 0
                torch.save(
                    {
                        "epoch":       epoch,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "val_acc":     val_acc,
                        "val_loss":    val_loss,
                    },
                    best_ckpt_path,
                )
                log.info("  ✓ New best checkpoint saved (val_acc=%.4f)", val_acc)
            else:
                self.no_improve += 1

            # Early stopping
            if self.no_improve >= train_cfg.patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)",
                         epoch, train_cfg.patience)
                break

        self.writer.close()
        log.info("Training complete. Best val_acc=%.4f | checkpoint=%s",
                 self.best_val_acc, best_ckpt_path)
        return {"best_val_acc": self.best_val_acc, "checkpoint": str(best_ckpt_path)}
