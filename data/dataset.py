"""
Dataset & DataLoader Module

Supports:
• Standard ImageFolder layout (train/val split)
• Eye-crop extraction dataset
• MixUp / CutMix collator
• Weighted random sampler for class imbalance
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config.config import data_cfg, train_cfg
from utils.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class _SimpleImageFolder(Dataset):
    """Minimal ImageFolder-style dataset that avoids the torchvision dependency."""

    def __init__(self, root: Path, transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        self.transform = transform or (lambda image: image)
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, target


def _to_tensor(image: Image.Image, mean: List[float], std: List[float]) -> torch.Tensor:
    array = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor.float()
    tensor = tensor - torch.tensor(mean).view(3, 1, 1)
    tensor = tensor / torch.tensor(std).view(3, 1, 1)
    return tensor


# ─── Transforms ───────────────────────────────────────────────────────────────

def build_transforms(split: str) -> Callable[[Image.Image], torch.Tensor]:
    """Returns a lightweight transform function for 'train', 'val', or 'test'."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    size = data_cfg.image_size

    if split == "train":
        def transform(image: Image.Image) -> torch.Tensor:
            image = image.resize((size[0] + 32, size[1] + 32))
            if data_cfg.aug_horizontal_flip and random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if data_cfg.aug_rotation_degrees > 0:
                angle = random.uniform(-data_cfg.aug_rotation_degrees, data_cfg.aug_rotation_degrees)
                image = image.rotate(angle, resample=Image.BILINEAR, expand=False)
            image = image.crop((0, 0, size[0], size[1])) if image.size[0] >= size[0] and image.size[1] >= size[1] else image.resize(size[::-1])
            return _to_tensor(image, mean, std)

        return transform

    def transform(image: Image.Image) -> torch.Tensor:
        image = image.resize(size[::-1])
        return _to_tensor(image, mean, std)

    return transform


def build_tta_transforms() -> List[Callable[[Image.Image], torch.Tensor]]:
    """Test-time augmentation — returns a list of 5 transform variants."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    size = data_cfg.image_size

    def make_transform(flip: bool = False, rotate: bool = False) -> Callable[[Image.Image], torch.Tensor]:
        def transform(image: Image.Image) -> torch.Tensor:
            image = image.resize(size[::-1])
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if rotate:
                image = image.rotate(10, resample=Image.BILINEAR, expand=False)
            return _to_tensor(image, mean, std)

        return transform

    return [
        make_transform(),
        make_transform(flip=True),
        make_transform(),
        make_transform(rotate=True),
        make_transform(),
    ]


def build_eye_transforms(split: str) -> Callable[[Image.Image], torch.Tensor]:
    """Transforms for 64×64 eye crops."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    size = data_cfg.eye_crop_size

    if split == "train":
        def transform(image: Image.Image) -> torch.Tensor:
            image = image.resize(size[::-1])
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            return _to_tensor(image, mean, std)

        return transform

    def transform(image: Image.Image) -> torch.Tensor:
        image = image.resize(size[::-1])
        return _to_tensor(image, mean, std)

    return transform


# ─── Datasets ─────────────────────────────────────────────────────────────────

class DrowsinessDataset(Dataset):
    """
    Wraps torchvision.datasets.ImageFolder and exposes class weights.

    Expected structure:
        root/
          awake/   *.jpg  *.png  ...
          drowsy/  *.jpg  *.png  ...
    """

    def __init__(self, root: Path, split: str = "train") -> None:
        self.split = split
        self.inner = _SimpleImageFolder(
            root,
            transform=build_transforms(split),
        )
        self.classes = self.inner.classes
        self.class_to_idx = self.inner.class_to_idx
        self._compute_class_weights()
        log.info(
            "DrowsinessDataset (%s) | root=%s | samples=%d | classes=%s",
            split, root, len(self), self.classes,
        )

    def _compute_class_weights(self) -> None:
        targets = torch.tensor(self.inner.targets)
        class_counts = torch.bincount(targets)
        self.class_weights = (1.0 / class_counts.float()).clamp(max=10.0)
        self.sample_weights = self.class_weights[targets]

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.inner[idx]

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self),
            replacement=True,
        )


# ─── MixUp / CutMix Collator ──────────────────────────────────────────────────

class MixUpCutMixCollator:
    """
    Randomly applies MixUp or CutMix to each batch.
    Returns soft labels (tensor of shape [N, num_classes]).
    """

    def __init__(
        self,
        num_classes: int = 2,
        mixup_alpha: float = train_cfg.mixup_alpha,
        cutmix_alpha: float = train_cfg.cutmix_alpha,
        mixup_prob: float = 0.5,
    ) -> None:
        self.num_classes  = num_classes
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob   = mixup_prob

    def __call__(self, batch: List[Tuple[torch.Tensor, int]]):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        soft   = torch.zeros(len(labels), self.num_classes).scatter_(
            1, labels.unsqueeze(1), 1.0
        )

        if random.random() < self.mixup_prob:
            images, soft = self._mixup(images, soft)
        else:
            images, soft = self._cutmix(images, soft)

        return images, soft

    def _mixup(
        self, images: torch.Tensor, soft: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam   = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        idx   = torch.randperm(images.size(0))
        mixed = lam * images + (1 - lam) * images[idx]
        mixed_soft = lam * soft + (1 - lam) * soft[idx]
        return mixed, mixed_soft

    def _cutmix(
        self, images: torch.Tensor, soft: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lam  = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        idx  = torch.randperm(images.size(0))
        _, _, h, w = images.shape
        cut_h = int(h * np.sqrt(1 - lam))
        cut_w = int(w * np.sqrt(1 - lam))
        cx    = np.random.randint(w)
        cy    = np.random.randint(h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam_adj  = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
        mixed_soft = lam_adj * soft + (1 - lam_adj) * soft[idx]
        return images, mixed_soft


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def build_dataloaders(
    train_root: Path,
    val_root: Path,
    use_mixup: bool = True,
) -> Dict[str, DataLoader]:
    """
    Constructs train and val DataLoaders.

    Parameters
    ----------
    train_root : Path  to training split (contains awake/ drowsy/ subfolders)
    val_root   : Path  to validation split
    use_mixup  : bool  whether to apply MixUp/CutMix on train

    Returns
    -------
    {"train": DataLoader, "val": DataLoader}
    """
    train_ds = DrowsinessDataset(train_root, split="train")
    val_ds   = DrowsinessDataset(val_root,   split="val")

    sampler = train_ds.make_weighted_sampler() if train_cfg.use_weighted_sampler else None
    shuffle  = sampler is None

    collate_fn = MixUpCutMixCollator(num_classes=data_cfg.num_classes) if use_mixup else None

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size * 2,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))
    return {"train": train_loader, "val": val_loader}
