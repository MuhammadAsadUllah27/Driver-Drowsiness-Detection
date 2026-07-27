"""
Frame Preprocessor

Converts raw OpenCV BGR frames / numpy crops into
normalised PyTorch tensors ready for model inference.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from config.config import data_cfg


_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class FramePreprocessor:
    """
    Converts BGR numpy arrays to normalised tensors.

    Usage
    -----
    prep  = FramePreprocessor()
    batch = prep.face_to_tensor(bgr_crop)   # (1, 3, 224, 224)
    eye   = prep.eye_to_tensor(bgr_eye)     # (1, 3, 64, 64)
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._face_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(data_cfg.image_size),
            transforms.Normalize(_MEAN, _STD),
        ])
        self._eye_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(data_cfg.eye_crop_size),
            transforms.Normalize(_MEAN, _STD),
        ])

    def face_to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """
        BGR numpy (H, W, 3)  →  tensor (1, 3, 224, 224) on device.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t   = self._face_tf(rgb).unsqueeze(0)
        return t.to(self.device, non_blocking=True)

    def eye_to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """
        BGR numpy (H, W, 3)  →  tensor (1, 3, 64, 64) on device.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t   = self._eye_tf(rgb).unsqueeze(0)
        return t.to(self.device, non_blocking=True)

    def tta_tensors(self, bgr: np.ndarray) -> list:
        """
        Returns 5 augmented tensors for test-time augmentation.
        """
        from data.dataset import build_tta_transforms
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil  = Image.fromarray(rgb)
        tfms = build_tta_transforms()
        return [tf(pil).unsqueeze(0).to(self.device) for tf in tfms]
