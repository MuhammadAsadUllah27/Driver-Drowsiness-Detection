import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import data_cfg, model_cfg, train_cfg
from models.architectures import DrowsinessModel, EyeModel


def test_lighter_default_training_config_is_used():
    assert model_cfg.backbone == "mobilenet_v3_large"
    assert data_cfg.num_workers <= 2
    assert train_cfg.batch_size <= 16
    assert train_cfg.num_epochs <= 25


def test_training_models_can_be_instantiated():
    face_model = DrowsinessModel(backbone="mobilenet_v3_large", num_classes=2, pretrained=False)
    eye_model = EyeModel(num_classes=2)

    dummy = torch.randn(2, 3, 224, 224)
    assert face_model(dummy).shape == (2, 2)
    assert eye_model(dummy).shape == (2, 2)
