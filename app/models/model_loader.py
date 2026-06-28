from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from app.config import settings
from app.models.cnn_model import RPSCNN
from app.services.predictor import Predictor

logger = logging.getLogger("rps_api")


class ModelLoader:
    def __init__(self, model_path: Path, classes: list[str]) -> None:
        self.model_path = model_path
        self.classes = classes

    def load_predictor(self) -> Predictor:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        model = RPSCNN(num_classes=len(self.classes)).to(device)
        state_dict = torch.load(self.model_path, map_location=device)

        if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if isinstance(state_dict, dict) and any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return Predictor(model=model, classes=self.classes, device=device)
