import time
from typing import Any

import torch
from torch import nn


class Predictor:
    def __init__(self, model: nn.Module, classes: list[str], device: torch.device) -> None:
        self.model = model
        self.classes = classes
        self.device = device

    def predict(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        image_tensor = image_tensor.to(self.device)
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        inference_time = (time.perf_counter() - start_time) * 1000
        confidence_value = round(float(confidence.item() * 100), 2)
        prediction = self.classes[int(predicted_idx.item())]

        return {
            "prediction": prediction,
            "confidence": confidence_value,
            "inference_time_ms": round(inference_time, 2),
        }
