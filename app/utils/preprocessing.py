from io import BytesIO

from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

from app.config import settings


class ImagePreprocessor:
    def __init__(self, image_size: int = settings.image_size) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError("Unsupported or corrupted image") from exc

        tensor = self.transform(image)
        return tensor.unsqueeze(0)
