from pathlib import Path


class Settings:
    project_root: Path = Path(__file__).resolve().parent.parent
    model_path: Path = project_root / "model" / "model.pth"
    class_names: list[str] = ["paper", "rock", "scissors"]
    image_size: int = 224


settings = Settings()
