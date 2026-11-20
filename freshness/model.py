from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms

from freshness.config import CLASS_LABELS


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class FreshnessModel:
    """Loads a locally trained ResNet18 checkpoint and classifies food images."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None
        self.classes: list[str] = list(CLASS_LABELS)
        self.error: str | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _load(self) -> None:
        if not self.checkpoint_path.is_file():
            self.error = "No trained model checkpoint was found."
            return

        try:
            checkpoint: dict[str, Any] = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
            classes = checkpoint.get("classes")
            state_dict = checkpoint.get("state_dict")
            if not isinstance(classes, list) or set(classes) != set(CLASS_LABELS):
                raise ValueError("Checkpoint classes must be fresh, aging, and spoiled.")
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint is missing a model state_dict.")

            network = models.resnet18(weights=None)
            network.fc = torch.nn.Linear(network.fc.in_features, len(classes))
            network.load_state_dict(state_dict)
            network.eval()
            network.to(self.device)
            self.classes = classes
            self.model = network
            self.error = None
        except Exception as exc:  # Keep the API available and expose a useful health status.
            self.model = None
            self.error = f"Could not load model checkpoint: {exc}"

    def predict(self, image: Image.Image) -> Prediction:
        if self.model is None:
            raise RuntimeError(self.error or "The model is not ready.")

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)[0].cpu()

        probabilities = {
            name: float(probabilities_tensor[index])
            for index, name in enumerate(self.classes)
        }
        label = max(probabilities, key=probabilities.get)
        return Prediction(label, probabilities[label], probabilities)
