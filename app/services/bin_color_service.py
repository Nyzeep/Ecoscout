from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms


@dataclass
class BinColorPrediction:
    label: str
    confidence: float


class ResNet18BinColorService:
    def __init__(self, checkpoint_path: Path, use_cpu: bool = False):
        self.checkpoint_path = checkpoint_path
        self.loaded = False
        self.device = torch.device("cpu")
        self.model: nn.Module | None = None
        self.idx_to_class: dict[int, str] = {}
        self.img_size = 224
        self._transform = None

        if not self.checkpoint_path.exists():
            return

        try:
            if torch.cuda.is_available() and not use_cpu:
                self.device = torch.device("cuda")

            ckpt = torch.load(self.checkpoint_path.as_posix(), map_location=self.device)
            class_to_idx: dict[str, int] = ckpt["class_to_idx"]
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}
            self.img_size = int(ckpt.get("img_size", 224))
            mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
            std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])

            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_to_idx))
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            self.model = model

            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ],
            )
            self.loaded = True
        except Exception:
            self.loaded = False
            self.model = None

    def predict(self, crop_bgr: np.ndarray) -> BinColorPrediction | None:
        if not self.loaded or self.model is None or self._transform is None or crop_bgr.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = self._transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        label = self.idx_to_class.get(int(idx.item()), str(int(idx.item())))
        return BinColorPrediction(label=label, confidence=float(conf.item()))
