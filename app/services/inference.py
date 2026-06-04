from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


def _ensure_ultralytics_config_dir() -> None:
    """
    Force Ultralytics to use a writable config directory inside project root.
    This avoids permission issues on %APPDATA%/Ultralytics/settings.json.
    """
    project_root = Path(__file__).resolve().parents[2]
    cfg_dir = project_root / ".ultralytics"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg_dir))


def _preferred_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


@dataclass
class RawPrediction:
    class_id: int
    confidence: float
    bbox: list[int]


class InferenceBackend(Protocol):
    model_type: str
    loaded: bool

    def predict(self, image: np.ndarray, conf_threshold: float, iou_threshold: float) -> list[RawPrediction]:
        ...


def _iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def non_max_suppression(predictions: list[RawPrediction], iou_threshold: float) -> list[RawPrediction]:
    sorted_preds = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    kept: list[RawPrediction] = []
    while sorted_preds:
        current = sorted_preds.pop(0)
        kept.append(current)
        sorted_preds = [
            candidate
            for candidate in sorted_preds
            if candidate.class_id != current.class_id or _iou(candidate.bbox, current.bbox) < iou_threshold
        ]
    return kept


class UltralyticsBackend:
    model_type = "ultralytics"

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.loaded = False
        self._model = None
        self.device = _preferred_torch_device()

        if not model_path.exists():
            return

        try:
            _ensure_ultralytics_config_dir()
            self._register_spd_compat()
            from ultralytics import YOLO

            self._model = YOLO(str(model_path))
            self.loaded = True
        except Exception:
            self._model = None
            self.loaded = False

    @staticmethod
    def _register_spd_compat() -> None:
        """
        Register the custom SPD layer name used by some checkpoints.
        This keeps compatibility with weights trained from modified ultralytics forks.
        """
        try:
            import torch
            import torch.nn as nn
            import ultralytics.nn.modules.block as block
        except Exception:
            return

        if hasattr(block, "space_to_depth"):
            return

        class space_to_depth(nn.Module):  # noqa: N801 - keep exact class name for checkpoint lookup
            def __init__(self, dimension: int = 1):
                super().__init__()
                self.d = dimension

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.cat(
                    [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]],
                    1,
                )

        setattr(block, "space_to_depth", space_to_depth)

    def predict(self, image: np.ndarray, conf_threshold: float, iou_threshold: float) -> list[RawPrediction]:
        if not self.loaded or self._model is None:
            return []

        output = self._model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=self.device,
        )[0]
        predictions: list[RawPrediction] = []
        for box in output.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            predictions.append(
                RawPrediction(class_id=class_id, confidence=confidence, bbox=[x1, y1, x2, y2]),
            )
        return predictions


class OnnxYoloBackend:
    model_type = "onnxruntime"

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.loaded = False
        self._session = None
        self.providers: list[str] = []
        self._input_name = ""
        self._input_hw = (640, 640)

        if not model_path.exists():
            return

        try:
            import onnxruntime as ort

            available = set(ort.get_available_providers())
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            self.providers = list(self._session.get_providers())
            model_input = self._session.get_inputs()[0]
            self._input_name = model_input.name
            input_shape = model_input.shape
            if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                self._input_hw = (input_shape[2], input_shape[3])
            self.loaded = True
        except Exception:
            self._session = None
            self.loaded = False

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        input_h, input_w = self._input_hw
        h, w = image.shape[:2]
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) / 2
        pad_y = (input_h - new_h) / 2
        left = int(np.floor(pad_x))
        top = int(np.floor(pad_y))
        canvas[top : top + new_h, left : left + new_w] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, 0), scale, (pad_x, pad_y)

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        image_shape: tuple[int, int],
        scale: float,
        padding: tuple[float, float],
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[RawPrediction]:
        if not outputs:
            return []

        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2:
            return []

        if raw.shape[0] < raw.shape[1]:
            raw = raw.transpose(1, 0)

        h, w = image_shape
        pad_x, pad_y = padding
        predictions: list[RawPrediction] = []
        for row in raw:
            if row.shape[0] < 5:
                continue

            boxes = row[:4]
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < conf_threshold:
                continue

            cx, cy, bw, bh = boxes.tolist()
            x1 = int(max(0, (cx - bw / 2 - pad_x) / scale))
            y1 = int(max(0, (cy - bh / 2 - pad_y) / scale))
            x2 = int(min(w, (cx + bw / 2 - pad_x) / scale))
            y2 = int(min(h, (cy + bh / 2 - pad_y) / scale))
            if x2 <= x1 or y2 <= y1:
                continue

            predictions.append(
                RawPrediction(
                    class_id=class_id,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                ),
            )

        return non_max_suppression(predictions, iou_threshold)

    def predict(self, image: np.ndarray, conf_threshold: float, iou_threshold: float) -> list[RawPrediction]:
        if not self.loaded or self._session is None:
            return []

        tensor, scale, padding = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: tensor})
        return self._postprocess(outputs, image.shape[:2], scale, padding, conf_threshold, iou_threshold)

