from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]


class DetectionEngine:
    """Adapter for model inference outputs."""

    def __init__(self, detector: Any):
        self.detector = detector

    def adapt(self, raw: list[dict]) -> list[Detection]:
        out: list[Detection] = []
        for item in raw:
            out.append(
                Detection(
                    class_id=int(item.get("class_id", -1)),
                    class_name=str(item.get("class_name", "unknown")),
                    confidence=float(item.get("confidence", 0.0)),
                    bbox=list(item.get("bbox", [])),
                )
            )
        return out

    def infer(self, frame) -> list[Detection]:
        return self.adapt(self.detector.detect(frame))
