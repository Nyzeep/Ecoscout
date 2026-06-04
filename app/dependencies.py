from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.services.detection_service import DetectionService


@lru_cache
def get_detection_service() -> DetectionService:
    settings = get_settings()
    return DetectionService(settings=settings)
