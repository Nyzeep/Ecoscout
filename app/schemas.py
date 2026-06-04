from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]
    alert: bool
    icon: str = ""
    source: str = ""
    track_id: int | None = None
    bin_color: str | None = None
    bin_color_confidence: float | None = None
    bin_type_key: str | None = None
    bin_type_name: str | None = None
    related_bin_type_key: str | None = None
    related_bin_type_name: str | None = None


class SceneInfo(BaseModel):
    status: str
    alert_count: int
    alert_types: list[str]
    normal_count: int
    total: int
    timestamp: str


class DetectImageResponse(BaseModel):
    success: bool = True
    detections: list[DetectionItem]
    scene: SceneInfo
    result_image: str | None = None


class AlertListItem(BaseModel):
    id: str
    time: str
    status: str
    types: list[str]
    total: int
    alert_count: int
    source: str


class AlertListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    records: list[AlertListItem]


class AlertImageResponse(BaseModel):
    image: str


class ClassInfo(BaseModel):
    id: int
    name: str
    en: str
    alert: bool
    icon: str = ""


class StatisticsClassItem(BaseModel):
    class_id: int
    class_name: str
    count: int
    is_alert: bool


class StatisticsResponse(BaseModel):
    total_detections: int
    total_alerts: int
    today_alerts: int
    hourly_alerts: list[int]
    class_stats: list[StatisticsClassItem]
    start_time: str
    alert_record_count: int


class SystemStatusResponse(BaseModel):
    model_loaded: bool
    garbage_model: bool
    fire_model: bool
    smoke_model: bool
    bin_color_model: bool = False
    mode: str
    uptime: str
    class_count: int
    version: str
    name: str


class VideoTaskCreateResponse(BaseModel):
    success: bool = True
    task_id: str
    status: str
    message: str


class VideoTaskStatusResponse(BaseModel):
    success: bool = True
    task_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: int
    message: str
    result_video: str | None = None
    stats: dict[str, Any] | None = None
    active_alerts: list[dict[str, Any]] = Field(default_factory=list)
    active_alert_count: int = 0
    highest_priority_alert: str | None = None
    new_alert_count: int = 0
    sustained_alert_count: int = 0
    ended_alert_count: int = 0


class Base64ImageRequest(BaseModel):
    image: str = Field(..., min_length=16)

