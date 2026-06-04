from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "社区垃圾与火情识别预警系统"
    app_version: str = "2.0.0"
    debug: bool = False
    api_prefix: str = "/api"

    database_url: str = Field(
        default=f"sqlite:///{(PROJECT_DIR / 'garbage_system.db').as_posix()}",
    )
    redis_url: str = "redis://localhost:6379/0"
    celery_task_always_eager: bool = False

    max_upload_size_mb: int = 200
    video_default_skip_frames: int = 1

    models_dir: Path = BASE_DIR / "models"
    uploads_dir: Path = BASE_DIR / "uploads"
    templates_dir: Path = BASE_DIR / "templates"

    garbage_pt_model: Path = BASE_DIR / "models" / "garbege.pt"
    fire_pt_model: Path = BASE_DIR / "models" / "only_fire.pt"
    smoke_pt_model: Path = BASE_DIR / "models" / "fire_smoke.pt"

    garbage_onnx_model: Path = BASE_DIR / "models" / "garbege.onnx"
    fire_onnx_model: Path = BASE_DIR / "models" / "only_fire.onnx"
    smoke_onnx_model: Path = BASE_DIR / "models" / "fire_smoke.onnx"
    bin_color_resnet18_model: Path = BASE_DIR / "models" / "bin_color_resnet18.pt"
    bin_color_min_confidence: float = 0.4
    # For combined smoke/fire models, default mapping is usually:
    # class 0 = smoke, class 1 = fire.
    smoke_model_include_fire: bool = True
    smoke_model_fire_class_id: int = 1
    smoke_model_smoke_class_id: int = 0

    default_conf_threshold: float = 0.5
    garbage_bin_conf_threshold: float = 0.4
    garbage_litter_conf_threshold: float = 0.38
    fire_conf_threshold: float = 0.15
    fire_conf_threshold_without_smoke: float = 0.65
    smoke_conf_threshold: float = 0.30
    fire_low_conf_color_ratio_threshold: float = 0.10
    default_iou_threshold: float = 0.3
    fire_nms_iou_threshold: float = 0.35
    smoke_nms_iou_threshold: float = 0.20
    fire_nms_ios_threshold: float = 0.60
    smoke_nms_ios_threshold: float = 0.70
    overflow_nms_iou_threshold: float = 0.30
    overflow_nms_ios_threshold: float = 0.60
    litter_nms_iou_threshold: float = 0.25
    litter_nms_ios_threshold: float = 0.55
    video_track_iou_default: float = 0.40
    fire_track_iou_threshold: float = 0.35
    smoke_track_iou_threshold: float = 0.30
    garbage_track_iou_threshold: float = 0.45
    overflow_track_iou_threshold: float = 0.45
    video_bbox_smooth_alpha_default: float = 0.65
    fire_bbox_smooth_alpha: float = 0.55
    smoke_bbox_smooth_alpha: float = 0.60
    garbage_bbox_smooth_alpha: float = 0.70
    overflow_bbox_smooth_alpha: float = 0.70

    @field_validator("debug", mode="before")
    @classmethod
    def _normalize_debug(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev", "development"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
