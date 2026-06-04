from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class AlertRecord(Base):
    __tablename__ = "alert_records"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    record_uid: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    alert_types: Mapped[list[str]] = mapped_column(JSON, default=list)
    total_detections: Mapped[int] = mapped_column(Integer, default=0)
    alert_count: Mapped[int] = mapped_column(Integer, default=0)
    result_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="image")

    detections: Mapped[list["DetectionRecord"]] = relationship(
        back_populates="alert_record",
        cascade="all, delete-orphan",
    )


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    alert_record_id: Mapped[int] = mapped_column(ForeignKey("alert_records.id", ondelete="CASCADE"))
    class_id: Mapped[int] = mapped_column(Integer, index=True)
    class_name: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    bbox: Mapped[list[int]] = mapped_column(JSON)
    is_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    source_model: Mapped[str] = mapped_column(String(32), default="")

    alert_record: Mapped[AlertRecord] = relationship(back_populates="detections")


class VideoTaskRecord(Base):
    __tablename__ = "video_task_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    input_filename: Mapped[str] = mapped_column(String(255))
    input_path: Mapped[str] = mapped_column(String(512))
    output_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(String(255), default="")
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    detected_frames: Mapped[int] = mapped_column(Integer, default=0)
    total_detections: Mapped[int] = mapped_column(Integer, default=0)
    total_alerts: Mapped[int] = mapped_column(Integer, default=0)
    video_info: Mapped[str] = mapped_column(String(128), default="")
    runtime_state: Mapped[str] = mapped_column(Text, default="{}")
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
