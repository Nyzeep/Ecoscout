from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.celery_app import celery_app
from app.config import Settings
from app.constants import ALLOWED_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.database import get_db
from app.dependencies import get_detection_service
from app.schemas import (
    AlertImageResponse,
    AlertListResponse,
    Base64ImageRequest,
    DetectImageResponse,
    StatisticsResponse,
    SystemStatusResponse,
    VideoTaskCreateResponse,
    VideoTaskStatusResponse,
)
from app.services.detection_service import DetectionService
from app.upgrade import AlarmEngine, DetectionEngine, TrackEngine, UpgradePipeline
from app.services.record_service import RecordService
from app.tasks import process_video_task, run_video_task
from app.utils import base64_to_frame


def build_api_router(settings: Settings, started_at: str) -> APIRouter:
    router = APIRouter(prefix=settings.api_prefix)
    record_service = RecordService(settings.uploads_dir)
    # Shared tracker state for image/base64 endpoints so continuous requests can keep stable IDs.
    upgrade_pipeline = UpgradePipeline(
        detection_engine=DetectionEngine(None),
        track_engine=TrackEngine(),
        alarm_engine=AlarmEngine(min_consecutive_frames=2),
    )

    def has_celery_worker() -> bool:
        if settings.celery_task_always_eager:
            return True
        try:
            return bool(celery_app.control.ping(timeout=0.8))
        except Exception:
            return False

    def start_local_video_task(task_id: str, input_path: Path, skip_frames: int) -> None:
        threading.Thread(
            target=run_video_task,
            kwargs={
                "task_id": task_id,
                "input_path": input_path.as_posix(),
                "skip_frames": skip_frames,
            },
            daemon=True,
        ).start()

    def validate_extension(filename: str, expected: set[str] | None = None) -> str:
        if "." not in filename:
            raise HTTPException(status_code=400, detail="鏂囦欢缂哄皯鍚庣紑")
        ext = filename.rsplit(".", 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        if expected is not None and ext not in expected:
            raise HTTPException(status_code=400, detail="File type mismatch")
        return ext

    async def read_image_file(file: UploadFile) -> np.ndarray:
        validate_extension(file.filename or "", IMAGE_EXTENSIONS)
        content = await file.read()
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="鍥剧墖瑙ｆ瀽澶辫触")
        return image

    def attach_track_ids(detections: list[dict]) -> list[dict]:
        pipe_result = upgrade_pipeline.run_detections(detections, timestamp=time.time())
        tracks = pipe_result.tracks
        out: list[dict] = []
        for idx, det in enumerate(detections):
            item = det.copy()
            if idx < len(tracks):
                item["track_id"] = int(tracks[idx].track_id)
            out.append(item)
        return out

    def build_detect_response(payload: dict) -> dict:
        return {
            "success": True,
            "detections": [
                {
                    "class_id": item["class_id"],
                    "class_name": item["class_name"],
                    "confidence": item["confidence"],
                    "bbox": item["bbox"],
                    "alert": item["alert"],
                    "icon": item.get("icon", ""),
                    "source": item.get("source_model", ""),
                    "track_id": item.get("track_id"),
                    "bin_color": item.get("bin_color"),
                    "bin_color_confidence": item.get("bin_color_confidence"),
                    "bin_type_key": item.get("bin_type_key"),
                    "bin_type_name": item.get("bin_type_name"),
                    "related_bin_type_key": item.get("related_bin_type_key"),
                    "related_bin_type_name": item.get("related_bin_type_name"),
                }
                for item in payload["detections"]
            ],
            "scene": payload["scene"],
            "result_image": payload.get("result_image"),
        }

    @router.post("/detect/image", response_model=DetectImageResponse)
    async def detect_image(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        detection_service: DetectionService = Depends(get_detection_service),
    ) -> dict:
        image = await read_image_file(file)
        # Image upload should alert on every request (no cooldown).
        detections = detection_service.detect(image)
        detections = attach_track_ids(detections)
        rendered = detection_service.draw_boxes(image, detections)
        payload = detection_service.build_response(image, detections, with_image=True)
        record_service.create_alert_record(db, payload["scene"], detections, rendered, source="image")
        return build_detect_response(payload)

    @router.post("/detect/base64", response_model=DetectImageResponse)
    async def detect_base64(
        request: Base64ImageRequest,
        db: Session = Depends(get_db),
        detection_service: DetectionService = Depends(get_detection_service),
    ) -> dict:
        try:
            image = base64_to_frame(request.image)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="鍥剧墖瑙ｆ瀽澶辫触") from exc

        # Camera/base64 requests follow the same no-cooldown behavior as image upload.
        detections = detection_service.detect(image)
        detections = attach_track_ids(detections)
        rendered = detection_service.draw_boxes(image, detections)
        payload = detection_service.build_response(image, detections, with_image=True)
        record_service.create_alert_record(db, payload["scene"], detections, rendered, source="camera")
        return build_detect_response(payload)

    @router.post("/detect/video", response_model=VideoTaskCreateResponse)
    async def detect_video(
        file: UploadFile = File(...),
        skip_frames: int = Form(default=settings.video_default_skip_frames),
        db: Session = Depends(get_db),
    ) -> dict:
        validate_extension(file.filename or "", VIDEO_EXTENSIONS)
        input_dir = settings.uploads_dir / "videos"
        input_dir.mkdir(parents=True, exist_ok=True)

        task_id = uuid.uuid4().hex
        safe_skip_frames = max(skip_frames, 1)
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        input_filename = f"{task_id}{suffix}"
        input_path = input_dir / input_filename

        with input_path.open("wb") as output_file:
            output_file.write(await file.read())

        record_service.upsert_video_task(
            db,
            task_id=task_id,
            input_filename=file.filename or input_filename,
            input_path=input_path.as_posix(),
            status="pending",
            message="浠诲姟宸叉彁浜わ紝绛夊緟澶勭悊",
        )

        dispatch_message = "Task queued for background processing"
        if has_celery_worker():
            try:
                process_video_task.apply_async(
                    kwargs={
                        "input_path": input_path.as_posix(),
                        "skip_frames": safe_skip_frames,
                    },
                    task_id=task_id,
                )
            except Exception:
                start_local_video_task(task_id=task_id, input_path=input_path, skip_frames=safe_skip_frames)
                dispatch_message = "Celery 鍒嗗彂澶辫触锛屽凡鍒囨崲鏈湴绾跨▼澶勭悊"
                record_service.update_video_task(db, task_id, message=dispatch_message)
        else:
            start_local_video_task(task_id=task_id, input_path=input_path, skip_frames=safe_skip_frames)
            dispatch_message = "鏈娴嬪埌 Celery worker锛屽凡鍒囨崲鏈湴绾跨▼澶勭悊"
            record_service.update_video_task(db, task_id, message=dispatch_message)

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": dispatch_message,
        }

    @router.get("/tasks/{task_id}", response_model=VideoTaskStatusResponse)
    async def get_task_status(task_id: str, db: Session = Depends(get_db)) -> dict:
        record = record_service.get_video_task(db, task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Task not found")

        try:
            runtime_state = json.loads(record.runtime_state or "{}")
            if not isinstance(runtime_state, dict):
                runtime_state = {}
        except Exception:
            runtime_state = {}

        payload = {
            "success": True,
            "task_id": task_id,
            "status": record.status,
            "progress": record.progress,
            "message": record.message,
            "result_video": None,
            "stats": None,
            "active_alerts": runtime_state.get("active_alerts", []),
            "active_alert_count": int(runtime_state.get("active_alert_count", 0) or 0),
            "highest_priority_alert": runtime_state.get("highest_priority_alert"),
            "new_alert_count": int(runtime_state.get("new_alert_count", 0) or 0),
            "sustained_alert_count": int(runtime_state.get("sustained_alert_count", 0) or 0),
            "ended_alert_count": int(runtime_state.get("ended_alert_count", 0) or 0),
        }
        if record.output_path:
            try:
                payload["result_video"] = Path(record.output_path).relative_to(settings.uploads_dir).as_posix()
            except ValueError:
                payload["result_video"] = Path(record.output_path).name
        if record.status == "completed":
            video_info = record.video_info or ""
            suppressed_alerts = 0
            m = re.search(r"suppressed=(\d+)", video_info)
            if m:
                suppressed_alerts = int(m.group(1))
            alert_types = record_service.get_video_alert_types(db, task_id)
            payload["stats"] = {
                "total_frames": record.total_frames,
                "detected_frames": record.detected_frames,
                "total_detections": record.total_detections,
                "total_alerts": record.total_alerts,
                "suppressed_alerts": suppressed_alerts,
                "alert_types": alert_types,
                "video_info": video_info,
                "active_alerts": payload["active_alerts"],
                "active_alert_count": payload["active_alert_count"],
                "highest_priority_alert": payload["highest_priority_alert"],
                "new_alert_count": payload["new_alert_count"],
                "sustained_alert_count": payload["sustained_alert_count"],
                "ended_alert_count": payload["ended_alert_count"],
            }
        return payload
    @router.get("/alerts", response_model=AlertListResponse)
    async def get_alerts(
        page: int = 1,
        per_page: int = 20,
        status: str = "all",
        db: Session = Depends(get_db),
    ) -> dict:
        total, records = record_service.list_alerts(db, page=page, per_page=per_page, status=status)
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "records": [
                {
                    "id": record.record_uid,
                    "time": record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": record.status,
                    "types": record.alert_types,
                    "total": record.total_detections,
                    "alert_count": record.alert_count,
                    "source": record.source,
                }
                for record in records
            ],
        }

    @router.get("/alerts/{record_uid}/image", response_model=AlertImageResponse)
    async def get_alert_image(record_uid: str, db: Session = Depends(get_db)) -> dict:
        image_b64 = record_service.get_alert_image_base64(db, record_uid)
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Record not found")
        return {"image": image_b64}

    @router.get("/alerts/{record_uid}/detail")
    async def get_alert_detail(record_uid: str, db: Session = Depends(get_db)) -> dict:
        payload = record_service.get_alert_detail(db, record_uid)
        if payload is None:
            raise HTTPException(status_code=404, detail="Record not found")
        return payload

    @router.get("/statistics", response_model=StatisticsResponse)
    async def get_statistics(db: Session = Depends(get_db)) -> dict:
        return record_service.build_statistics(db, started_at=started_at)

    @router.get("/classes")
    async def get_classes() -> dict:
        return record_service.list_classes()

    @router.get("/status", response_model=SystemStatusResponse)
    async def get_status(detection_service: DetectionService = Depends(get_detection_service)) -> dict:
        models_loaded = detection_service.models_loaded
        detector_loaded = any(
            [
                models_loaded.get("garbage", False),
                models_loaded.get("fire", False),
                models_loaded.get("smoke", False),
            ],
        )
        return {
            "model_loaded": detector_loaded,
            "garbage_model": models_loaded.get("garbage", False),
            "fire_model": models_loaded.get("fire", False),
            "smoke_model": models_loaded.get("smoke", False),
            "bin_color_model": models_loaded.get("bin_color", False),
            "mode": "正常检测" if detector_loaded else "演示模式",
            "uptime": started_at,
            "class_count": 5,
            "version": settings.app_version,
            "name": settings.app_name,
        }

    return router







