from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from celery.utils.log import get_task_logger

from app.bootstrap import bootstrap_application
from app.celery_app import celery_app
from app.config import get_settings
from app.database import SessionLocal
from app.services.detection_service import DetectionService
from app.services.record_service import RecordService
from app.services.video_service import VideoProcessingService


logger = get_task_logger(__name__)
settings = get_settings()


def run_video_task(
    task_id: str,
    input_path: str,
    skip_frames: int = 1,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    bootstrap_application()
    db = SessionLocal()
    record_service = RecordService(settings.uploads_dir)
    detection_service = DetectionService(settings=settings)
    video_service = VideoProcessingService(detection_service=detection_service)

    input_file = Path(input_path)
    output_dir = settings.uploads_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file.stem}_detected.mp4"

    def update_progress(current_frame: int, total_frames: int) -> None:
        progress = int((current_frame / max(total_frames, 1)) * 100)
        record_service.update_video_task(
            db,
            task_id,
            status="processing",
            progress=progress,
            message=f"视频处理中 {progress}%",
        )
        if progress_callback is not None:
            progress_callback(progress)

    def update_runtime_state(runtime_state: dict) -> None:
        processed_frames = int(runtime_state.get("processed_frames", 0) or 0)
        total_frames = max(int(runtime_state.get("total_frames", 1) or 1), 1)
        progress = int((processed_frames / total_frames) * 100)
        active_alert_count = int(runtime_state.get("active_alert_count", 0) or 0)
        highest_priority_alert = runtime_state.get("highest_priority_alert")
        sustained_alert_count = int(runtime_state.get("sustained_alert_count", 0) or 0)
        ended_alert_count = int(runtime_state.get("ended_alert_count", 0) or 0)

        message = f"视频处理中 {progress}%"
        if active_alert_count > 0 and highest_priority_alert:
            message = f"视频处理中 {progress}% · 当前报警 {active_alert_count} 项 · 最高优先级 {highest_priority_alert}"

        record_service.update_video_task(
            db,
            task_id,
            message=message,
            runtime_state={
                "active_alerts": runtime_state.get("active_alerts", []),
                "active_alert_count": active_alert_count,
                "highest_priority_alert": highest_priority_alert,
                "new_alert_count": int(runtime_state.get("new_alert_count", 0) or 0),
                "sustained_alert_count": sustained_alert_count,
                "ended_alert_count": ended_alert_count,
            },
        )

    try:
        record_service.update_video_task(
            db,
            task_id,
            status="processing",
            progress=5,
            message="视频任务开始处理",
            runtime_state={},
        )

        stats = video_service.process_video(
            input_path=input_file,
            output_path=output_path,
            skip_frames=max(skip_frames, 1),
            progress_callback=update_progress,
            status_callback=update_runtime_state,
        )

        record_service.create_video_alert_summary_record(db, task_id=task_id, stats=stats)
        record_service.update_video_task(
            db,
            task_id,
            status="completed",
            progress=100,
            message="视频处理完成",
            output_path=output_path.as_posix(),
            total_frames=stats["total_frames"],
            detected_frames=stats["detected_frames"],
            total_detections=stats["total_detections"],
            total_alerts=stats["total_alerts"],
            video_info=stats["video_info"],
            runtime_state={
                "active_alerts": stats.get("active_alerts", []),
                "active_alert_count": int(stats.get("active_alert_count", 0) or 0),
                "highest_priority_alert": stats.get("highest_priority_alert"),
                "new_alert_count": int(stats.get("new_alert_count", 0) or 0),
                "sustained_alert_count": int(stats.get("sustained_alert_count", 0) or 0),
                "ended_alert_count": int(stats.get("ended_alert_count", 0) or 0),
            },
            error_detail=None,
        )

        if input_file.exists():
            os.remove(input_file)

        return {"task_id": task_id, "status": "completed", "result_video": output_path.name, "stats": stats}
    except Exception as exc:
        logger.exception("video task failed: %s", task_id)
        record_service.update_video_task(
            db,
            task_id,
            status="failed",
            progress=0,
            message="视频处理失败",
            runtime_state={},
            error_detail=str(exc),
        )
        raise
    finally:
        db.close()


@celery_app.task(name="app.tasks.process_video_task", bind=True)
def process_video_task(self, input_path: str, skip_frames: int = 1) -> dict:
    task_id = self.request.id
    return run_video_task(
        task_id=task_id,
        input_path=input_path,
        skip_frames=skip_frames,
        progress_callback=lambda progress: self.update_state(state="PROGRESS", meta={"progress": progress}),
    )
