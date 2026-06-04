from __future__ import annotations

from sqlalchemy import inspect, text

from app.config import get_settings
from app.database import Base, engine
from app.utils import ensure_dir


def _ensure_video_task_runtime_state_column() -> None:
    inspector = inspect(engine)
    try:
        columns = {col["name"] for col in inspector.get_columns("video_task_records")}
    except Exception:
        return
    if "runtime_state" in columns:
        return
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE video_task_records ADD COLUMN runtime_state TEXT DEFAULT '{}'"))


def bootstrap_application() -> None:
    settings = get_settings()
    ensure_dir(settings.uploads_dir)
    ensure_dir(settings.uploads_dir / "alerts")
    ensure_dir(settings.uploads_dir / "videos")
    Base.metadata.create_all(bind=engine)
    _ensure_video_task_runtime_state_column()
