from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw

from app.config import Settings
from app.services.detection_service import DetectionService
from app.upgrade import AlarmEngine, DetectionEngine, TrackEngine, UpgradePipeline


class VideoProcessingError(RuntimeError):
    pass


@dataclass
class DetectionTrackState:
    class_id: int
    class_name: str
    track_id: int | None
    bbox: list[int]
    frame_history: deque[bool] = field(default_factory=deque)
    last_seen_frame: int = 0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    absent_streak: int = 0
    active: bool = False
    last_push_ts: float = -9999.0
    suppressed_by_priority: bool = False


@dataclass
class ActiveEvent:
    class_id: int
    class_name: str
    priority: int
    track_id: int | None
    bbox: list[int] | None
    first_triggered_ts: float
    last_triggered_ts: float
    last_seen_ts: float
    state: str = "remembered"
    confirmation_mode: str = ""
    duration_seconds: float = 0.0
    last_push_ts: float = -9999.0
    suppressed_by_priority: bool = False
    event_ended: bool = False


class VideoEventStateManager:
    PRIORITY_SUPPRESS_IOU_THRESHOLD = 0.15

    CLASS_PRIORITIES = {
        3: 0,  # fire
        4: 1,  # smoke
        2: 2,  # garbage
        1: 3,  # overflow
    }

    CLEAR_RULES = {
        3: 3,
        4: 3,
        2: 4,
        1: 4,
    }

    BASE_COOLDOWN_SECONDS = {
        3: 2.0,
        4: 2.0,
        2: 3.0,
        1: 4.0,
    }

    NIGHT_COOLDOWN_SECONDS = {
        3: 2.0,
        4: 2.0,
        2: 4.0,
        1: 5.0,
    }

    ESCALATED_COOLDOWN_SECONDS = {
        3: 1.0,
        4: 1.0,
    }

    ESCALATION_DURATION_SECONDS = 10.0
    NIGHT_START_HOUR = 21
    NIGHT_END_HOUR = 6

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.track_states: dict[tuple[int, int], DetectionTrackState] = {}
        self.active_events: list[ActiveEvent] = []
        self.frame_index = 0

    @staticmethod
    def _compute_iou(box1: list[int], box2: list[int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _priority_for_class(self, class_id: int) -> int:
        return self.CLASS_PRIORITIES.get(class_id, 99)

    def _confirm_rule(self, class_id: int) -> dict:
        if class_id in (3, 4):
            return {"window": 2, "required": 1, "mode": "2-frame / 1-hit"}
        if class_id in (1, 2):
            return {"window": 3, "required": 3, "mode": "3-frame continuous"}
        return {"window": 1, "required": 1, "mode": "single-frame"}

    def _clear_frames_for_class(self, class_id: int) -> int:
        return self.CLEAR_RULES.get(class_id, 3)

    def _track_iou_threshold(self, class_id: int) -> float:
        if class_id == 3:
            return float(self.settings.fire_track_iou_threshold)
        if class_id == 4:
            return float(self.settings.smoke_track_iou_threshold)
        if class_id == 2:
            return float(self.settings.garbage_track_iou_threshold)
        if class_id == 1:
            return float(self.settings.overflow_track_iou_threshold)
        return float(self.settings.video_track_iou_default)

    def _bbox_smooth_alpha(self, class_id: int) -> float:
        if class_id == 3:
            alpha = float(self.settings.fire_bbox_smooth_alpha)
        elif class_id == 4:
            alpha = float(self.settings.smoke_bbox_smooth_alpha)
        elif class_id == 2:
            alpha = float(self.settings.garbage_bbox_smooth_alpha)
        elif class_id == 1:
            alpha = float(self.settings.overflow_bbox_smooth_alpha)
        else:
            alpha = float(self.settings.video_bbox_smooth_alpha_default)
        return min(max(alpha, 0.0), 0.95)

    @staticmethod
    def _smooth_bbox(previous_bbox: list[int] | None, current_bbox: list[int], alpha: float) -> list[int]:
        if previous_bbox is None or len(previous_bbox) != 4 or len(current_bbox) != 4:
            return list(current_bbox)

        smoothed = [
            int(round((alpha * float(prev)) + ((1.0 - alpha) * float(curr))))
            for prev, curr in zip(previous_bbox, current_bbox)
        ]
        x1, y1, x2, y2 = smoothed
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def _is_night_time(self, current_ts: float) -> bool:
        hour = int(current_ts // 3600) % 24
        return hour >= self.NIGHT_START_HOUR or hour < self.NIGHT_END_HOUR

    def _cooldown_seconds_for_class(self, class_id: int, current_ts: float, duration_seconds: float = 0.0) -> float:
        if class_id in self.ESCALATED_COOLDOWN_SECONDS and duration_seconds >= self.ESCALATION_DURATION_SECONDS:
            return self.ESCALATED_COOLDOWN_SECONDS[class_id]
        if self._is_night_time(current_ts):
            return self.NIGHT_COOLDOWN_SECONDS.get(class_id, self.BASE_COOLDOWN_SECONDS.get(class_id, 0.0))
        return self.BASE_COOLDOWN_SECONDS.get(class_id, 0.0)

    def _find_track_key(self, detection: dict) -> tuple[int, int] | None:
        track_id = detection.get("track_id")
        class_id = int(detection.get("class_id", -1))
        if track_id is None or class_id < 0:
            return None
        return class_id, int(track_id)

    def _match_active_event(self, detection: dict) -> ActiveEvent | None:
        class_id = int(detection.get("class_id", -1))
        track_id = detection.get("track_id")
        bbox = detection.get("bbox", [])
        iou_threshold = self._track_iou_threshold(class_id)

        for event in self.active_events:
            if event.class_id != class_id or event.event_ended:
                continue
            if (track_id is not None) and (event.track_id is not None) and int(track_id) == int(event.track_id):
                return event
            if bbox and event.bbox and self._compute_iou(bbox, event.bbox) >= iou_threshold:
                return event
        return None

    def _upsert_track_state(self, detection: dict, current_ts: float) -> DetectionTrackState | None:
        key = self._find_track_key(detection)
        if key is None:
            return None

        state = self.track_states.get(key)
        rule = self._confirm_rule(key[0])
        raw_bbox = list(detection.get("bbox", []))
        if state is None:
            state = DetectionTrackState(
                class_id=key[0],
                class_name=str(detection.get("class_name", "unknown")),
                track_id=key[1],
                bbox=list(raw_bbox),
                frame_history=deque(maxlen=int(rule["window"])),
                last_seen_frame=self.frame_index,
                first_seen_ts=current_ts,
                last_seen_ts=current_ts,
            )
            self.track_states[key] = state
        else:
            state.bbox = self._smooth_bbox(
                previous_bbox=state.bbox,
                current_bbox=raw_bbox,
                alpha=self._bbox_smooth_alpha(state.class_id),
            )

        state.class_name = str(detection.get("class_name", "unknown"))
        state.last_seen_frame = self.frame_index
        state.last_seen_ts = current_ts
        state.absent_streak = 0
        state.frame_history.append(True)

        detection["raw_bbox"] = raw_bbox
        detection["bbox"] = list(state.bbox)
        detection["smoothed_bbox"] = list(state.bbox)
        return state

    def _update_absent_states(self, seen_keys: set[tuple[int, int]]) -> None:
        for key, state in list(self.track_states.items()):
            if key in seen_keys:
                continue
            state.frame_history.append(False)
            state.absent_streak += 1

    def _confirmation_met(self, state: DetectionTrackState) -> bool:
        rule = self._confirm_rule(state.class_id)
        required = int(rule["required"])
        if len(state.frame_history) < required:
            return False
        return sum(1 for item in state.frame_history if item) >= required

    def _activate_event(self, state: DetectionTrackState, current_ts: float) -> ActiveEvent:
        event = self._match_active_event(
            {
                "class_id": state.class_id,
                "class_name": state.class_name,
                "track_id": state.track_id,
                "bbox": state.bbox,
            }
        )
        if event is None:
            event = ActiveEvent(
                class_id=state.class_id,
                class_name=state.class_name,
                priority=self._priority_for_class(state.class_id),
                track_id=state.track_id,
                bbox=list(state.bbox),
                first_triggered_ts=current_ts,
                last_triggered_ts=current_ts,
                last_seen_ts=current_ts,
                state="new",
                confirmation_mode=str(self._confirm_rule(state.class_id)["mode"]),
                duration_seconds=0.0,
            )
            self.active_events.append(event)
        else:
            event.class_name = state.class_name
            event.track_id = state.track_id
            event.bbox = list(state.bbox)
            event.last_seen_ts = current_ts
            event.duration_seconds = max(0.0, current_ts - event.first_triggered_ts)
            event.event_ended = False
        state.active = True
        return event

    def _should_push_event(self, event: ActiveEvent, current_ts: float) -> bool:
        duration = max(0.0, current_ts - event.first_triggered_ts)
        cooldown = self._cooldown_seconds_for_class(event.class_id, current_ts, duration_seconds=duration)
        if event.last_push_ts < -1000:
            return True
        return (current_ts - event.last_push_ts) >= cooldown

    def _suppress_lower_priority_events(self) -> None:
        high_priority_events = [event for event in self.active_events if not event.event_ended and event.class_id in (3, 4)]
        if not high_priority_events:
            for event in self.active_events:
                event.suppressed_by_priority = False
            return

        for event in self.active_events:
            if event.event_ended or event.class_id in (3, 4):
                event.suppressed_by_priority = False
                continue

            event.suppressed_by_priority = any(
                event.bbox
                and high.bbox
                and self._compute_iou(event.bbox, high.bbox) >= self.PRIORITY_SUPPRESS_IOU_THRESHOLD
                for high in high_priority_events
            )

    def _cleanup_ended_events(self) -> None:
        self.active_events = [event for event in self.active_events if not event.event_ended]
        expired_keys = []
        for key, state in self.track_states.items():
            if state.absent_streak >= self._clear_frames_for_class(state.class_id):
                expired_keys.append(key)
        for key in expired_keys:
            self.track_states.pop(key, None)

    def apply(self, detections: list[dict], current_ts: float) -> tuple[list[dict], dict]:
        self.frame_index += 1
        seen_keys: set[tuple[int, int]] = set()

        enriched: list[dict] = []
        for det in detections:
            item = det.copy()
            item["new_alert"] = False
            item["active_memory"] = False
            item["event_state"] = "inactive"
            item["event_suppressed_by_priority"] = False
            enriched.append(item)

        detection_by_key: dict[tuple[int, int], dict] = {}
        for item in enriched:
            if not item.get("alert", False):
                continue
            key = self._find_track_key(item)
            if key is None:
                continue
            seen_keys.add(key)
            detection_by_key[key] = item
            state = self._upsert_track_state(item, current_ts=current_ts)
            if state is None:
                continue
            if self._confirmation_met(state):
                event = self._activate_event(state, current_ts=current_ts)
                event.last_seen_ts = current_ts
                event.duration_seconds = max(0.0, current_ts - event.first_triggered_ts)

        self._update_absent_states(seen_keys)

        for event in self.active_events:
            if event.event_ended:
                continue
            if event.track_id is None:
                continue
            state = self.track_states.get((event.class_id, int(event.track_id)))
            if state is None:
                continue
            event.duration_seconds = max(0.0, current_ts - event.first_triggered_ts)
            if state.absent_streak >= self._clear_frames_for_class(event.class_id):
                event.event_ended = True
                state.active = False

        self._suppress_lower_priority_events()

        new_alert_count = 0
        sustained_alert_count = 0
        ended_alert_count = 0
        suppressed_count = 0

        for event in self.active_events:
            if event.event_ended:
                ended_alert_count += 1
                continue

            event.state = "remembered"
            if event.suppressed_by_priority:
                suppressed_count += 1
                continue

            state_key = (event.class_id, int(event.track_id)) if event.track_id is not None else None
            detection = detection_by_key.get(state_key) if state_key is not None else None

            if self._should_push_event(event, current_ts):
                if detection is not None:
                    detection["new_alert"] = True
                    detection["active_memory"] = True
                    detection["event_state"] = "new"
                    detection["event_suppressed_by_priority"] = False
                event.state = "new"
                event.last_triggered_ts = current_ts
                event.last_push_ts = current_ts
                new_alert_count += 1
            else:
                sustained_alert_count += 1
                if detection is not None:
                    detection["alert"] = False
                    detection["active_memory"] = True
                    detection["event_state"] = "remembered"
                    detection["event_suppressed_by_priority"] = False

        for item in enriched:
            key = self._find_track_key(item)
            if key is None:
                continue
            state = self.track_states.get(key)
            event = self._match_active_event(item)
            if event is None or event.event_ended:
                continue
            item["active_memory"] = not event.suppressed_by_priority
            item["event_suppressed_by_priority"] = event.suppressed_by_priority
            if event.suppressed_by_priority:
                item["alert"] = False
                item["active_memory"] = False
                item["event_state"] = "suppressed"
            elif not item.get("new_alert", False):
                item["event_state"] = "remembered"
            if state is not None:
                state.active = not event.suppressed_by_priority

        active_alerts = self.get_active_alerts()
        summary = {
            "active_alerts": active_alerts,
            "active_alert_count": len(active_alerts),
            "highest_priority_alert": active_alerts[0]["class_name"] if active_alerts else None,
            "new_alert_count": new_alert_count,
            "sustained_alert_count": sustained_alert_count,
            "ended_alert_count": ended_alert_count,
            "suppressed_alerts": suppressed_count,
        }
        return enriched, summary

    def get_active_alerts(self) -> list[dict]:
        active = sorted(
            [
                event for event in self.active_events
                if not event.event_ended and not event.suppressed_by_priority
            ],
            key=lambda item: (item.priority, item.first_triggered_ts),
        )
        return [
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "priority": item.priority,
                "state": item.state,
                "duration_seconds": round(item.duration_seconds, 1),
                "confirmation_mode": item.confirmation_mode,
            }
            for item in active
        ]

    def mark_frame_complete(self) -> None:
        for event in self.active_events:
            if event.state == "new":
                event.state = "remembered"
        self._cleanup_ended_events()


class VideoProcessingService:
    def __init__(self, detection_service: DetectionService):
        self.detection_service = detection_service
        self.upgrade_pipeline = UpgradePipeline(
            detection_engine=DetectionEngine(detection_service),
            track_engine=TrackEngine(),
            alarm_engine=AlarmEngine(min_consecutive_frames=1),
        )
        self.event_state = VideoEventStateManager(settings=detection_service.settings)

    @staticmethod
    def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _attach_upgrade_metadata(
        self,
        detections: list[dict],
        timestamp: float | None = None,
    ) -> tuple[list[dict], int]:
        pipe_result = self.upgrade_pipeline.run_detections(detections, timestamp=timestamp)
        tracks = pipe_result.tracks
        alarms = pipe_result.alarms

        out: list[dict] = []
        for idx, det in enumerate(detections):
            item = det.copy()
            if idx < len(tracks):
                item["track_id"] = int(tracks[idx].track_id)
            out.append(item)
        return out, len(alarms)

    def _render_alert_panel(self, image: np.ndarray, active_alerts: list[dict]) -> np.ndarray:
        if not active_alerts:
            return image

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        font_title = self.detection_service._load_chinese_font(size=22)
        font_line = self.detection_service._load_chinese_font(size=18)

        width, height = pil_img.size
        line_height = 30
        panel_width = min(420, max(280, int(width * 0.42)))
        panel_height = 58 + len(active_alerts) * line_height
        left = max(12, width - panel_width - 18)
        top = max(12, height - panel_height - 18)
        right = width - 18
        bottom = height - 18

        draw.rounded_rectangle([left, top, right, bottom], radius=18, fill=(120, 0, 0, 180))
        draw.text((left + 16, top + 12), "报警事件", fill=(255, 255, 255), font=font_title)

        y = top + 44
        for idx, alert in enumerate(active_alerts, start=1):
            state_text = "新触发" if alert.get("state") == "new" else "持续提示"
            duration = float(alert.get("duration_seconds", 0.0) or 0.0)
            text = f"{idx}. {alert.get('class_name', '未知事件')}  {state_text}  {duration:.1f}s"
            draw.text((left + 16, y), text, fill=(255, 236, 236), font=font_line)
            y += line_height

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_video_frame(self, image: np.ndarray, detections: list[dict], active_alerts: list[dict]) -> np.ndarray:
        visual_detections: list[dict] = []
        for det in detections:
            item = det.copy()
            if item.get("active_memory", False):
                item["alert"] = True
                item["color"] = (0, 0, 255)
            visual_detections.append(item)

        rendered = self.detection_service.draw_boxes(image, visual_detections)
        return self._render_alert_panel(rendered, active_alerts)

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        skip_frames: int,
        progress_callback=None,
        status_callback=None,
    ) -> dict:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise VideoProcessingError("无法读取视频文件")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or fps > 120:
            fps = 30.0

        frame_count = 0
        total_detections = 0
        total_alerts = 0
        total_suppressed_alerts = 0
        total_pipeline_alarms = 0
        alert_frames = 0
        alert_type_set: set[str] = set()
        prev_result = None
        effective_skip = max(skip_frames, 1)
        latest_runtime_state = {
            "active_alerts": [],
            "active_alert_count": 0,
            "highest_priority_alert": None,
            "new_alert_count": 0,
            "sustained_alert_count": 0,
            "ended_alert_count": 0,
        }

        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if (frame_count - 1) % effective_skip != 0:
                    frame_to_write = prev_result if prev_result is not None else frame
                    writer.append_data(self._bgr_to_rgb(frame_to_write))
                    if progress_callback and total_frames:
                        progress_callback(frame_count, total_frames)
                    continue

                detections = self.detection_service.detect(frame)
                current_ts = (frame_count / fps) if fps > 0 else 0.0
                detections, pipeline_alarm_count = self._attach_upgrade_metadata(
                    detections,
                    timestamp=current_ts,
                )
                detections, runtime_state = self.event_state.apply(
                    detections=detections,
                    current_ts=current_ts,
                )

                total_pipeline_alarms += int(pipeline_alarm_count)
                total_suppressed_alerts += int(runtime_state.get("suppressed_alerts", 0))
                latest_runtime_state = {
                    "active_alerts": runtime_state.get("active_alerts", []),
                    "active_alert_count": int(runtime_state.get("active_alert_count", 0) or 0),
                    "highest_priority_alert": runtime_state.get("highest_priority_alert"),
                    "new_alert_count": int(runtime_state.get("new_alert_count", 0) or 0),
                    "sustained_alert_count": int(runtime_state.get("sustained_alert_count", 0) or 0),
                    "ended_alert_count": int(runtime_state.get("ended_alert_count", 0) or 0),
                }

                rendered = self._draw_video_frame(
                    frame,
                    detections=detections,
                    active_alerts=latest_runtime_state["active_alerts"],
                )
                prev_result = rendered.copy()

                frame_alerts = sum(1 for item in detections if item.get("new_alert", False))
                for item in detections:
                    if item.get("active_memory", False):
                        alert_type_set.add(str(item.get("class_name", "unknown")))
                if latest_runtime_state["active_alert_count"] > 0:
                    alert_frames += 1
                total_alerts += frame_alerts
                total_detections += len(detections)

                cv2.putText(
                    rendered,
                    f"Frame {frame_count}: {len(detections)} detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    rendered,
                    f"Upgrade alarms: {total_pipeline_alarms}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 255),
                    2,
                )
                cv2.putText(
                    rendered,
                    f"Suppressed: {total_suppressed_alerts}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (150, 220, 255),
                    2,
                )

                writer.append_data(self._bgr_to_rgb(rendered))

                if status_callback is not None:
                    status_callback(
                        {
                            **latest_runtime_state,
                            "processed_frames": frame_count,
                            "total_frames": total_frames,
                        }
                    )
                if progress_callback and total_frames:
                    progress_callback(frame_count, total_frames)

                self.event_state.mark_frame_complete()
        finally:
            writer.close()
            cap.release()

        final_active_alerts = self.event_state.get_active_alerts()
        return {
            "total_frames": frame_count,
            "detected_frames": alert_frames,
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "suppressed_alerts": total_suppressed_alerts,
            "alert_types": sorted(alert_type_set),
            "video_info": f"{width}x{height}, {fps:.1f}fps, suppressed={total_suppressed_alerts}",
            "active_alerts": final_active_alerts,
            "active_alert_count": len(final_active_alerts),
            "highest_priority_alert": final_active_alerts[0]["class_name"] if final_active_alerts else None,
            "new_alert_count": 0,
            "sustained_alert_count": 0,
            "ended_alert_count": 0,
        }
