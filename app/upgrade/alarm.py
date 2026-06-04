from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlarmEvent:
    track_id: int
    class_id: int
    class_name: str
    level: str
    reason: str


class AlarmEngine:
    """Temporal alarm scaffold.

    Rules you can customize:
    - min_consecutive_frames
    - min_duration_seconds
    - confidence rising trend
    """

    def __init__(self, min_consecutive_frames: int = 3):
        self.min_consecutive_frames = min_consecutive_frames
        self._seen_count: dict[int, int] = {}

    def evaluate(self, tracks) -> list[AlarmEvent]:
        events: list[AlarmEvent] = []
        for tr in tracks:
            c = self._seen_count.get(tr.track_id, 0) + 1
            self._seen_count[tr.track_id] = c
            if c >= self.min_consecutive_frames:
                events.append(
                    AlarmEvent(
                        track_id=tr.track_id,
                        class_id=tr.class_id,
                        class_name=tr.class_name,
                        level="medium",
                        reason=f"consecutive_frames>={self.min_consecutive_frames}",
                    )
                )
        return events
