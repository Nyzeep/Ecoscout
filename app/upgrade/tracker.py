from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from time import monotonic


def _compute_iou(box1: list[int], box2: list[int]) -> float:
    x1 = max(int(box1[0]), int(box2[0]))
    y1 = max(int(box1[1]), int(box2[1]))
    x2 = min(int(box1[2]), int(box2[2]))
    y2 = min(int(box1[3]), int(box2[3]))
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, int(box1[2]) - int(box1[0])) * max(0, int(box1[3]) - int(box1[1]))
    area2 = max(0, int(box2[2]) - int(box2[0])) * max(0, int(box2[3]) - int(box2[1]))
    union = area1 + area2 - inter
    return (inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]


@dataclass
class _TrackState:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]
    last_seen_ts: float


class TrackEngine:
    """Greedy IoU tracker with timestamp-based lifetime."""

    def __init__(self, iou_match_threshold: float = 0.3, max_age_seconds: float = 1.2):
        self._id_gen = count(1)
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_age_seconds = float(max_age_seconds)
        self._tracks: dict[int, _TrackState] = {}

    def _drop_expired(self, now_ts: float) -> None:
        expired_ids = [
            tid
            for tid, st in self._tracks.items()
            if (now_ts - float(st.last_seen_ts)) > self.max_age_seconds
        ]
        for tid in expired_ids:
            self._tracks.pop(tid, None)

    def _create_track(self, det, now_ts: float) -> _TrackState:
        tr = _TrackState(
            track_id=next(self._id_gen),
            class_id=int(det.class_id),
            class_name=str(det.class_name),
            confidence=float(det.confidence),
            bbox=[int(v) for v in det.bbox],
            last_seen_ts=float(now_ts),
        )
        self._tracks[tr.track_id] = tr
        return tr

    def update(self, detections, timestamp: float | None = None) -> list[Track]:
        now_ts = float(timestamp) if timestamp is not None else float(monotonic())
        self._drop_expired(now_ts)

        if not detections:
            return []

        track_items = list(self._tracks.items())
        pair_candidates: list[tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for tr_pos, (_, tr) in enumerate(track_items):
                if int(det.class_id) != int(tr.class_id):
                    continue
                iou = _compute_iou(det.bbox, tr.bbox)
                if iou >= self.iou_match_threshold:
                    pair_candidates.append((iou, det_idx, tr_pos))

        pair_candidates.sort(key=lambda x: x[0], reverse=True)

        det_to_track: dict[int, int] = {}
        used_dets: set[int] = set()
        used_track_ids: set[int] = set()

        for _, det_idx, tr_pos in pair_candidates:
            track_id, _ = track_items[tr_pos]
            if det_idx in used_dets or track_id in used_track_ids:
                continue
            det_to_track[det_idx] = int(track_id)
            used_dets.add(det_idx)
            used_track_ids.add(int(track_id))

        out: list[Track] = []
        for det_idx, det in enumerate(detections):
            matched_tid = det_to_track.get(det_idx)
            if matched_tid is None:
                st = self._create_track(det, now_ts)
            else:
                st = self._tracks[matched_tid]
                st.class_id = int(det.class_id)
                st.class_name = str(det.class_name)
                st.confidence = float(det.confidence)
                st.bbox = [int(v) for v in det.bbox]
                st.last_seen_ts = float(now_ts)

            out.append(
                Track(
                    track_id=int(st.track_id),
                    class_id=int(st.class_id),
                    class_name=str(st.class_name),
                    confidence=float(st.confidence),
                    bbox=[int(v) for v in st.bbox],
                )
            )
        return out
