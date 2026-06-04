from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineResult:
    detections: list
    tracks: list
    alarms: list


class UpgradePipeline:
    """Detection -> Tracking -> Temporal Alarm pipeline."""

    def __init__(self, detection_engine, track_engine, alarm_engine):
        self.detection_engine = detection_engine
        self.track_engine = track_engine
        self.alarm_engine = alarm_engine

    def run_frame(self, frame, timestamp: float | None = None) -> PipelineResult:
        detections = self.detection_engine.infer(frame)
        tracks = self.track_engine.update(detections, timestamp=timestamp)
        alarms = self.alarm_engine.evaluate(tracks)
        return PipelineResult(detections=detections, tracks=tracks, alarms=alarms)

    def run_detections(self, raw_detections: list[dict], timestamp: float | None = None) -> PipelineResult:
        detections = self.detection_engine.adapt(raw_detections)
        tracks = self.track_engine.update(detections, timestamp=timestamp)
        alarms = self.alarm_engine.evaluate(tracks)
        return PipelineResult(detections=detections, tracks=tracks, alarms=alarms)
