import yaml
import datetime
from dataclasses import dataclass
from typing import Optional, List
from src.eyes.detector import TrackedDetection


@dataclass
class AudioEvent:
    event_type: str
    confidence: float
    timestamp: datetime.datetime
    camera_id: str
    location_zone: str


@dataclass
class EnvironmentalReading:
    temperature: float
    pressure: float
    humidity: float
    timestamp: datetime.datetime
    location_zone: str


@dataclass
class FusedAssessment:
    individual_id: str
    species: str
    camera_id: str
    location_zone: str
    timestamp: datetime.datetime
    behavior_label: str
    behavior_confidence: float
    anomaly_score: float
    audio_event: Optional[str]
    environmental_summary: Optional[str]
    alert_level: str
    alert_message: str
    requires_ranger_attention: bool


class SensorFusion:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.anomaly_alert_threshold = self.config['fusion']['anomaly_alert_threshold']
        self.high_priority_threshold = self.config['fusion']['high_priority_threshold']
        self.audio_weight = self.config['fusion']['audio_weight']
        self.visual_weight = self.config['fusion']['visual_weight']
        self.environmental_weight = self.config['fusion']['environmental_weight']

    def fuse(self,
             detection: TrackedDetection,
             behavior_label: str,
             behavior_confidence: float,
             anomaly_score: float,
             audio_event: Optional[AudioEvent] = None,
             environmental: Optional[EnvironmentalReading] = None) -> FusedAssessment:

        fused_score = anomaly_score * self.visual_weight

        if audio_event is not None:
            if audio_event.event_type in ["gunshot", "vehicle"]:
                fused_score += audio_event.confidence * self.audio_weight

        if environmental is not None:
            if environmental.temperature > 40.0:
                fused_score += 0.1 * self.environmental_weight

        if fused_score >= self.high_priority_threshold:
            alert_level = "HIGH"
            requires_attention = True
            alert_message = (
                f"HIGH ALERT: {detection.species} {detection.individual_id} — "
                f"anomaly score {anomaly_score:.2f}, behavior: {behavior_label}"
            )
        elif fused_score >= self.anomaly_alert_threshold:
            alert_level = "MEDIUM"
            requires_attention = True
            alert_message = (
                f"MEDIUM ALERT: {detection.species} {detection.individual_id} — "
                f"unusual behavior detected: {behavior_label}"
            )
        else:
            alert_level = "NORMAL"
            requires_attention = False
            alert_message = (
                f"NORMAL: {detection.species} {detection.individual_id} — "
                f"{behavior_label}"
            )

        audio_summary = None
        if audio_event is not None:
            audio_summary = f"{audio_event.event_type} detected at {audio_event.location_zone}"

        env_summary = None
        if environmental is not None:
            env_summary = (
                f"Temp: {environmental.temperature}C, "
                f"Pressure: {environmental.pressure}hPa, "
                f"Humidity: {environmental.humidity}%"
            )

        return FusedAssessment(
            individual_id=detection.individual_id,
            species=detection.species or "unknown",
            camera_id=detection.camera_id,
            location_zone=detection.camera_id,
            timestamp=detection.timestamp,
            behavior_label=behavior_label,
            behavior_confidence=behavior_confidence,
            anomaly_score=anomaly_score,
            audio_event=audio_summary,
            environmental_summary=env_summary,
            alert_level=alert_level,
            alert_message=alert_message,
            requires_ranger_attention=requires_attention
        )