import pytest
import datetime
from src.brain.fusion import SensorFusion, FusedAssessment, AudioEvent, EnvironmentalReading
from src.eyes.detector import TrackedDetection

FUSION_CONFIG = "configs/fusion.yaml"

@pytest.fixture
def fusion():
    return SensorFusion(config_path=FUSION_CONFIG)

@pytest.fixture
def detection():
    return TrackedDetection(
        bbox=[100.0, 200.0, 300.0, 400.0],
        confidence=0.85,
        category="animal",
        timestamp=datetime.datetime(2026, 5, 1, 6, 0, 0),
        camera_id="CAM_001",
        source="video_file",
        model_version="MDV6-yolov9-c",
        species="lion",
        track_id=1,
        individual_id="LION_001"
    )

def test_import():
    from src.brain.fusion import SensorFusion, FusedAssessment
    assert SensorFusion is not None

def test_fusion_loads(fusion):
    assert fusion.anomaly_alert_threshold == 0.15
    assert fusion.high_priority_threshold == 0.35

def test_normal_assessment(fusion, detection):
    result = fusion.fuse(detection, "walking", 0.9, 0.01)
    assert isinstance(result, FusedAssessment)
    assert result.alert_level == "NORMAL"
    assert result.requires_ranger_attention == False

def test_medium_alert(fusion, detection):
    result = fusion.fuse(detection, "alert", 0.8, 0.4)
    assert result.alert_level == "MEDIUM"
    assert result.requires_ranger_attention == True

def test_high_alert(fusion, detection):
    result = fusion.fuse(detection, "fleeing", 0.9, 0.8)
    assert result.alert_level == "HIGH"
    assert result.requires_ranger_attention == True

def test_audio_event_raises_score(fusion, detection):
    audio = AudioEvent(
        event_type="gunshot",
        confidence=0.95,
        timestamp=datetime.datetime(2026, 5, 1, 6, 0, 0),
        camera_id="CAM_001",
        location_zone="north"
    )
    result = fusion.fuse(detection, "fleeing", 0.9, 0.4, audio_event=audio)
    assert result.alert_level in ["MEDIUM", "HIGH"]
    assert result.audio_event is not None

def test_fused_assessment_fields(fusion, detection):
    result = fusion.fuse(detection, "walking", 0.9, 0.01)
    assert result.individual_id == "LION_001"
    assert result.species == "lion"
    assert result.camera_id == "CAM_001"
    assert result.behavior_label == "walking"