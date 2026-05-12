import pytest
import numpy as np
import datetime
from src.brain.features import extract_features, register_unknown_behavior, BEHAVIOR_MAP
from src.eyes.detector import TrackedDetection

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
        track_id=1,
        individual_id="LION_001"
    )

@pytest.fixture
def frame_shape():
    return (1536, 2048, 3)

def test_import():
    from src.brain.features import extract_features, register_unknown_behavior
    assert extract_features is not None

def test_returns_6_features(detection, frame_shape):
    features = extract_features(detection, "walking", 0.9, frame_shape)
    assert len(features) == 6

def test_known_behavior_is_unknown_zero(detection, frame_shape):
    features = extract_features(detection, "walking", 0.9, frame_shape)
    assert features[5] == 0.0

def test_unknown_behavior_is_unknown_one(detection, frame_shape):
    features = extract_features(detection, "somethingweird", 0.5, frame_shape)
    assert features[5] == 1.0

def test_hour_normalized(detection, frame_shape):
    features = extract_features(detection, "walking", 0.9, frame_shape)
    assert 0.0 <= features[2] <= 1.0

def test_speed_zero_without_prev(detection, frame_shape):
    features = extract_features(detection, "walking", 0.9, frame_shape)
    assert features[4] == 0.0

def test_speed_nonzero_with_prev(detection, frame_shape):
    prev = TrackedDetection(
        bbox=[50.0, 100.0, 250.0, 350.0],
        confidence=0.85,
        category="animal",
        timestamp=datetime.datetime(2026, 5, 1, 5, 0, 0),
        camera_id="CAM_001",
        source="video_file",
        model_version="MDV6-yolov9-c",
        track_id=1,
        individual_id="LION_001"
    )
    features = extract_features(detection, "walking", 0.9, frame_shape, prev_detection=prev)
    assert features[4] > 0.0

def test_register_unknown_behavior(detection, frame_shape):
    result = register_unknown_behavior("scratching_tree", "Ranger Jean", "LION_001")
    assert result["registered"] == True
    assert "scratching_tree" in BEHAVIOR_MAP

def test_register_existing_behavior(detection, frame_shape):
    result = register_unknown_behavior("walking", "Ranger Jean", "LION_001")
    assert result["registered"] == False