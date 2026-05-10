import cv2
import numpy as np
import pytest
from src.brain.behavior import BehaviorClassifier
from src.eyes.detector import TrackedDetection
import datetime

SERENGETI_IMAGE = "/media/umugabekazi/Extreme SSD/serengeti/s1/B04/B04_R1/S1_B04_R1_PICT0003.JPG"
BEHAVIOR_CONFIG = "configs/behavior.yaml"

@pytest.fixture
def classifier():
    return BehaviorClassifier(config_path=BEHAVIOR_CONFIG)

@pytest.fixture
def sample_detection():
    return TrackedDetection(
        bbox=[1127.0, 1351.0, 1437.0, 1435.0],
        confidence=0.20,
        category="animal",
        timestamp=datetime.datetime.now(),
        camera_id="CAM_001",
        source="video_file",
        model_version="MDV6-yolov9-c",
        track_id=1,
        individual_id="animal_1"
    )

@pytest.fixture
def sample_frames():
    base_path = "/media/umugabekazi/Extreme SSD/serengeti/s1/B04/B04_R1"
    frame_files = [
        f"S1_B04_R1_PICT{str(i).zfill(4)}.JPG"
        for i in range(3, 19)  # PICT0003 to PICT0018 = 16 frames
    ]
    frames = []
    for f in frame_files:
        frame = cv2.imread(f"{base_path}/{f}")
        if frame is not None:
            frames.append(frame)
    return frames

def test_import():
    from src.brain.behavior import BehaviorClassifier
    assert BehaviorClassifier is not None

def test_classifier_loads(classifier):
    assert classifier.model is not None
    assert classifier.processor is not None
    assert classifier.num_frames == 16

def test_classify_returns_label_and_confidence(classifier, sample_frames, sample_detection):
    label, confidence = classifier.classify(sample_frames, sample_detection)
    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_classify_returns_unknown_or_label(classifier, sample_frames, sample_detection):
    label, confidence = classifier.classify(sample_frames, sample_detection)
    assert label is not None
    print(f"Behavior: {label}, Confidence: {confidence:.2f}")