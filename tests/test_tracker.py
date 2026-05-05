import cv2
import pytest
from src.eyes.detector import Detector, TrackedDetection
from src.eyes.species import SpeciesClassifier
from src.eyes.tracker import Tracker

SERENGETI_IMAGE = "/media/umugabekazi/Extreme SSD/serengeti/s1/B04/B04_R1/S1_B04_R1_PICT0003.JPG"
DETECTOR_CONFIG = "configs/detector.yaml"
SPECIES_CONFIG = "configs/species.yaml"
TRACKER_CONFIG = "configs/tracker.yaml"

@pytest.fixture
def detector():
    return Detector(config_path=DETECTOR_CONFIG)

@pytest.fixture
def classifier():
    return SpeciesClassifier(config_path=SPECIES_CONFIG)

@pytest.fixture
def tracker():
    return Tracker(config_path=TRACKER_CONFIG)

def test_import():
    from src.eyes.tracker import Tracker
    assert Tracker is not None

def test_tracker_loads(tracker):
    assert tracker.model is not None
    assert tracker.registry == {}

def test_track_returns_list(detector, classifier, tracker):
    frame = cv2.imread(SERENGETI_IMAGE)
    detections = detector.detect(frame)
    assert len(detections) > 0
    d = classifier.classify(detections[0], frame)
    result = tracker.track(d, frame)
    assert isinstance(result, list)

def test_tracked_detection_has_id(detector, classifier, tracker):
    frame = cv2.imread(SERENGETI_IMAGE)
    detections = detector.detect(frame)
    d = classifier.classify(detections[0], frame)
    result = tracker.track(d, frame)
    if len(result) > 0:
        assert isinstance(result[0], TrackedDetection)
        assert result[0].track_id is not None
        assert result[0].individual_id is not None