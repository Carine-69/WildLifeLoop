import cv2
import pytest
from src.eyes.detector import Detector, Detection

SERENGETI_IMAGE = "/media/umugabekazi/Extreme SSD/serengeti/s1/B04/B04_R1/S1_B04_R1_PICT0003.JPG"
CONFIG_PATH = "configs/detector.yaml"

@pytest.fixture
def detector():
    return Detector(config_path=CONFIG_PATH)

def test_import():
    from src.eyes.detector import Detector, Detection, TrackedDetection
    assert Detector is not None

def test_detector_loads(detector):
    assert detector.model is not None
    assert detector.confidence_threshold == 0.2
    assert detector.review_threshold == 0.8

def test_detect_returns_list(detector):
    frame = cv2.imread(SERENGETI_IMAGE)
    detections = detector.detect(frame)
    assert isinstance(detections, list)

def test_detect_returns_detection_objects(detector):
    frame = cv2.imread(SERENGETI_IMAGE)
    detections = detector.detect(frame)
    assert len(detections) > 0
    assert isinstance(detections[0], Detection)

def test_detection_fields(detector):
    frame = cv2.imread(SERENGETI_IMAGE)
    detections = detector.detect(frame)
    d = detections[0]
    assert d.category in ['animal', 'person', 'vehicle']
    assert 0 <= d.confidence <= 1
    assert len(d.bbox) == 4
    assert d.review_status in ['pending', 'auto_approved']
    assert d.camera_id == 'CAM_001'