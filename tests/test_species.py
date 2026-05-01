import cv2
import pytest
from src.eyes.detector import Detector, Detection
from src.eyes.species import SpeciesClassifier

SERENGETI_IMAGE = "/media/umugabekazi/Extreme SSD/serengeti/s1/B04/B04_R1/S1_B04_R1_PICT0003.JPG"
DETECTOR_CONFIG = "configs/detector.yaml"
SPECIES_CONFIG = "configs/species.yaml"

@pytest.fixture
def detector():
    return Detector(config_path=DETECTOR_CONFIG)

@pytest.fixture
def classifier():
    return SpeciesClassifier(config_path=SPECIES_CONFIG)

def test_import():
    from src.eyes.species import SpeciesClassifier
    assert SpeciesClassifier is not None

def test_classifier_loads(classifier):
    assert classifier.model is not None
    assert classifier.confidence_threshold is not None

def test_classify_returns_detection(detector, classifier):
    frame = cv2.imread(SERENGETI_IMAGE)
    detections = detector.detect(frame)
    assert len(detections) > 0
    result = classifier.classify(detections[0], frame)
    assert isinstance(result, Detection)

def test_species_field_is_string_or_none(detector, classifier):
    frame = cv2.imread(SERENGETI_IMAGE)
    detections = detector.detect(frame)
    result = classifier.classify(detections[0], frame)
    assert result.species is None or isinstance(result.species, str)