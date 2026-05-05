from dataclasses import dataclass
from typing import List, Optional
import yaml
import datetime
from PytorchWildlife.models import detection as pw_detection

CATEGORY_MAP = {0: 'animal', 1: 'person', 2: 'vehicle'}

@dataclass
class Detection:
    bbox: List[float]
    confidence: float
    category: str
    timestamp: datetime.datetime
    camera_id: str
    source: str
    model_version: str
    species: Optional[str] = None
    review_status: str = "pending"

@dataclass
class TrackedDetection(Detection):
    track_id: int = 0
    individual_id: Optional[str] = None

class Detector:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = pw_detection.MegaDetectorV6(
            version="MDV6-yolov9-c",
            device="cuda",
            pretrained=True
        )

        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.camera_id = self.config['source']['camera_id']
        self.source = self.config['source']['type']
        self.model_version = self.config['model']['version']
        self.review_threshold = self.config['detection']['review_threshold']

    def _parse_results(self, results):
        detections = []
        raw = results['detections']
        for i in range(len(raw.xyxy)):
            confidence = float(raw.confidence[i])
            if confidence >= self.confidence_threshold:
                detections.append(Detection(
                    bbox=raw.xyxy[i].tolist(),
                    confidence=confidence,
                    category=CATEGORY_MAP.get(int(raw.class_id[i]), 'unknown'),
                    timestamp=datetime.datetime.now(),
                    camera_id=self.camera_id,
                    source=self.source,
                    model_version=self.model_version,
                    species=None,
                    review_status="auto_approved" if confidence >= self.review_threshold else "pending"
                ))
        return detections

    def detect(self, frame):
        results = self.model.single_image_detection(frame)
        return self._parse_results(results)

    def detect_batch(self, frames):
        all_detections = []
        for frame in frames:
            results = self.model.multi_image_detection(frame)
            all_detections.extend(self._parse_results(results))
        return all_detections