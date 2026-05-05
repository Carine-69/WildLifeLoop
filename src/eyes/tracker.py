from deep_sort_realtime.deepsort_tracker import DeepSort
import yaml
import datetime
from src.eyes.detector import Detection, TrackedDetection

class Tracker:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = DeepSort(
            max_age=self.config['tracking']['max_age'],
            n_init=self.config['tracking']['min_hits']
        )

        self.registry = {}

        self.confidence_threshold = self.config['tracking']['confidence_threshold']
        self.max_age = self.config['tracking']['max_age']
        self.min_hits = self.config['tracking']['min_hits']

    def track(self, detection, frame):
        x1, y1, x2, y2 = detection.bbox
        w = x2 - x1
        h = y2 - y1
        raw_detection = [([x1, y1, w, h], detection.confidence, detection.category)]

        tracks = self.model.update_tracks(raw_detection, frame=frame)
        print(f"Total tracks: {len(tracks)}")
        for t in tracks:
            print(f"  track_id={t.track_id}, confirmed={t.is_confirmed()}, state={t.state}")

        tracked_detections = []
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id

                if track_id not in self.registry:
                    self.registry[track_id] = f"{detection.species}_{track_id}"
                individual_id = self.registry[track_id]

                tracked_detections.append(TrackedDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    category=detection.category,
                    timestamp=detection.timestamp,
                    camera_id=detection.camera_id,
                    source=detection.source,
                    model_version=detection.model_version,
                    species=detection.species,
                    review_status=detection.review_status,
                    track_id=track_id,
                    individual_id=individual_id
                ))

        return tracked_detections