import numpy as np
from src.eyes.detector import TrackedDetection

BEHAVIOR_MAP = {
    "walking": 0,
    "running": 1,
    "swimming": 2,
    "climbing": 3,
    "jumping": 4,
    "resting": 5,
    "sleeping": 6,
    "lying": 7,
    "grazing": 8,
    "browsing": 9,
    "hunting": 10,
    "feeding": 11,
    "chasing": 12,
    "drinking": 13,
    "fighting": 14,
    "grooming": 15,
    "playing": 16,
    "mating": 17,
    "nursing": 18,
    "alert": 19,
    "fleeing": 20,
    "vocalizing": 21,
    "charging": 22,
    "unknown": 23
}

def extract_features(detection, behavior_label, behavior_confidence, frame_shape, prev_detection=None):
    behavior_id = BEHAVIOR_MAP.get(behavior_label, 23) / len(BEHAVIOR_MAP)
    is_unknown = 1.0 if behavior_label not in BEHAVIOR_MAP or behavior_label == "unknown" else 0.0
    confidence = detection.confidence
    hour = detection.timestamp.hour / 23.0

    x1, y1, x2, y2 = detection.bbox
    frame_area = frame_shape[0] * frame_shape[1]
    bbox_size = ((x2 - x1) * (y2 - y1)) / frame_area

    if prev_detection is not None:
        px1, py1, px2, py2 = prev_detection.bbox
        prev_cx = (px1 + px2) / 2
        prev_cy = (py1 + py2) / 2
        curr_cx = (x1 + x2) / 2
        curr_cy = (y1 + y2) / 2
        speed = np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) / frame_shape[1]
    else:
        speed = 0.0

    return np.array([behavior_id, confidence, hour, bbox_size, speed, is_unknown], dtype=np.float32)

def register_unknown_behavior(behavior_label, ranger_name, individual_id):
    if behavior_label not in BEHAVIOR_MAP:
        new_id = max(BEHAVIOR_MAP.values()) + 1
        BEHAVIOR_MAP[behavior_label] = new_id
        return {
            "behavior": behavior_label,
            "id": new_id,
            "named_by": ranger_name,
            "individual_id": individual_id,
            "registered": True
        }
    return {"registered": False, "reason": "behavior already exists"}