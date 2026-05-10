from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import numpy as np
import torch
import cv2
from src.eyes.detector import TrackedDetection
import yaml

class BehaviorClassifier:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.processor = VideoMAEImageProcessor.from_pretrained(self.config['model']['version'])
        self.model = VideoMAEForVideoClassification.from_pretrained(self.config['model']['version'])
        self.model.to("cuda")

        self.confidence_threshold = self.config['behavior']['confidence_threshold']
        self.num_frames = self.config['behavior']['num_frames']

    def classify(self, frames, detection):
        x1, y1, x2, y2 = map(int, detection.bbox)

        clips = []
        for frame in frames:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            clips.append(crop)

        inputs = self.processor(clips, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_prob, top_idx = probs[0].topk(1)

        confidence = top_prob.item()
        label = self.model.config.id2label[top_idx.item()]

        if confidence >= self.confidence_threshold:
            return label, confidence

        return "unknown", confidence