from speciesnet.classifier import SpeciesNetClassifier
from PIL import Image 
import cv2
from src.eyes.detector import Detection
import yaml

class SpeciesClassifier:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = SpeciesNetClassifier(
            model_name=self.config['model']['version'],
            device="cuda"
            )

        self.confidence_threshold = self.config['classification']['confidence_threshold']
        self.review_threshold = self.config['classification']['review_threshold']

    def classify(self, detection, frame):
        x1, y1, x2, y2 = map(int, detection.bbox)
        cropped_img = frame[y1:y2, x1:x2]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        preprocessed = self.model.preprocess(pil_image, bboxes=None)

        result = self.model.predict(filepath="", img=preprocessed)

        classifications = result['classifications']
        top_score = classifications['scores'][0]
        top_label = classifications['classes'][0]

        if top_score >= self.confidence_threshold:
            detection.species = top_label.split(';')[-1]

        return detection