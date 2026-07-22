import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import io
import logging
import base64
import tempfile
from typing import Tuple, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths mapping
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATHS = {
    "resnet50": os.path.join(BASE_DIR, "models/classification/transfer_resnet50_final.h5"),
    "cnn": os.path.join(BASE_DIR, "models/classification/custom_cnn_best.h5")
}

YOLO_PATH = os.path.join(BASE_DIR, "models/detection/aerial_detection/weights/best.pt")
YOLO_FALLBACK_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

class MLService:
    def __init__(self):
        self.classification_model = None
        self.detection_model = None
        self.current_cls_model_type = None
        self.device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
        logger.info(f"Using device: {self.device}")
        
    def load_models(self, cls_model_type: str = "resnet50"):
        # Load classification model
        path = MODEL_PATHS.get(cls_model_type, MODEL_PATHS["resnet50"])
        if os.path.exists(path):
            logger.info(f"Loading Classification model from {path}")
            self.classification_model = tf.keras.models.load_model(path, compile=False)
            self.current_cls_model_type = cls_model_type
        else:
            logger.warning(f"Classification model not found at {path}")

        # Load YOLO model
        if os.path.exists(YOLO_PATH):
            logger.info(f"Loading custom YOLOv8 from {YOLO_PATH}")
            self.detection_model = YOLO(YOLO_PATH)
        else:
            logger.warning(f"Custom YOLOv8 not found at {YOLO_PATH}. Attempting fallback.")
            if os.path.exists(YOLO_FALLBACK_PATH):
                self.detection_model = YOLO(YOLO_FALLBACK_PATH)
            else:
                self.detection_model = YOLO("yolov8n.pt") # Will download if not exists

    def preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image.resize(target_size)) / 255.0
        return np.expand_dims(img_array, axis=0)

    def analyze_image(self, image_bytes: bytes, task: str = "both", cls_model_type: str = "resnet50", conf_threshold: float = 0.5) -> Dict[str, Any]:
        """
        task: 'classification', 'detection', 'both'
        """
        if self.classification_model is None or self.detection_model is None or self.current_cls_model_type != cls_model_type:
            self.load_models(cls_model_type)

        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        results = {}

        # Classification
        if task in ["classification", "both"] and self.classification_model:
            processed = self.preprocess_image(image)
            prediction = self.classification_model.predict(processed, verbose=0)[0][0]
            label = "DRONE" if prediction > 0.5 else "BIRD"
            confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
            results["classification"] = {
                "label": label,
                "confidence": confidence
            }

        # Detection
        if task in ["detection", "both"] and self.detection_model:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                image.save(temp_path, quality=95)

            try:
                # Use tracking if you want to identify unique objects over time (good for video, works on images too)
                det_results = self.detection_model.predict(
                    source=temp_path,
                    save=False,
                    conf=conf_threshold,
                    verbose=False,
                    device=0 if self.device == 'cuda' else 'cpu'
                )
                
                if det_results and len(det_results) > 0:
                    result = det_results[0]
                    num_detections = len(result.boxes)
                    
                    # Create annotated image
                    annotated = result.plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    # Convert to base64
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    results["detection"] = {
                        "count": num_detections,
                        "image_base64": img_base64
                    }
                else:
                    results["detection"] = {
                        "count": 0,
                        "image_base64": None
                    }
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        return results

# Singleton pattern for service
ml_service = MLService()
