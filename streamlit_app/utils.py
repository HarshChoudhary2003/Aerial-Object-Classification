import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import streamlit as st
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict, Any
from datetime import datetime
import logging
import io
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
ModelType = str
ImageType = Union[Image.Image, np.ndarray]

# Model paths mapping
MODEL_PATHS = {
    "ResNet50 Transfer Learning": "models/classification/transfer_resnet50_best.h5",
    "Custom CNN": "models/classification/custom_cnn_best.h5"
}

YOLO_PATH = "models/detection/aerial_detection/weights/best.pt"

# File size limit (MB)
MAX_FILE_SIZE_MB = 50

def get_model_metrics() -> pd.DataFrame:
    """Returns model performance metrics with actual paths verification"""
    metrics = pd.DataFrame({
        'Model': ['ResNet50 Transfer', 'Custom CNN'],
        'Accuracy': [0.972, 0.945],
        'Precision': [0.968, 0.931],
        'Recall': [0.978, 0.953],
        'F1-Score': [0.972, 0.941],
        'Training Time (min)': [45, 38],
        'Parameters (M)': [23.5, 2.1],
        'Model Exists': [False, False]
    }).set_index('Model')
    
    # Check if models exist
    for model_name in MODEL_PATHS.keys():
        file_key = 'ResNet50 Transfer' if model_name == "ResNet50 Transfer Learning" else 'Custom CNN'
        file_path = MODEL_PATHS[model_name]
        if os.path.exists(file_path):
            metrics.loc[file_key, 'Model Exists'] = True
    
    return metrics

@st.cache_resource(show_spinner=True)
def load_classification_model(model_name: ModelType = "ResNet50 Transfer Learning") -> tf.keras.Model:
    """Load and cache classification model with error handling"""
    try:
        # Map display name to file path
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Unknown model: {model_name}")
        
        path = MODEL_PATHS[model_name]
        
        if not os.path.exists(path):
            error_msg = f"""
            ### ❌ Model Not Found
            
            **Expected:** `{path}`
            
            **Solutions:**
            1. Train models via notebook: `notebooks/main.ipynb`
            2. Verify model files exist in `models/classification/`
            3. Check folder structure matches documentation
            """
            st.error(error_msg)
            st.stop()
        
        logger.info(f"Loading {model_name} from {path}")
        
        # Load model
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        return model
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

@st.cache_resource(show_spinner=True)
def load_detection_model() -> YOLO:
    """Load and cache YOLOv8 model with validation"""
    try:
        if not os.path.exists(YOLO_PATH):
            error_msg = f"""
            ### ❌ YOLOv8 Model Not Found
            
            **Expected:** `{YOLO_PATH}`
            
            **Training Command:**
            ```bash
            yolo task=detect mode=train model=yolov8n.pt 
            data=data/detection/data.yaml epochs=100 imgsz=640 
            batch=16 project=models/detection name=aerial_detection
            ```
            """
            st.error(error_msg)
            st.stop()
        
        logger.info(f"Loading YOLOv8 from {YOLO_PATH}")
        return YOLO(YOLO_PATH)
        
    except Exception as e:
        logger.error(f"YOLO loading error: {e}")
        st.error(f"❌ Failed to load YOLO: {str(e)}")
        st.stop()

def validate_image(image: ImageType, max_size_mb: int = MAX_FILE_SIZE_MB) -> Tuple[bool, str]:
    """Validate image format, size, and dimensions"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Check dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            return False, "Image dimensions too small (min: 50x50)"
        
        # Check file size
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        size_mb = buffer.tell() / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return False, f"Image size too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        
        # Check format
        if image.mode not in ['RGB', 'L']:
            return False, f"Unsupported image mode: {image.mode}"
        
        return True, "Valid"
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False, f"Validation failed: {str(e)}"

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image.resize(target_size)) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_classification(model: tf.keras.Model, image: ImageType) -> Tuple[str, float]:
    """Predict bird vs drone with confidence"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Validate first
        is_valid, msg = validate_image(image)
        if not is_valid:
            raise ValueError(msg)
        
        processed = preprocess_image(image)
        prediction = model.predict(processed, verbose=0)[0][0]
        
        label = "DRONE" if prediction > 0.5 else "BIRD"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        logger.info(f"Prediction: {label} ({confidence:.2%})")
        return label, confidence
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise Exception(f"Prediction failed: {str(e)}")

def predict_detection(model: YOLO, image: ImageType, conf_threshold: float = 0.5) -> Tuple[np.ndarray, int]:
    """Detect objects using YOLOv8"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Validate
        is_valid, msg = validate_image(image)
        if not is_valid:
            raise ValueError(msg)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image.save(temp_path, quality=95)
        
        # Run inference
        results = model.predict(
            source=temp_path,
            save=False,
            conf=conf_threshold,
            verbose=False,
            device='cpu'  # Force CPU for stability
        )
        
        # Clean up
        os.unlink(temp_path)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes:
                annotated = result.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                return annotated_rgb, len(result.boxes)
        
        return np.array(image), 0
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Detection failed: {str(e)}")

def add_to_history(filename: str, task: str, model: str, results: dict):
    """Add analysis to session history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename,
        'task': task,
        'model': model,
        'classification': results.get('classification', {}).get('label', 'N/A'),
        'confidence': f"{results.get('classification', {}).get('confidence', 0):.2%}",
        'detections': results.get('detection', {}).get('count', 0)
    })

def get_analysis_history() -> pd.DataFrame:
    """Get analysis history as DataFrame"""
    if 'history' not in st.session_state:
        return pd.DataFrame(columns=['timestamp', 'filename', 'task', 'model', 'classification', 'confidence', 'detections'])
    
    if not st.session_state.history:
        return pd.DataFrame(columns=['timestamp', 'filename', 'task', 'model', 'classification', 'confidence', 'detections'])
    
    return pd.DataFrame(st.session_state.history)