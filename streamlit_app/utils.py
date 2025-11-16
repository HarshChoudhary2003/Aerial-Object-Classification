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
import urllib.request
import gzip

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

# MODEL DOWNLOAD URLs (REPLACE THESE WITH YOUR ACTUAL MODEL URLs)
# You can upload your models to Google Drive, Dropbox, or any direct download service
# and replace the URLs below. The URLs must be direct download links.
MODEL_URLS = {
    "ResNet50 Transfer Learning": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",  # PLACEHOLDER
    "Custom CNN": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",  # PLACEHOLDER
    "YOLOv8": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"  # PLACEHOLDER
}

# File size limit (MB)
MAX_FILE_SIZE_MB = 50

def ensure_models_available():
    """
    Ensure all required models are available locally.
    If not present, download them from the specified URLs.
    This is the key function for easy deployment.
    """
    # Create directories
    os.makedirs("models/classification", exist_ok=True)
    os.makedirs("models/detection/aerial_detection/weights", exist_ok=True)
    
    # Check and download classification models
    for model_name, file_path in MODEL_PATHS.items():
        if not os.path.exists(file_path):
            st.warning(f"🔄 Downloading {model_name}... (one-time setup)")
            url = MODEL_URLS.get(model_name)
            if url:
                try:
                    download_model(url, file_path, model_name)
                    st.success(f"✅ {model_name} downloaded successfully")
                except Exception as e:
                    st.error(f"❌ Failed to download {model_name}: {e}")
                    st.stop()
            else:
                st.error(f"❌ No download URL configured for {model_name}. Please update MODEL_URLS in utils.py")
                st.stop()
    
    # Check and download YOLO model
    if not os.path.exists(YOLO_PATH):
        st.warning("🔄 Downloading YOLOv8 Detection Model... (one-time setup)")
        url = MODEL_URLS.get("YOLOv8")
        if url:
            try:
                download_model(url, YOLO_PATH, "YOLOv8 Detection")
                st.success("✅ YOLOv8 model downloaded successfully")
            except Exception as e:
                st.error(f"❌ Failed to download YOLOv8: {e}")
                st.stop()
        else:
            st.error("❌ No download URL configured for YOLOv8. Please update MODEL_URLS in utils.py")
            st.stop()

def download_model(url: str, destination: str, model_name: str):
    """
    Download a model file with progress bar
    
    Args:
        url: Direct download URL for the model
        destination: Local path to save the model
        model_name: Name of the model for display
    """
    try:
        with st.spinner(f"Downloading {model_name}..."):
            # Download with progress
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                st.progress(percent / 100, f"Downloading... {percent:.1f}%")
            
            # For Streamlit Cloud, we need to handle the download differently
            # Using urllib with a progress hook
            urllib.request.urlretrieve(url, destination, reporthook=download_progress)
            
            # Verify file was downloaded and has content
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                logger.info(f"Successfully downloaded {model_name} to {destination}")
            else:
                raise Exception(f"Downloaded file is empty or missing: {destination}")
                
    except Exception as e:
        logger.error(f"Download error for {model_name}: {e}")
        raise Exception(f"Failed to download {model_name}: {str(e)}")

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
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
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
        
        # Double-check model exists (should be handled by ensure_models_available)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            error_msg = f"""
            ### ❌ Model Not Found: {model_name}
            
            **Expected:** `{path}`
            
            **Solutions:**
            1. The model should have been downloaded automatically. Check the logs above.
            2. Verify the download URL in `MODEL_URLS` is correct and accessible.
            3. Manually upload the model file to the expected path.
            """
            st.error(error_msg)
            logger.error(f"Model file missing or empty: {path}")
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
        # Double-check YOLO model exists
        if not os.path.exists(YOLO_PATH) or os.path.getsize(YOLO_PATH) == 0:
            error_msg = f"""
            ### ❌ YOLOv8 Model Not Found
            
            **Expected:** `{YOLO_PATH}`
            
            **Solutions:**
            1. Wait for the automatic download to complete (check logs above).
            2. Update the `MODEL_URLS['YOLOv8']` in utils.py with a direct download link.
            3. Manually upload the model to the specified path.
            """
            st.error(error_msg)
            logger.error(f"YOLO model file missing or empty: {YOLO_PATH}")
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
            device='cpu'  # Force CPU for stability in cloud deployment
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