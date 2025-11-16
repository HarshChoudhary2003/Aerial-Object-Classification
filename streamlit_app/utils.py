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
import time

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

# ⚠️ CRITICAL: REPLACE THESE WITH YOUR ACTUAL MODEL URLs
# The placeholders below point to YOLO models for demonstration.
# You MUST upload your actual .h5 and .pt files to a file host.
# Use direct download links (Google Drive: ?export=download, Dropbox: ?dl=1, Hugging Face: raw link)
MODEL_URLS = {
    "ResNet50 Transfer Learning": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",  # ❌ PLACEHOLDER - REPLACE!
    "Custom CNN": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",  # ❌ PLACEHOLDER - REPLACE!
    "YOLOv8": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"  # ✅ This one is OK for YOLO
}

# Expected minimum file sizes in MB (adjust based on your actual models)
EXPECTED_SIZES = {
    "ResNet50 Transfer Learning": 80,
    "Custom CNN": 10,
    "YOLOv8": 20
}

# File size limit (MB)
MAX_FILE_SIZE_MB = 50

def ensure_models_available():
    """
    Ensure all required models are available locally.
    Downloads models if missing or corrupted, with verification and retry logic.
    """
    # Create directories
    os.makedirs("models/classification", exist_ok=True)
    os.makedirs("models/detection/aerial_detection/weights", exist_ok=True)
    
    # Define models to check
    models_to_check = {
        "ResNet50 Transfer Learning": {
            "path": MODEL_PATHS["ResNet50 Transfer Learning"],
            "url": MODEL_URLS.get("ResNet50 Transfer Learning"),
            "min_size_mb": EXPECTED_SIZES["ResNet50 Transfer Learning"]
        },
        "Custom CNN": {
            "path": MODEL_PATHS["Custom CNN"],
            "url": MODEL_URLS.get("Custom CNN"),
            "min_size_mb": EXPECTED_SIZES["Custom CNN"]
        },
        "YOLOv8": {
            "path": YOLO_PATH,
            "url": MODEL_URLS.get("YOLOv8"),
            "min_size_mb": EXPECTED_SIZES["YOLOv8"]
        }
    }
    
    # Check each model
    for model_name, config in models_to_check.items():
        path = config["path"]
        url = config["url"]
        min_size_mb = config["min_size_mb"]
        
        # Check if model exists and is valid
        is_valid = verify_model_file(path, min_size_mb)
        
        if not is_valid:
            if not url or "placeholder" in url.lower():
                st.error(f"❌ **CRITICAL**: No valid URL configured for `{model_name}`")
                st.error(f"Please update `MODEL_URLS` in `utils.py` with a direct download link.")
                st.stop()
            
            # Delete corrupted file if it exists
            if os.path.exists(path):
                st.warning(f"⚠️ Removing corrupted file: {os.path.basename(path)}")
                os.remove(path)
            
            # Download with retry
            st.info(f"⬇️ Downloading {model_name} (~{min_size_mb}MB)...")
            try:
                download_model_with_retry(url, path, model_name, min_size_mb)
                st.success(f"✅ {model_name} ready")
            except Exception as e:
                st.error(f"❌ Failed to download {model_name} after multiple attempts: {e}")
                st.stop()
        else:
            logger.info(f"✅ {model_name} verified at {path}")

def verify_model_file(path: str, min_size_mb: int) -> bool:
    """
    Verify model file exists and meets minimum size requirement.
    Also performs a basic format check.
    """
    if not os.path.exists(path):
        return False
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    
    # Size check
    if size_mb < min_size_mb:
        logger.warning(f"File {path} too small: {size_mb:.1f}MB (min: {min_size_mb}MB)")
        return False
    
    # Format check (basic)
    try:
        with open(path, 'rb') as f:
            header = f.read(8)
            
            # Check for HDF5 format (Keras models)
            if path.endswith('.h5') and not header.startswith(b'\x89HDF\r\n\x1a\n'):
                logger.error(f"File {path} is not a valid HDF5 format")
                return False
            
            # Check for PyTorch format (YOLO models)
            if path.endswith('.pt') and not header.startswith(b'PK'):
                logger.error(f"File {path} is not a valid PyTorch model")
                return False
    except Exception as e:
        logger.error(f"Cannot verify file {path}: {e}")
        return False
    
    return True

def download_model_with_retry(url: str, destination: str, model_name: str, min_size_mb: int, max_retries: int = 3):
    """
    Download model with progress bar, integrity checks, and retry logic.
    """
    for attempt in range(max_retries):
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def download_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min((block_num * block_size * 100) / total_size, 100)
                    progress_bar.progress(percent / 100)
                    status_text.text(f"📥 Downloading {model_name}... {percent:.1f}%")
            
            # Download
            urllib.request.urlretrieve(url, destination, reporthook=download_progress)
            
            # Verify download
            actual_size_mb = os.path.getsize(destination) / (1024 * 1024)
            
            if actual_size_mb < min_size_mb * 0.5:  # File is less than 50% of expected size
                raise Exception(f"File too small: {actual_size_mb:.1f}MB (expected ~{min_size_mb}MB). Likely corrupted or incomplete.")
            
            # Verify file format
            if not verify_model_file(destination, min_size_mb):
                raise Exception("Downloaded file format is invalid")
            
            progress_bar.empty()
            status_text.text(f"✅ {model_name} download complete: {actual_size_mb:.1f} MB")
            time.sleep(1)  # Brief pause for user to see success message
            progress_bar.empty()
            status_text.empty()
            
            return
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            
            if attempt < max_retries - 1:
                st.warning(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:50]}... Retrying in 3s...")
                time.sleep(3)
                if os.path.exists(destination):
                    os.remove(destination)
            else:
                raise Exception(f"Download failed after {max_retries} attempts. Last error: {e}")

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
    
    # Check if models exist and are valid
    for model_name in MODEL_PATHS.keys():
        file_key = 'ResNet50 Transfer' if model_name == "ResNet50 Transfer Learning" else 'Custom CNN'
        file_path = MODEL_PATHS[model_name]
        is_valid = verify_model_file(file_path, EXPECTED_SIZES[model_name])
        metrics.loc[file_key, 'Model Exists'] = is_valid
    
    return metrics

@st.cache_resource(show_spinner=True)
def load_classification_model(model_name: ModelType = "ResNet50 Transfer Learning"):
    """Load and cache classification model with error handling"""
    try:
        path = MODEL_PATHS[model_name]
        
        # Verify model file before loading
        if not verify_model_file(path, EXPECTED_SIZES[model_name]):
            raise ValueError(f"Model file invalid or corrupted: {path}")
        
        logger.info(f"Loading {model_name} from {path}")
        
        # Load model - use compile=False first, then compile manually for stability
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    except Exception as e:
        logger.error(f"Model loading error for {model_name}: {e}")
        st.error(f"❌ Failed to load model `{model_name}`: {str(e)}")
        st.error("**This usually means the file is corrupted.** The app will attempt to re-download on next run.")
        
        # Delete corrupted file so it re-downloads next time
        if os.path.exists(MODEL_PATHS[model_name]):
            os.remove(MODEL_PATHS[model_name])
        
        st.stop()

@st.cache_resource(show_spinner=True)
def load_detection_model():
    """Load and cache YOLOv8 model with validation"""
    try:
        # Verify model file before loading
        if not verify_model_file(YOLO_PATH, EXPECTED_SIZES["YOLOv8"]):
            raise ValueError(f"YOLO model file invalid or corrupted: {YOLO_PATH}")
        
        logger.info(f"Loading YOLOv8 from {YOLO_PATH}")
        
        # Force CPU for stability in cloud environments
        return YOLO(YOLO_PATH)
        
    except Exception as e:
        logger.error(f"YOLO loading error: {e}")
        st.error(f"❌ Failed to load YOLO model: {str(e)}")
        
        # Delete corrupted file so it re-downloads next time
        if os.path.exists(YOLO_PATH):
            os.remove(YOLO_PATH)
        
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
        
        # Run inference on CPU (force for cloud stability)
        results = model.predict(
            source=temp_path,
            save=False,
            conf=conf_threshold,
            verbose=False,
            device='cpu'
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