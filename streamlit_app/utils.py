import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime
import logging
import io
import tempfile
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Replace with YOUR Hugging Face username
YOLO_URL = "https://huggingface.co/HarshChoudhary2003/aerial-object-models/resolve/main/best.pt"
YOLO_PATH = "models/detection/aerial_detection/weights/best.pt"

MAX_FILE_SIZE_MB = 50

def download_model():
    """Download YOLO model from Hugging Face"""
    if os.path.exists(YOLO_PATH):
        return True
    
    os.makedirs(os.path.dirname(YOLO_PATH), exist_ok=True)
    
    try:
        with st.spinner("⬇️ Downloading YOLO model... This may take 1-2 minutes on first run."):
            response = requests.get(YOLO_URL, stream=True)
            response.raise_for_status()
            
            with open(YOLO_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        st.success("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        st.error("Please check your Hugging Face URL or use manual upload")
        return False

@st.cache_resource(show_spinner=True)
def load_detection_model():
    """Load YOLO model (auto-downloads if missing)"""
    try:
        if not os.path.exists(YOLO_PATH):
            if not download_model():
                st.stop()
        
        logger.info(f"Loading YOLO from {YOLO_PATH}")
        return YOLO(YOLO_PATH)
    except Exception as e:
        logger.error(f"YOLO error: {e}")
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

def validate_image(image, max_size_mb=MAX_FILE_SIZE_MB):
    """Check image validity"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.size[0] < 50 or image.size[1] < 50:
            return False, "Image too small (min 50x50)"

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        size_mb = buffer.tell() / (1024 * 1024)

        if size_mb > max_size_mb:
            return False, f"Image too large: {size_mb:.1f}MB (max {max_size_mb}MB)"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation failed: {str(e)}"

def predict_detection(model, image, conf_threshold=0.5):
    """Run YOLO detection"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        is_valid, msg = validate_image(image)
        if not is_valid:
            raise ValueError(msg)

        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image.save(temp_path, quality=95)

        results = model.predict(
            source=temp_path,
            save=False,
            conf=conf_threshold,
            verbose=False,
            device="cpu"
        )

        os.unlink(temp_path)

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes:
                annotated = result.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                return annotated_rgb, len(result.boxes)

        return np.array(image), 0
    except Exception as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise Exception(f"Detection failed: {str(e)}")

def add_to_history(filename, task, model, results):
    """Save analysis to history"""
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "task": task,
        "model": model,
        "detections": results.get("detection", {}).get("count", 0)
    })

def get_analysis_history():
    """Get analysis history"""
    if "history" not in st.session_state or not st.session_state.history:
        return pd.DataFrame(columns=["timestamp", "filename", "task", "model", "detections"])

    return pd.DataFrame(st.session_state.history)