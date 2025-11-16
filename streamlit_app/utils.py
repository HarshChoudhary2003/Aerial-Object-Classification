import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
from typing import Tuple, Union
from datetime import datetime
import logging
import io
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO path
YOLO_PATH = "models/detection/aerial_detection/weights/best.pt"

# Max upload size
MAX_FILE_SIZE_MB = 50

# ------------------------------
# YOLO MODEL LOADING
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_detection_model() -> YOLO:
    """Load YOLO model only."""
    try:
        if not os.path.exists(YOLO_PATH):
            st.error(
                f"❌ YOLO model not found at:\n\n`{YOLO_PATH}`\n\n"
                "Please upload the model file to this path in your repository."
            )
            st.stop()

        logger.info(f"Loading YOLO model from {YOLO_PATH}")
        return YOLO(YOLO_PATH)

    except Exception as e:
        logger.error(f"YOLO loading error: {e}")
        st.error(f"❌ Failed to load YOLO model: {str(e)}")
        st.stop()

# ------------------------------
# IMAGE VALIDATION
# ------------------------------
def validate_image(image, max_size_mb=MAX_FILE_SIZE_MB):
    """Check size, dimensions, format."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.size[0] < 50 or image.size[1] < 50:
            return False, "Image is too small (min 50x50)"

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        size_mb = buffer.tell() / (1024 * 1024)

        if size_mb > max_size_mb:
            return False, f"Image too large: {size_mb:.1f}MB (max {max_size_mb}MB)"

        return True, "Valid"

    except Exception as e:
        return False, f"Validation failed: {str(e)}"

# ------------------------------
# YOLO DETECTION
# ------------------------------
def predict_detection(model: YOLO, image, conf_threshold=0.5):
    """Run YOLO detection and return annotated image."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        is_valid, msg = validate_image(image)
        if not is_valid:
            raise ValueError(msg)

        # Save to temp file (YOLO requires file path)
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

# ------------------------------
# HISTORY FUNCTIONS
# ------------------------------
def add_to_history(filename: str, task: str, model: str, results: dict):
    """Save results in Streamlit session state."""
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "task": task,
        "model": model,
        "detections": results.get("detection", {}).get("count", 0)
    })

def get_analysis_history() -> pd.DataFrame:
    """Return history as a DataFrame."""
    if "history" not in st.session_state or not st.session_state.history:
        return pd.DataFrame(columns=["timestamp", "filename", "task", "model", "detections"])

    return pd.DataFrame(st.session_state.history)