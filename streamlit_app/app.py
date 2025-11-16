import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import io
import os
from datetime import datetime

# Import utilities
from utils import (
    load_detection_model,
    predict_detection,
    validate_image,
    add_to_history,
    get_analysis_history
)

# Model path (defined here, not in utils)
YOLO_PATH = "models/detection/aerial_detection/weights/best.pt"

# Page Configuration
st.set_page_config(
    page_title="AerialSurveillance AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - YOUR ORIGINAL DESIGN (UNCHANGED)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .drone-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
        color: white;
        border-left: 5px solid #c92a2a;
    }
    .bird-card {
        background: linear-gradient(135deg, #51cf66 0%, #6bcf7f 100%);
        color: white;
        border-left: 5px solid #2b8a3e;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header - YOUR ORIGINAL DESIGN (UNCHANGED)
st.markdown("""
<div class="main-header">
    <h1> Aerial Object Classification & Detection</h1>
    <p>Advanced AI for Bird vs Drone Identification in Real-Time</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - YOUR ORIGINAL DESIGN (ENHANCED with status)
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # ENHANCED: Model status check
    # Use isfile to ensure we're checking for the model file itself (not just a directory)
    if not os.path.isfile(YOLO_PATH):
        st.warning("⚠️ Model not found. Click button below to install.")
        if st.button("📥 Install YOLO Model"):
            load_detection_model()
            st.rerun()
    else:
        st.success("✅ YOLOv8 Model Ready")

    task = st.radio(
        "Select Task",
        ["📊 Classification Only", "🎯 Detection Only", "🔮 Both Tasks"],
        help="Choose what analysis to perform on uploaded images",
        index=1
    )

    conf_threshold = st.slider(
        "Detection Confidence",
        0.1, 0.9, 0.5, 0.05,
        help="Lower = more detections, Higher = fewer false positives"
    )

    st.markdown("---")

    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
        ### **Three-Step Process**
        
        **1. Image Upload & Validation**
        - Supports JPG, PNG formats
        - Max 50MB size limit
        - Automatic quality checks
        
        **2. AI Model Analysis**
        - **Detection Only** available on Streamlit Cloud
        - YOLOv8 draws bounding boxes
        
        **3. Results & Visualization**
        - Detection count displayed
        - Bounding boxes for detection
        - Download processed images
        """)

    st.markdown("---")

# Main Tabs - YOUR ORIGINAL DESIGN (UNCHANGED)
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Analyze", "📚 Guide", "📊 Comparison", "⚙️ Settings"])

def process_single_image(uploaded_file, task, conf_threshold):
    """Process single image with progress tracking"""
    
    # Validate file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 50:
        st.error(f"❌ File too large: {file_size_mb:.1f}MB (max: 50MB)")
        st.stop()

    # Load image
    image = Image.open(uploaded_file)

    is_valid, msg = validate_image(image)
    if not is_valid:
        st.error(f"❌ {msg}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, use_column_width=True, caption=f"Original: {image.size}")

    results = {}

    # --- CLASSIFICATION DISABLED ---
    if task == "📊 Classification Only":
        st.warning("⚠️ Classification is disabled on Streamlit Cloud")

    # --- DETECTION ---
    if task in ["🎯 Detection Only", "🔮 Both Tasks"]:
        with col2:
            with st.spinner("🔍 Running detection..."):
                model = load_detection_model()
                result_image, num_detections = predict_detection(model, image, conf_threshold)

                st.image(result_image, use_column_width=True,
                         caption=f"Detection: {num_detections} objects")

                if num_detections > 0:
                    st.success(f"✅ Detected {num_detections} objects")
                else:
                    st.warning("⚠️ No objects detected")

                results['detection'] = {'count': num_detections}

    # History
    add_to_history(uploaded_file.name, task, "YOLOv8", results)

    # Download
    if 'result_image' in locals():
        buf = io.BytesIO()
        Image.fromarray(result_image).save(buf, format='JPEG')
        st.download_button(
            label="📥 Download Processed Image",
            data=buf.getvalue(),
            file_name=f"detected_{uploaded_file.name}",
            mime="image/jpeg"
        )

# ---------------- TAB 1 ----------------
with tab1:
    st.header("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, PNG. Max 50MB"
    )

    if uploaded_file:
        process_single_image(uploaded_file, task, conf_threshold)

# ---------------- TAB 2 ----------------
with tab2:
    st.header("📚 Detailed User Guide")
    st.markdown("""
    ### **Quick Start**
    1. **Upload** an aerial image
    2. **Select Detection** task
    3. **Adjust confidence** threshold
    4. **View** bounding boxes
    5. **Download** results
    
    **Note:** Classification is disabled on Streamlit Cloud due to TensorFlow incompatibility.
    """)

# ---------------- TAB 3 ----------------
with tab3:
    st.header("📊 Model Performance Comparison")
    st.info("📌 Classification metrics are disabled. Only YOLO detection is available.")
    
    history_df = get_analysis_history()
    if not history_df.empty:
        total_images = len(history_df)
        total_detections = history_df['detections'].sum()
        avg_detections = history_df['detections'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Images Processed", total_images)
        with col2:
            st.metric("Total Detections", total_detections)
        with col3:
            st.metric("Avg Detections/Image", f"{avg_detections:.1f}")

# ---------------- TAB 4 ----------------
with tab4:
    st.header("⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Path")
        st.text_input("YOLO Model Path", YOLO_PATH, disabled=True)
    
    with col2:
        st.subheader("System Status")
        st.success("✅ YOLOv8 Model Ready")
        st.info("⚠️ Classification Unavailable")

    if st.button("🗑️ Clear Cache & History"):
        st.cache_resource.clear()
        if 'history' in st.session_state:
            st.session_state.history = []
        st.success("✅ Cache cleared!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🛰️ AerialSurveillance AI System v1.0 | YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)