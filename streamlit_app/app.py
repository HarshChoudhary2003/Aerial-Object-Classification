import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import utilities (now with automatic model setup)
from utils import (
    load_classification_model, 
    load_detection_model, 
    predict_classification, 
    predict_detection,
    get_model_metrics,
    validate_image,
    add_to_history,
    ensure_models_available  # New: Auto-download missing models
)

# --- DEPLOYMENT SETUP ---
# This runs on first load to ensure models are available
@st.cache_resource(show_spinner="🔧 Setting up models for first run...")
def initialize_app():
    """Initialize app by ensuring models are available"""
    ensure_models_available()
    return True

# Initialize the app
initialize_app()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AerialSurveillance AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .deploy-badge {
        display: inline-block;
        background: #ff4b4b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛰️ Aerial Object Classification & Detection</h1>
    <p>Advanced AI for Bird vs Drone Identification in Real-Time</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Deployment indicator
    if st.session_state.get('app_initialized', False):
        st.success("✅ Models Ready")
    
    task = st.radio(
        "Select Task",
        ["📊 Classification Only", "🎯 Detection Only", "🔮 Both Tasks"],
        help="Choose what analysis to perform on uploaded images"
    )
    
    if task != "🎯 Detection Only":
        model_type = st.selectbox(
            "Classification Model",
            ["ResNet50 Transfer Learning", "Custom CNN"],
            help="Choose model architecture"
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
        - **Classification**: ResNet50/Custom CNN predicts Bird/Drone
        - **Detection**: YOLOv8 draws bounding boxes
        - **Both**: Runs both models in parallel
        
        **3. Results & Visualization**
        - Confidence scores displayed
        - Bounding boxes for detections
        - Download processed images
        
        **Tech Stack**: TensorFlow, YOLOv8, Streamlit
        """)
    
    st.markdown("---")
    
    if st.checkbox("Show Model Performance"):
        st.subheader("📊 Model Metrics")
        metrics = get_model_metrics()
        st.dataframe(metrics, use_container_width=True)
        
        # Show which models are actually available
        available_models = metrics[metrics['Model Exists']]['Model Exists'].count()
        st.info(f"📦 {available_models} models available")

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Analyze", "📚 Guide", "📊 Comparison", "⚙️ Settings"])

def process_single_image(uploaded_file, task, model_type, conf_threshold):
    """Process single image with progress tracking"""
    # Validate file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 50:
        st.error(f"❌ File too large: {file_size_mb:.1f}MB (max: 50MB)")
        st.stop()
    
    # Load image
    image = Image.open(uploaded_file)
    
    # Validate image
    is_valid, msg = validate_image(image)
    if not is_valid:
        st.error(f"❌ Invalid image: {msg}")
        st.stop()
    
    # Display original
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, use_column_width=True, caption=f"Original: {image.size}")
    
    results = {}
    
    # Classification
    if task in ["📊 Classification Only", "🔮 Both Tasks"]:
        with st.spinner("🤖 Running classification..."):
            try:
                model = load_classification_model(model_type)
                label, confidence = predict_classification(model, image)
                
                st.markdown(f"""
                <div class="result-card {'drone-card' if label == 'DRONE' else 'bird-card'}">
                    <h2 style="color: white; margin: 0;">{'🚁' if label == 'DRONE' else '🐦'} {label}</h2>
                    <h3 style="color: white; margin: 0; opacity: 0.9;">Confidence: {confidence:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                results['classification'] = {'label': label, 'confidence': confidence}
            except Exception as e:
                st.error(f"❌ Classification failed: {e}")
                return
    
    # Detection
    if task in ["🎯 Detection Only", "🔮 Both Tasks"]:
        with col2:
            with st.spinner("🔍 Running detection..."):
                try:
                    yolo_model = load_detection_model()
                    result_image, num_detections = predict_detection(yolo_model, image, conf_threshold)
                    
                    st.image(result_image, use_column_width=True, 
                            caption=f"Detection: {num_detections} objects")
                    
                    if num_detections > 0:
                        st.success(f"✅ Detected {num_detections} objects")
                    else:
                        st.warning("⚠️ No objects detected")
                    
                    results['detection'] = {'count': num_detections}
                    
                    # Download button
                    buf = io.BytesIO()
                    Image.fromarray(result_image).save(buf, format='JPEG')
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=buf.getvalue(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
                except Exception as e:
                    st.error(f"❌ Detection failed: {e}")
                    return
    
    # Add to history
    add_to_history(uploaded_file.name, task, model_type, results)

with tab1:
    st.header("📤 Upload Image")
    
    # Show setup instructions if models are missing
    metrics = get_model_metrics()
    if not metrics['Model Exists'].any():
        st.warning("""
        ### ⚠️ Models Not Found
        
        The AI models need to be downloaded on first run. This may take a few minutes.
        
        **What to expect:**
        - Models will be downloaded automatically when you upload an image
        - Download size: ~200MB total
        - This only happens once
        """)
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, PNG. Max 50MB"
    )
    
    if uploaded_file:
        process_single_image(uploaded_file, task, model_type, conf_threshold)

with tab2:
    st.header("📚 Detailed User Guide")
    
    st.markdown("""
    ### **Getting Started**
    
    1. **Setup**: Models are downloaded automatically on first use
    2. **Upload**: Drag & drop or click to select images
    3. **Analyze**: Choose task (Classification/Detection/Both)
    4. **Review**: Check confidence scores and bounding boxes
    5. **Export**: Download results or view history
    
    ### **Troubleshooting**
    
    **Model Not Found Error:**
    - Wait for automatic download to complete
    - Check internet connection
    - Verify disk space (at least 500MB free)
    
    **Memory Issues:**
    - Use smaller images (<10MB)
    - Run one task at a time
    - Restart the app if needed
    """)

with tab3:
    st.header("📊 Model Performance Dashboard")
    
    metrics = get_model_metrics().reset_index()
    
    # Show availability status
    st.info("✅ Models marked as 'Available' are ready to use")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(metrics, x='Model', y='Accuracy', title='Accuracy Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(metrics, x='Precision', y='Recall', size='F1-Score', color='Model',
                        title='Precision vs Recall')
        st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.subheader("📋 Detailed Metrics")
    st.dataframe(metrics, use_container_width=True)

with tab4:
    st.header("⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Paths")
        st.text_input("Classification Path", "models/classification/")
        st.text_input("Detection Path", "models/detection/")
    
    with col2:
        st.subheader("Advanced Settings")
        st.checkbox("Enable GPU", value=False, disabled=True, 
                   help="GPU not available in Streamlit Cloud")
        debug_mode = st.checkbox("Debug Mode", value=False)
        if debug_mode:
            st.info("Debug mode enabled. Check logs in the app.")
    
    if st.button("🗑️ Clear Cache & History"):
        st.cache_resource.clear()
        st.cache_data.clear()
        if 'history' in st.session_state:
            st.session_state.history = []
        st.success("✅ Cache cleared")
        st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🛰️ AerialSurveillance AI System v1.0 | Built with TensorFlow, YOLOv8 & Streamlit</p>
    <p style="font-size: 0.8rem;">Deployed on Streamlit Cloud 🚀</p>
</div>
""", unsafe_allow_html=True)