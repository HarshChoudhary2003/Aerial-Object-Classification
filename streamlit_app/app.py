import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import (
    load_classification_model, 
    load_detection_model, 
    predict_classification, 
    predict_detection,
    get_model_metrics,
    validate_image,
    add_to_history,
    get_analysis_history
)

# Page Configuration
st.set_page_config(
    page_title="AerialSurveillance AI",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e);
        color: #e94560;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 3rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #a2aab8;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    .result-card {
        padding: 2rem;
        border-radius: 18px;
        margin: 1.5rem 0;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .drone-card {
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.15) 0%, rgba(233, 69, 96, 0.05) 100%);
        border-left: 6px solid #e94560;
    }
    
    .bird-card {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(79, 172, 254, 0.05) 100%);
        border-left: 6px solid #4facfe;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
        transform: scale(1.02);
    }
    
    .sidebar .sidebar-content {
        background-color: #0f3460;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>AERIAL SURVEILLANCE AI</h1>
    <p>Next-Gen Computer Vision for Airspace Safety & Wildlife Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙ Configuration")
    
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
        ### *Three-Step Process*
        
        *1. Image Upload & Validation*
        - Supports JPG, PNG formats
        - Max 50MB size limit
        - Automatic quality checks
        
        *2. AI Model Analysis*
        - *Classification*: ResNet50/Custom CNN predicts Bird/Drone
        - *Detection*: YOLOv8 draws bounding boxes
        - *Both*: Runs both models in parallel
        
        *3. Results & Visualization*
        - Confidence scores displayed
        - Bounding boxes for detections
        - Download processed images
        
        *Tech Stack*: TensorFlow, YOLOv8, Streamlit
        """)
    
    st.markdown("---")
    
    if st.checkbox("Show Model Performance"):
        st.subheader("📊 Model Metrics")
        metrics = get_model_metrics()
        st.dataframe(metrics, use_container_width=True)

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Analyze", "📚 Guide", "📊 Comparison", "⚙ Settings"])

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
    if not validate_image(image):
        st.error("❌ Invalid image format")
        st.stop()
    
    # Display original
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, use_column_width=True, caption=f"Original: {image.size}")
    
    results = {}
    
    # Classification
    if task in ["📊 Classification Only", "🔮 Both Tasks"]:
        with st.spinner("🤖 Running classification..."):
            model = load_classification_model(model_type)
            label, confidence = predict_classification(model, image)
            
            st.markdown(f"""
            <div class="result-card {'drone-card' if label == 'DRONE' else 'bird-card'}">
                <h2 style="color: white; margin: 0;">{'🚁' if label == 'DRONE' else '🐦'} {label}</h2>
                <h3 style="color: white; margin: 0; opacity: 0.9;">Confidence: {confidence:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            results['classification'] = {'label': label, 'confidence': confidence}
    
    # Detection
    if task in ["🎯 Detection Only", "🔮 Both Tasks"]:
        with col2:
            with st.spinner("🔍 Running detection..."):
                yolo_model = load_detection_model()
                result_image, num_detections = predict_detection(yolo_model, image, conf_threshold)
                
                st.image(result_image, use_column_width=True, 
                        caption=f"Detection: {num_detections} objects")
                
                if num_detections > 0:
                    st.success(f"✅ Detected {num_detections} objects")
                else:
                    st.warning("⚠ No objects detected")
                
                results['detection'] = {'count': num_detections}
    
    # Add to history
    add_to_history(uploaded_file.name, task, model_type, results)
    
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

with tab1:
    st.header("📤 Upload & Analyze")
    
    # Sample selection
    st.subheader("🧪 Quick Test")
    samples = {
        "None": None,
        "🐦 Bird Sample 1": "data/classification/test/bird/00083b384685315d_jpg.rf.abfd1b2cc8c681777bae66d5327bb9ea.jpg",
        "🐦 Bird Sample 2": "data/classification/test/bird/00188d7f40a84793_jpg.rf.7f9da2b662dc236fbdcc1f22d8e0983e.jpg",
        "🚁 Drone Sample 1": "data/classification/test/drone/foto01799_png.rf.7b06ce6abb9f307efa437ed34e863e21.jpg",
        "🚁 Drone Sample 2": "data/classification/test/drone/foto01915_png.rf.7d7cd852392707f519d13e9cf051de3f.jpg"
    }
    
    selected_sample = st.selectbox("Choose a sample image to test immediately", list(samples.keys()))
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Or upload your own image...",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, PNG. Max 50MB"
    )
    
    if uploaded_file:
        process_single_image(uploaded_file, task, model_type, conf_threshold)
    elif selected_sample != "None":
        sample_path = samples[selected_sample]
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                content = f.read()
                # Use io.BytesIO to properly mimic an uploaded file
                wrapped_file = io.BytesIO(content)
                wrapped_file.name = os.path.basename(sample_path)
                wrapped_file.size = len(content)
                
                process_single_image(wrapped_file, task, model_type, conf_threshold)
        else:
            st.warning(f"Sample not found: {sample_path}")

with tab2:
    st.header("📚 Detailed User Guide")
    
    st.markdown("""
    ### *Getting Started*
    
    1. *Setup*: Ensure models are trained and saved in models/ folder
    2. *Upload*: Drag & drop or click to select images
    3. *Analyze*: Choose task (Classification/Detection/Both)
    4. *Review*: Check confidence scores and bounding boxes
    5. *Export*: Download results or view history
    
    ### *Troubleshooting*
    
    *Model Not Found Error:*
    - Train classification models via notebook
    - Train YOLOv8 model in terminal
    - Verify folder structure matches documentation
    """)

with tab3:
    st.header("📊 Model Performance Dashboard")
    
    metrics = get_model_metrics().reset_index()
    
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
    st.header("⚙ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Paths")
        st.text_input("Classification Path", "models/classification/")
        st.text_input("Detection Path", "models/detection/")
    
    with col2:
        st.subheader("Advanced Settings")
        st.checkbox("Enable GPU", value=False)
        st.checkbox("Debug Mode", value=False)
    
    if st.button("🗑 Clear Cache & History"):
        st.cache_resource.clear()
        if 'history' in st.session_state:
            st.session_state.history = []
        st.success("✅ Cache cleared")

st.markdown("---")
#  CORRECT (renders properly)
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🛰 AerialSurveillance AI System v1.0 | Built with TensorFlow, YOLOv8 & Streamlit</p>
</div>
""", unsafe_allow_html=True)