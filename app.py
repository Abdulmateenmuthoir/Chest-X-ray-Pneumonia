"""
Streamlit Web Application
A professional web interface for pneumonia classification from chest X-rays.
Designed for live demonstration during project defense.

Run with: streamlit run app.py
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import config


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Pneumonia Classifier - Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for Professional Styling
# ============================================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .result-normal {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4caf50;
    }
    .result-pneumonia {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border: 2px solid #f44336;
    }
    .result-label {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.8;
    }
    
    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: #fff3e0;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load Model (cached for performance)
# ============================================================
@st.cache_resource
def load_classification_model():
    """Load the trained model with caching."""
    model_path = os.path.join(config.MODEL_DIR, "pneumonia_classifier_final.keras")
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    return None


@st.cache_data
def load_metrics():
    """Load evaluation metrics if available."""
    metrics_path = os.path.join(config.LOGS_DIR, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def preprocess_image(image):
    """Preprocess uploaded image for prediction."""
    # Resize to model input size
    img = image.resize(config.IMG_SIZE)
    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Convert to array and normalize
    img_array = img_to_array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def make_prediction(model, img_array):
    """Make prediction and return results."""
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = 1 if prediction > 0.5 else 0
    class_name = config.CLASS_NAMES[predicted_class]
    confidence = prediction if predicted_class == 1 else 1 - prediction
    
    return {
        "class_name": class_name,
        "confidence": float(confidence),
        "raw_score": float(prediction)
    }


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.title("📋 Project Info")
    
    st.markdown("""
    **Project Title:**  
    Automated Pneumonia Classification from Chest Radiographs Using Deep CNNs
    
    **Student:**  
    Dauda Suliyat Adewumi  
    (AI/HND/F24/0073)
    
    **Institution:**  
    Department of Artificial Intelligence  
    School of Computing  
    Federal Polytechnic Offa
    
    **Supervisor:**  
    Mr. Amari B.R
    
    ---
    
    **Model:** ResNet50  
    **Framework:** TensorFlow/Keras  
    **Task:** Binary Classification  
    (Normal vs Pneumonia)
    """)
    
    st.divider()
    
    # Display model metrics if available
    metrics = load_metrics()
    if metrics:
        st.subheader("📊 Model Performance")
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
    
    st.divider()
    st.caption("© 2024/2025 Academic Session")


# ============================================================
# Main Content
# ============================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🫁 Pneumonia Classification System</h1>
    <p>Automated Chest X-ray Analysis Using Deep Learning (ResNet50)</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_classification_model()

if model is None:
    st.error("""
    ⚠️ **Model not found!**  
    Please train the model first by running:
    ```
    python train.py
    ```
    """)
    st.stop()

# Two-column layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Chest X-ray")
    
    st.markdown("""
    <div class="info-box">
        Upload a chest X-ray image (JPEG, PNG, or JPG format) to get an 
        automated pneumonia classification result.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a frontal chest X-ray image for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
        
        # Classify button
        classify_btn = st.button("🔍 Classify Image", type="primary", use_container_width=True)
    else:
        classify_btn = False
        st.markdown("---")
        st.info("👆 Please upload a chest X-ray image to begin analysis")

with col2:
    st.subheader("📊 Classification Result")
    
    if uploaded_file is not None and classify_btn:
        with st.spinner("🔄 Analyzing chest X-ray..."):
            # Preprocess and predict
            img_array = preprocess_image(image)
            result = make_prediction(model, img_array)
        
        # Display result
        if result["class_name"] == "NORMAL":
            st.markdown(f"""
            <div class="result-card result-normal">
                <div style="font-size: 3rem;">✅</div>
                <div class="result-label" style="color: #2e7d32;">NORMAL</div>
                <div class="confidence-text">Confidence: {result['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.success("The chest X-ray appears to be **NORMAL**.")
        else:
            st.markdown(f"""
            <div class="result-card result-pneumonia">
                <div style="font-size: 3rem;">⚠️</div>
                <div class="result-label" style="color: #c62828;">PNEUMONIA</div>
                <div class="confidence-text">Confidence: {result['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.warning("The chest X-ray shows signs of **PNEUMONIA**.")
        
        # Detailed results
        st.markdown("---")
        st.subheader("📋 Detailed Results")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        with detail_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['class_name']}</div>
                <div class="metric-label">Prediction</div>
            </div>
            """, unsafe_allow_html=True)
        with detail_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['confidence']:.1%}</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with detail_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['raw_score']:.4f}</div>
                <div class="metric-label">Raw Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(result["confidence"])
        
    elif uploaded_file is None:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #9e9e9e;">
            <div style="font-size: 4rem;">🫁</div>
            <p style="font-size: 1.1rem;">Upload a chest X-ray image<br>to see classification results</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Click **Classify Image** to analyze the uploaded X-ray")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Disclaimer:</strong> This tool is designed for educational and research purposes only. 
    It should NOT be used as a substitute for professional medical diagnosis. Always consult a 
    qualified healthcare professional for medical decisions.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9e9e9e; font-size: 0.85rem;">
    Automated Pneumonia Classification System | ResNet50 + TensorFlow/Keras<br>
    Dauda Suliyat Adewumi | Federal Polytechnic Offa | 2024/2025
</div>
""", unsafe_allow_html=True)
