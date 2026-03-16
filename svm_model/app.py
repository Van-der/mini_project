"""
Deepfake Detection — Streamlit App

Setup:
    pip install streamlit

Run:
    cd svm_model
    streamlit run app.py

Dependencies (already in requirements.txt):
    torch, efficientnet_pytorch, scikit-learn, joblib, opencv-python, Pillow, streamlit
"""

import os
import sys
import tempfile
import streamlit as st
from PIL import Image

# Ensure imports from svm_model/ work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import predict_image, generate_gradcam

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Deepfake Detector", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 18px; }
    p, .stText, .stMarkdown p   { font-size: 18px; }
    [data-testid="stMetricValue"]{ font-size: 2rem;  }
    [data-testid="stMetricLabel"]{ font-size: 1rem;  }
</style>
""", unsafe_allow_html=True)

st.title("Deepfake & AI-Generated Face Detector")
st.caption("3-class classifier: Real / Deepfake / AI-Generated  |  EfficientNet-B0 + SVM")
st.divider()

# ---------------------------------------------------------------------------
# Section 1 — Upload
# ---------------------------------------------------------------------------
st.header("Upload Image")
uploaded = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"])

if uploaded is not None:

    # Save to a temporary file so inference functions can read it by path
    suffix   = os.path.splitext(uploaded.name)[1]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded.read())
    tmp_file.close()
    tmp_path = tmp_file.name

    st.divider()

    # ---------------------------------------------------------------------------
    # Section 2 — Results
    # ---------------------------------------------------------------------------
    st.header("Model Results")

    col_left, col_right = st.columns(2)

    # Left: uploaded image
    with col_left:
        st.subheader("Uploaded Image")
        st.image(Image.open(tmp_path), use_container_width=True)

    # Right: prediction
    with col_right:
        st.subheader("Prediction")
        with st.spinner("Running classifier..."):
            try:
                label, probs, confidence = predict_image(tmp_path)

                # Predicted class + confidence
                st.metric("Predicted Class", label)
                st.metric("Confidence", f"{confidence * 100:.1f}%")

                st.write("")
                st.subheader("Class Probabilities")

                for cls, prob in probs.items():
                    st.text(f"{cls:<14} {prob:.4f}")
                    st.progress(float(prob))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.divider()

    # ---------------------------------------------------------------------------
    # Explainability — Grad-CAM
    # ---------------------------------------------------------------------------
    st.subheader("Explainability — Grad-CAM")
    st.caption("Highlighted regions show what the model focused on to make its decision.")

    with st.spinner("Generating Grad-CAM..."):
        try:
            overlay = generate_gradcam(tmp_path)
            col_orig, col_cam = st.columns(2)

            with col_orig:
                st.image(Image.open(tmp_path), caption="Original", use_container_width=True)

            with col_cam:
                st.image(overlay, caption="Grad-CAM overlay", use_container_width=True)

        except Exception as e:
            st.error(f"Grad-CAM failed: {e}")

    # Clean up temp file
    os.unlink(tmp_path)
