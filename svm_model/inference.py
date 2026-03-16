"""
inference.py — Backend for the Streamlit app.

Two public functions:
    predict_image(image_path)    -> (label, probs, confidence)
    generate_gradcam(image_path) -> np.ndarray (RGB overlay)

Models are loaded once on first call and reused for all subsequent calls.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import joblib
import cv2
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_svm import FeatureExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLASSES   = ['Real', 'Deepfake', 'AI-Gen']
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------------------------
# Lazy model loader — called once, results cached in _models dict
# ---------------------------------------------------------------------------
_models = {}

def _load_models():
    if _models:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fe = FeatureExtractor().to(device)
    fe.load_state_dict(
        torch.load(os.path.join(_MODEL_DIR, 'feature_extractor.pth'), map_location=device)
    )
    fe.eval()

    _models['device']            = device
    _models['feature_extractor'] = fe
    _models['svm']               = joblib.load(os.path.join(_MODEL_DIR, 'svm_model.joblib'))
    _models['scaler']            = joblib.load(os.path.join(_MODEL_DIR, 'scaler.joblib'))
    _models['pca']               = joblib.load(os.path.join(_MODEL_DIR, 'pca.joblib'))

    # GradCAM hooks are registered once onto the target layer
    _models['gradcam'] = _GradCAM(fe, fe.rgb_backbone._conv_head)


# ---------------------------------------------------------------------------
# Shared preprocessing helper
# ---------------------------------------------------------------------------
def _preprocess(image_path):
    """
    Load image, attempt MTCNN face detection, resize to 224x224.

    Returns:
        img_cropped : np.ndarray (224, 224, 3) uint8 — face crop (or full image)
        img_tensor  : torch.Tensor (1, 3, 224, 224) — normalised tensor
        face_found  : bool
    """
    device  = _models['device']
    img_pil = Image.open(image_path).convert('RGB')
    img_np  = np.array(img_pil)

    face_found = False
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, device=device)
        boxes, _ = mtcnn.detect(img_np)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0].astype(int)
            h, w = img_np.shape[:2]
            pad  = 20
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            img_np     = img_np[y1:y2, x1:x2]
            face_found = True
    except Exception:
        pass  # MTCNN not installed or no face — fall back to full image

    img_cropped = cv2.resize(img_np, (224, 224))
    img_tensor  = _TRANSFORM(Image.fromarray(img_cropped)).unsqueeze(0).to(device)

    return img_cropped, img_tensor, face_found


# ---------------------------------------------------------------------------
# Function 1: Prediction
# ---------------------------------------------------------------------------
def predict_image(image_path):
    """
    Run SVM prediction pipeline on an image.

    Args:
        image_path : str — path to image file (jpg/png)

    Returns:
        label      : str  — predicted class, one of ['Real', 'Deepfake', 'AI-Gen']
        probs      : dict — {class_name: float} probabilities for all 3 classes
        confidence : float — probability of the predicted class (0.0 - 1.0)
    """
    _load_models()

    _, img_tensor, _ = _preprocess(image_path)

    with torch.no_grad():
        features = _models['feature_extractor'](img_tensor).cpu().numpy()

    features_scaled = _models['scaler'].transform(features)
    features_pca    = _models['pca'].transform(features_scaled)

    pred_idx  = _models['svm'].predict(features_pca)[0]
    raw_probs = _models['svm'].predict_proba(features_pca)[0]

    label      = _CLASSES[pred_idx]
    probs      = {cls: float(raw_probs[i]) for i, cls in enumerate(_CLASSES)}
    confidence = float(raw_probs[pred_idx])

    return label, probs, confidence


# ---------------------------------------------------------------------------
# Grad-CAM helper class  (instantiated once in _load_models)
# ---------------------------------------------------------------------------
class _GradCAM:
    """
    Grad-CAM using EfficientNet activations.
    Hooks are registered on init and reused for every generate() call.
    """
    def __init__(self, feature_extractor, target_layer):
        self.feature_extractor = feature_extractor
        self.gradients  = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor):
        self.feature_extractor.eval()
        t     = img_tensor.clone().requires_grad_(True)
        score = self.feature_extractor.rgb_backbone(t).mean()
        self.feature_extractor.zero_grad()
        score.backward()

        weights = torch.mean(self.gradients[0], dim=(1, 2), keepdim=True)
        cam     = torch.relu(torch.sum(weights * self.activations[0], dim=0))
        cam     = cam - cam.min()
        cam     = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# ---------------------------------------------------------------------------
# Function 2: Grad-CAM
# ---------------------------------------------------------------------------
def generate_gradcam(image_path):
    """
    Generate Grad-CAM heatmap overlay for an image.

    Args:
        image_path : str — path to image file (jpg/png)

    Returns:
        overlay : np.ndarray (224, 224, 3) uint8 RGB
                  Grad-CAM heatmap blended over the face crop.
    """
    _load_models()

    img_cropped, img_tensor, _ = _preprocess(image_path)

    cam        = _models['gradcam'].generate(img_tensor)
    cam_resized = cv2.resize(cam, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.4 * img_cropped + 0.6 * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)
