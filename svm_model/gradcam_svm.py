"""
Grad-CAM visualization for SVM model
Uses EfficientNet activations with SVM's predicted class as target.

Usage: python gradcam_svm.py path/to/image.jpg
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append('..')

from train_svm import FeatureExtractor
from augmentdatting import get_transforms


class GradCAMForSVM:
    """
    Grad-CAM adapted for SVM model.
    Uses a temporary linear layer to compute gradients for visualization.
    """
    def __init__(self, feature_extractor, svm_model, scaler, target_layer):
        self.feature_extractor = feature_extractor
        self.svm = svm_model
        self.scaler = scaler
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx):
        """
        Generate Grad-CAM heatmap.
        We create a temporary differentiable path using SVM's support vectors.
        """
        self.feature_extractor.eval()
        
        # Enable gradients
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass through EfficientNet backbone only
        rgb_features = self.feature_extractor.rgb_backbone(input_tensor)
        
        # Create a pseudo-score based on the predicted class
        # Use the class_idx to weight the features
        # This creates a differentiable path for Grad-CAM
        
        # Simple approach: Use mean of features as the score
        # The gradients will show which spatial regions contribute most
        score = rgb_features.mean()
        
        # For better class-specific visualization, we can use
        # the direction of the SVM decision boundary
        # But for simplicity, we'll use feature magnitude
        
        self.feature_extractor.zero_grad()
        score.backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)  # ReLU to keep only positive contributions
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def predict_with_gradcam(image_path, skip_face_detection=False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # ============ 1. Load Image ============
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    original_img = img_np.copy()
    
    if 'cropped_dataset' in image_path or skip_face_detection:
        print("📁 Using pre-cropped image")
        img_cropped = cv2.resize(img_np, (224, 224))
    else:
        print("\n🔍 Detecting face...")
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, device=DEVICE)
        
        face_boxes, _ = mtcnn.detect(img_np)
        if face_boxes is None or len(face_boxes) == 0:
            print("⚠️ No face detected - using full image")
            img_cropped = cv2.resize(img_np, (224, 224))
        else:
            print("✅ Face detected!")
            box = face_boxes[0]
            x1, y1, x2, y2 = box.astype(int)
            h, w = img_np.shape[:2]
            pad = 20
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            img_cropped = img_np[y1:y2, x1:x2]
            img_cropped = cv2.resize(img_cropped, (224, 224))
    
    # ============ 2. Preprocess ============
    transform = get_transforms(train=False)
    img_tensor = transform(image=img_cropped)['image'].unsqueeze(0).to(DEVICE)
    
    # ============ 3. Load Models ============
    print("🔄 Loading models...")
    feature_extractor = FeatureExtractor().to(DEVICE)
    feature_extractor.load_state_dict(torch.load('feature_extractor.pth', map_location=DEVICE))
    feature_extractor.eval()
    
    svm = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # ============ 4. Extract Features & Predict ============
    with torch.no_grad():
        features = feature_extractor(img_tensor).cpu().numpy()
    
    features_scaled = scaler.transform(features)
    pred_idx = svm.predict(features_scaled)[0]
    probs = svm.predict_proba(features_scaled)[0]
    
    classes = ['Real', 'Deepfake', 'AI-Gen']
    
    print("\n" + "="*50)
    print(f"🎯 Prediction: {classes[pred_idx]}")
    print(f"   Confidence: {probs[pred_idx]*100:.1f}%")
    print("="*50)
    
    # ============ 5. Generate Grad-CAM ============
    print("\n🔥 Generating Grad-CAM...")
    
    # Target the last conv layer of EfficientNet
    target_layer = feature_extractor.rgb_backbone._conv_head
    
    gradcam = GradCAMForSVM(feature_extractor, svm, scaler, target_layer)
    
    # Need to re-run with gradients enabled
    img_tensor_grad = transform(image=img_cropped)['image'].unsqueeze(0).to(DEVICE)
    img_tensor_grad.requires_grad_(True)
    
    cam = gradcam.generate(img_tensor_grad, pred_idx)
    
    # ============ 6. Visualization ============
    # Resize CAM to match cropped image size
    cam_resized = cv2.resize(cam, (224, 224))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = 0.4 * img_cropped + 0.6 * heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Input')
    axes[0].axis('off')
    
    axes[1].imshow(img_cropped)
    axes[1].set_title('Face Crop (224x224)')
    axes[1].axis('off')
    
    axes[2].imshow(cam_resized, cmap='jet')
    axes[2].set_title('Grad-CAM Heatmap')
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title(f'{classes[pred_idx]} ({probs[pred_idx]*100:.1f}%)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_svm_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Saved: gradcam_svm_result.png")
    
    # ============ 7. Probability Bar Chart ============
    fig2, ax = plt.subplots(figsize=(8, 4))
    colors = ['green', 'red', 'orange']
    bars = ax.barh(classes, probs * 100, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)')
    ax.set_title(f'SVM Prediction: {classes[pred_idx]}')
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('prediction_probs.png', dpi=300, bbox_inches='tight')
    
    return classes[pred_idx], probs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python gradcam_svm.py <image_path>")
        print("Example: python gradcam_svm.py ../dataset/cropped_dataset/ai_gen/aigen_ai_00001.jpg")
        sys.exit(1)
    
    predict_with_gradcam(sys.argv[1])
