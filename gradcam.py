"""
Grad-CAM visualization for SVM model
Usage: python gradcam.py path/to/image.jpg
"""
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import sys

from train import FeatureExtractor
from dataset import get_transforms


class GradCAMForSVM:
    def __init__(self, feature_extractor, svm_model, scaler, target_layer):
        self.feature_extractor = feature_extractor
        self.svm = svm_model
        self.scaler = scaler
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.feature_extractor.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        rgb_features = self.feature_extractor.rgb_backbone(input_tensor)
        score = rgb_features.mean()
        self.feature_extractor.zero_grad()
        score.backward()

        gradients  = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def predict_with_gradcam(image_path, skip_face_detection=False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    original_img = img_np.copy()

    if 'cropped_dataset' in image_path or skip_face_detection:
        print("Using pre-cropped image")
        img_cropped = cv2.resize(img_np, (224, 224))
    else:
        print("\nDetecting face...")
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(keep_all=False, device=DEVICE)

        face_boxes, _ = mtcnn.detect(img_np)
        if face_boxes is None or len(face_boxes) == 0:
            print("No face detected - using full image")
            img_cropped = cv2.resize(img_np, (224, 224))
        else:
            print("Face detected!")
            box = face_boxes[0]
            x1, y1, x2, y2 = box.astype(int)
            h, w = img_np.shape[:2]
            pad = 20
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            img_cropped = img_np[y1:y2, x1:x2]
            img_cropped = cv2.resize(img_cropped, (224, 224))

    transform = get_transforms(train=False)
    img_tensor = transform(image=img_cropped)['image'].unsqueeze(0).to(DEVICE)

    print("Loading models...")
    feature_extractor = FeatureExtractor().to(DEVICE)
    feature_extractor.load_state_dict(torch.load('feature_extractor.pth', map_location=DEVICE))
    feature_extractor.eval()

    svm    = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    pca    = joblib.load('pca.joblib')

    with torch.no_grad():
        features = feature_extractor(img_tensor).cpu().numpy()

    features_scaled = scaler.transform(features)
    features_pca    = pca.transform(features_scaled)
    pred_idx = svm.predict(features_pca)[0]
    probs    = svm.predict_proba(features_pca)[0]

    classes = ['Real', 'Deepfake', 'AI-Gen']
    print(f"\nPrediction: {classes[pred_idx]}  ({probs[pred_idx]*100:.1f}%)")

    print("\nGenerating Grad-CAM...")
    target_layer = feature_extractor.rgb_backbone._conv_head
    gradcam = GradCAMForSVM(feature_extractor, svm, scaler, target_layer)

    img_tensor_grad = transform(image=img_cropped)['image'].unsqueeze(0).to(DEVICE)
    cam = gradcam.generate(img_tensor_grad, pred_idx)

    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.4 * img_cropped + 0.6 * heatmap, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original_img);  axes[0].set_title('Original');        axes[0].axis('off')
    axes[1].imshow(img_cropped);   axes[1].set_title('Face Crop');       axes[1].axis('off')
    axes[2].imshow(cam_resized, cmap='jet'); axes[2].set_title('Grad-CAM'); axes[2].axis('off')
    axes[3].imshow(overlay);       axes[3].set_title(f'{classes[pred_idx]} ({probs[pred_idx]*100:.1f}%)'); axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nSaved: gradcam_result.png")

    return classes[pred_idx], probs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python gradcam.py <image_path>")
        sys.exit(1)

    predict_with_gradcam(sys.argv[1])
