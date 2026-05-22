"""
Single image prediction using SVM model
Usage: python predict.py path/to/image.jpg
"""
import torch
import numpy as np
import joblib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

from train import FeatureExtractor
from dataset import get_transforms


def predict_image(image_path, skip_face_detection=False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)

    if 'cropped_dataset' in image_path or skip_face_detection:
        print("Using pre-cropped image (skipping face detection)")
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

    print("Extracting features...")
    feature_extractor = FeatureExtractor().to(DEVICE)
    feature_extractor.load_state_dict(torch.load('feature_extractor.pth', map_location=DEVICE))
    feature_extractor.eval()

    with torch.no_grad():
        features = feature_extractor(img_tensor).cpu().numpy()

    svm    = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    pca    = joblib.load('pca.joblib')

    features_scaled = scaler.transform(features)
    features_pca    = pca.transform(features_scaled)
    pred_idx = svm.predict(features_pca)[0]
    probs    = svm.predict_proba(features_pca)[0]

    classes = ['Real', 'Deepfake', 'AI-Gen']

    print("\n" + "="*50)
    print(f"Prediction: {classes[pred_idx]}")
    print(f"Confidence: {probs[pred_idx]*100:.1f}%")
    print("="*50)
    print("\nAll probabilities:")
    for i, cls in enumerate(classes):
        bar = '█' * int(probs[i] * 30)
        print(f"  {cls:10s}: {probs[i]*100:5.1f}% {bar}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    colors = ['green', 'red', 'orange']
    bars = axes[1].barh(classes, probs * 100, color=colors)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title(f'Prediction: {classes[pred_idx]} ({probs[pred_idx]*100:.1f}%)')

    for bar, prob in zip(bars, probs):
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nSaved: prediction_result.png")
    return classes[pred_idx], probs


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    predict_image(sys.argv[1])
