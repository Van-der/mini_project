import torch
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import BalancedFaceDataset
from train import FeatureExtractor, extract_features


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    print("\nLoading models...")
    svm    = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    pca    = joblib.load('pca.joblib')

    try:
        print("\nLoading pre-extracted features...")
        X = np.load('features_X.npy')
        y = np.load('features_y.npy')
        print(f"Loaded features: {X.shape}")
    except FileNotFoundError:
        print("\nExtracting features (pre-saved not found)...")
        dataset = BalancedFaceDataset('data/cropped_dataset', mode='eval')
        feature_extractor = FeatureExtractor().to(DEVICE)
        X, y = extract_features(dataset, feature_extractor, DEVICE)

    X_scaled = scaler.transform(X)
    X_pca    = pca.transform(X_scaled)
    y_pred   = svm.predict(X_pca)
    y_prob   = svm.predict_proba(X_pca)

    accuracy = accuracy_score(y, y_pred)

    print("\n" + "="*50)
    print(f"FULL EVALUATION REPORT")
    print("="*50)
    print(f"\nOverall Accuracy: {accuracy*100:.1f}%")
    print("\n" + classification_report(y, y_pred,
                              target_names=['Real', 'Deepfake', 'AI-Gen']))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Deepfake', 'AI-Gen'],
                yticklabels=['Real', 'Deepfake', 'AI-Gen'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'SVM Confusion Matrix ({accuracy*100:.1f}% Accuracy)')
    plt.savefig('confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nSaved: confusion_matrix_svm.png")

    print("\nAverage Confidence per Class:")
    for i, cls in enumerate(['Real', 'Deepfake', 'AI-Gen']):
        mask = y == i
        if mask.sum() > 0:
            avg_conf = y_prob[mask, i].mean()
            print(f"  {cls}: {avg_conf*100:.1f}%")


if __name__ == '__main__':
    main()
