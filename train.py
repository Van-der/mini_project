# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import json
import joblib
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from efficientnet_pytorch import EfficientNet
from dataset import BalancedFaceDataset


class FeatureExtractor(nn.Module):
    """Frozen EfficientNet-B0 + FFT feature extractor (no learnable weights)"""

    def __init__(self):
        super().__init__()
        self.rgb_backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.rgb_backbone._fc = nn.Identity()
        for param in self.parameters():
            param.requires_grad = False

    def get_fft_features(self, x):
        """64-dim log-magnitude FFT features (grayscale, 8x8 pooled)"""
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray)
        magnitude = torch.log1p(torch.abs(torch.fft.fftshift(fft)))
        pooled = nn.functional.adaptive_avg_pool2d(magnitude.unsqueeze(1), (8, 8))
        return pooled.view(pooled.size(0), -1)

    def forward(self, x):
        rgb_features = self.rgb_backbone(x)          # [B, 1280]
        fft_features = self.get_fft_features(x)      # [B,   64]
        return torch.cat([rgb_features, fft_features], dim=1)  # [B, 1344]


def extract_features(dataset, feature_extractor, device, batch_size=16):
    """Extract features from entire dataset (no grad, deterministic)"""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features, all_labels = [], []
    feature_extractor.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            all_features.append(feature_extractor(imgs).cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_features), np.concatenate(all_labels)


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    print("\nLoading dataset (eval mode - no augmentation)...")
    dataset = BalancedFaceDataset('data/cropped_dataset', mode='eval')
    print(f"Total samples: {len(dataset)}")

    print("\nExtracting features with frozen EfficientNet + FFT...")
    feature_extractor = FeatureExtractor().to(DEVICE)
    torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')

    X, y = extract_features(dataset, feature_extractor, DEVICE)
    print(f"Feature shape: {X.shape}")   # (800, 1344)

    np.save('features_X.npy', X)
    np.save('features_y.npy', y)

    # 70 / 15 / 15 stratified split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )
    print(f"\nSplit -> train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.joblib')

    print("\nFitting PCA (1344 -> 100 components)...")
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    X_test_pca  = pca.transform(X_test_sc)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Variance retained: {explained*100:.1f}%")
    joblib.dump(pca, 'pca.joblib')

    print("\nRunning 5-fold cross-validation on training set...")
    cv_svm = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True, random_state=42
    )
    cv_scores = cross_val_score(cv_svm, X_train_pca, y_train, cv=5, scoring='accuracy')
    print(f"CV scores:  {cv_scores.round(3)}")
    print(f"CV mean:    {cv_scores.mean()*100:.1f}%  +/-  {cv_scores.std()*100:.1f}%")

    print("\nTraining SVM (C=1.0, RBF kernel)...")
    svm = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True, random_state=42
    )
    svm.fit(X_train_pca, y_train)
    joblib.dump(svm, 'svm_model.joblib')

    y_train_pred = svm.predict(X_train_pca)
    y_val_pred   = svm.predict(X_val_pca)
    y_test_pred  = svm.predict(X_test_pca)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc   = accuracy_score(y_val,   y_val_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)

    print(f"\n{'='*55}")
    print(f"  Train accuracy : {train_acc*100:.1f}%")
    print(f"  Val accuracy   : {val_acc*100:.1f}%")
    print(f"  Test accuracy  : {test_acc*100:.1f}%  <- honest generalisation")
    gap = train_acc - test_acc
    print(f"  Overfit gap    : {gap*100:.1f}%  (train - test)")
    print(f"{'='*55}")

    print("\nValidation report:")
    print(classification_report(y_val, y_val_pred,
                                target_names=['Real', 'Deepfake', 'AI-Gen']))

    print("Test report:")
    print(classification_report(y_test, y_test_pred,
                                target_names=['Real', 'Deepfake', 'AI-Gen']))

    history = {
        'train_accuracy':    train_acc,
        'val_accuracy':      val_acc,
        'test_accuracy':     test_acc,
        'overfit_gap':       float(gap),
        'cv_mean':           float(cv_scores.mean()),
        'cv_std':            float(cv_scores.std()),
        'pca_components':    100,
        'pca_variance_kept': float(explained),
        'feature_dim_raw':   int(X.shape[1]),
        'feature_dim_pca':   100,
        'train_samples':     len(X_train),
        'val_samples':       len(X_val),
        'test_samples':      len(X_test),
        'svm_params':        svm.get_params()
    }
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print("\nFiles saved:")
    print("  svm_model.joblib")
    print("  scaler.joblib")
    print("  pca.joblib")
    print("  feature_extractor.pth")
    print("  features_X.npy / features_y.npy")
    print("  training_history.json")


if __name__ == '__main__':
    main()
