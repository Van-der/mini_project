import torch
import torch.nn as nn
import numpy as np
import json
import joblib
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
sys.path.append('..')

from efficientnet_pytorch import EfficientNet
from augmentdatting import BalancedFaceDataset

class FeatureExtractor(nn.Module):
    """Frozen feature extractor using EfficientNet + raw FFT features"""
    def __init__(self):
        super().__init__()
        # RGB Branch (frozen pretrained) - well-trained features!
        self.rgb_backbone = EfficientNet.from_pretrained('efficientnet-b0')
        rgb_features = self.rgb_backbone._fc.in_features
        self.rgb_backbone._fc = nn.Identity()
        
        # Freeze all parameters - no training!
        for param in self.parameters():
            param.requires_grad = False
    
    def get_fft_features(self, x):
        """Extract raw FFT magnitude features (no learnable weights)"""
        # Convert to grayscale: [B, 3, H, W] -> [B, 1, H, W]
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        
        # 2D FFT
        fft = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft)  # Center low frequencies
        magnitude = torch.abs(fft_shifted)
        
        # Log scale for better dynamic range
        magnitude = torch.log1p(magnitude)
        
        # Pool to fixed size features using adaptive pooling
        # Divide into 8x8 regions and take mean of each
        magnitude = magnitude.unsqueeze(1)  # [B, 1, H, W]
        pooled = nn.functional.adaptive_avg_pool2d(magnitude, (8, 8))  # [B, 1, 8, 8]
        
        # Flatten: [B, 64]
        fft_features = pooled.view(pooled.size(0), -1)
        return fft_features

    def forward(self, x):
        # RGB features: [B, 1280]
        rgb_features = self.rgb_backbone(x)
        
        # FFT features: [B, 64] - raw, no learnable weights
        fft_features = self.get_fft_features(x)
        
        # Combined: [B, 1344]
        combined = torch.cat([rgb_features, fft_features], dim=1)
        return combined


def extract_features(dataset, feature_extractor, device, batch_size=16):
    """Extract features from entire dataset"""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_features = []
    all_labels = []
    
    feature_extractor.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            features = feature_extractor(imgs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    return X, y


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # ============ STEP 1: Load Dataset ============
    print("\n📁 Loading dataset...")
    dataset = BalancedFaceDataset('../dataset/cropped_dataset')
    print(f"✓ Total samples: {len(dataset)}")
    
    # ============ STEP 2: Extract Features ============
    print("\n🔄 Extracting features using frozen EfficientNet + FFT...")
    feature_extractor = FeatureExtractor().to(DEVICE)
    
    # Save feature extractor for consistent inference
    torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')
    print("✓ Saved: feature_extractor.pth")
    
    X, y = extract_features(dataset, feature_extractor, DEVICE)
    print(f"✓ Feature shape: {X.shape}")  # Should be (600, 1344)
    
    # Save features for future use
    np.save('features_X.npy', X)
    np.save('features_y.npy', y)
    print("✓ Saved: features_X.npy, features_y.npy")
    
    # ============ STEP 3: Train/Val Split ============
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Split: {len(X_train)} train, {len(X_val)} val")
    
    # ============ STEP 4: Scale Features ============
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    joblib.dump(scaler, 'scaler.joblib')
    print("✓ Saved: scaler.joblib")
    
    # ============ STEP 5: Train SVM ============
    print("\n🚀 Training SVM classifier...")
    
    # Option A: Quick training with good defaults
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',  # Handles class imbalance!
        probability=True,  # For confidence scores
        random_state=42
    )
    
    svm.fit(X_train_scaled, y_train)
    print("✓ SVM trained!")
    
    # ============ STEP 6: Evaluate ============
    print("\n📊 Evaluating...")
    y_train_pred = svm.predict(X_train_scaled)
    y_val_pred = svm.predict(X_val_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print(f"\n{'='*50}")
    print(f"Train Accuracy: {train_acc*100:.1f}%")
    print(f"Val Accuracy:   {val_acc*100:.1f}%")
    print(f"{'='*50}")
    
    print("\n📋 Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, 
                              target_names=['Real', 'Deepfake', 'AI-Gen']))
    
    # ============ STEP 7: Save Model ============
    joblib.dump(svm, 'svm_model.joblib')
    print("\n✅ Saved: svm_model.joblib")
    
    # Save training history
    history = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'feature_dim': X.shape[1],
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'svm_params': svm.get_params()
    }
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print("✅ Saved: training_history.json")
    
    print("\n" + "="*50)
    print("🎉 Training complete!")
    print("="*50)
    print("\nFiles saved:")
    print("  - svm_model.joblib    (trained SVM)")
    print("  - scaler.joblib       (feature scaler)")
    print("  - features_X.npy      (extracted features)")
    print("  - features_y.npy      (labels)")
    print("  - training_history.json")


if __name__ == '__main__':
    main()
