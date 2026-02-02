# SVM-Based Deepfake Detection

This folder contains the **hybrid ML approach** using:
- **Deep Learning** for feature extraction (frozen EfficientNet + raw FFT)
- **Classical ML (SVM)** for classification

## 🏗️ Architecture

```
Input Image (224×224×3)
         │
         ├──────────────────┐
         │                  │
    [RGB Branch]       [FFT Branch]
         │                  │
  EfficientNet-B0     2D FFT Magnitude
   (FROZEN)           (raw features)
         │                  │
   1280-dim features    64-dim features
         │                  │
         └────── Concat ────┘
                   │
            1344-dim vector
                   │
           ┌──────────────┐
           │ StandardScaler │
           └──────────────┘
                   │
           ┌──────────────┐
           │   SVM (RBF)   │  ← Only this is trained!
           └──────────────┘
                   │
         3-class prediction
```

## 📊 Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 97.2% |
| Real F1-Score | 0.96 |
| Deepfake F1-Score | 0.97 |
| AI-Generated F1-Score | 0.99 |

## 📁 Files

| File | Description |
|------|-------------|
| `train_svm.py` | Feature extraction + SVM training |
| `evaluate_svm.py` | Full evaluation with confusion matrix |
| `predict_svm.py` | Single image prediction with visualization |
| `svm_model.joblib` | Trained SVM classifier |
| `scaler.joblib` | Feature scaler (StandardScaler) |
| `feature_extractor.pth` | EfficientNet weights |
| `features_X.npy` | Extracted features (1344-dim) |
| `features_y.npy` | Labels |

## 🚀 How to Run

### Train the model
```bash
cd svm_model
python train_svm.py
```

### Evaluate
```bash
python evaluate_svm.py
```

### Predict on new image
```bash
python predict_svm.py ../dataset/cropped_dataset/real/some_image.jpg
```

## 🔬 Why SVM?

| Advantage | Explanation |
|-----------|-------------|
| ✅ Classical ML | Satisfies ML course requirements |
| ✅ Small dataset friendly | Works great with 800 samples |
| ✅ Fast training | Trains in seconds vs. minutes for neural nets |
| ✅ Less overfitting | Fewer parameters than MLP |
| ✅ Interpretable | Maximum margin classifier with theory |

## 📊 Dataset

| Class | Count |
|-------|-------|
| Real | 300 |
| Deepfake | 250 |
| AI-Generated | 250 |
| **Total** | **800** |
