# Multi-Branch Deepfake and AI-Generated Face Detection

A **3-class classification model** to detect and distinguish between **Real faces**, **Deepfake faces**, and **AI-Generated faces** using a dual-branch neural network combining RGB spatial features with frequency domain (FFT) analysis.

![Confusion Matrix](svm_model/confusion_matrix_svm.png)

---

## Project Overview

| Property | Value |
|----------|-------|
| **Institution** | Sree Chitra Thirunal College of Engineering (SCTCE) |
| **Course** | B.Tech CSE - AI & ML Specialization |
| **Project Type** | Mini-project |
| **Date** | February 2026 |

### Key Features
- **Multi-class classification:** Real / Deepfake / AI-Generated
- **Dual-branch architecture:** EfficientNet-B0 (RGB) + FFT frequency analysis
- **Two model variants:** SVM (classical ML) and MLP (deep learning)
- **MTCNN preprocessing:** Robust face detection and alignment
- **PCA dimensionality reduction:** 1344 -> 100 components (anti-overfitting)
- **Explainability:** Grad-CAM visualizations for model interpretability
- **Streamlit UI:** Interactive web app for image upload and analysis
- **Docker support:** Reproducible containerized environment

---

## Recent Updates

### v3.0 - Overfitting Fix + Streamlit UI (March 2026)

**Overfitting fixes:**
- Fixed feature extraction to use eval mode (no augmentation) — previously heavy random augmentation during extraction inflated AI-Gen feature diversity and biased the model
- Added PCA (1344 -> 100 components) to prevent RBF-SVM overfitting on high-dimensional features
- Reduced SVM C from 10 -> 1.0 for a wider, more generalisable decision boundary
- Replaced single 80/20 split with a proper 70/15/15 train/val/test split — the previous "97.2%" was measured on training data
- Added 5-fold cross-validation for honest generalisation estimates

**New files:**
- `inference.py` — clean backend with two independent functions: `predict_image()` and `generate_gradcam()`
- `app.py` — Streamlit web UI (image upload, confidence scores, Grad-CAM overlay)

### v2.0 - SVM Model Addition (February 2026)
- Added SVM-based classifier using frozen EfficientNet features
- Expanded AI-Gen dataset from 50 to 250 images (scraped from thispersondoesnotexist.com)
- Reorganized project into `mlp_model/` and `svm_model/` folders

---

## Architecture

```
Input Image (224x224x3)
         |
         +------------------+
         |                  |
    [RGB Branch]       [FFT Branch]
         |                  |
  EfficientNet-B0     2D FFT Magnitude
   (pretrained,        (Pooled features)
    frozen)                 |
         |            64-dim features
   1280-dim features        |
         |                  |
         +------ Concat ----+
                   |
            1344-dim vector
                   |
            [StandardScaler]
                   |
              [PCA -> 100]
                   |
         +---------+---------+
         |                   |
    [SVM Model]         [MLP Model]
   C=1.0, RBF            (PyTorch)
   (sklearn)                 |
         |            3-class output
   3-class output
```

---

## Results

### SVM Model (Recommended) — v3.0 Honest Evaluation

| Metric | Value |
|--------|-------|
| **Train Accuracy** | 92.5% |
| **Val Accuracy** | 81.7% |
| **Test Accuracy (held-out)** | 80.0% |
| **Overfit Gap (train - test)** | 12.5% |
| **5-fold CV Mean** | 81.2% +/- 2.0% |
| Real Precision / Recall | 0.80 / 0.62 |
| Deepfake Precision / Recall | 0.74 / 0.82 |
| AI-Gen Precision / Recall | 0.86 / 1.00 |

> Note: The previous v2.0 figure of 97.2% was evaluated on the full dataset including training samples and is not a valid generalisation metric.

### MLP Model (Original)
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 94.2% |
| Training Epochs | 20 |

---

## Project Structure

```
MiniProject/
|-- svm_model/                  # SVM-based classifier (Classical ML)
|   |-- train_svm.py            # Feature extraction + SVM training
|   |-- evaluate_svm.py         # Evaluation with confusion matrix
|   |-- predict_svm.py          # Single image prediction (CLI)
|   |-- gradcam_svm.py          # Grad-CAM visualization (CLI)
|   |-- inference.py            # Backend: predict_image() + generate_gradcam()
|   |-- app.py                  # Streamlit web UI
|   `-- README.md               # SVM model documentation
|
|-- mlp_model/                  # MLP-based classifier (Deep Learning)
|   |-- train_mlp.py            # End-to-end training
|   |-- evaluate_mlp.py         # Evaluation script
|   |-- gradcam_mlp.py          # Grad-CAM visualization
|   `-- README.md               # MLP model documentation
|
|-- dataset/
|   |-- cropped_dataset/        # MTCNN-preprocessed face images
|   |   |-- real/               # 300 images
|   |   |-- deepfake/           # 250 images
|   |   `-- ai_gen/             # 250 images
|   `-- download_aigen.py       # Script to download AI-gen faces
|
|-- augmentdatting.py           # Dataset class (mode='train'/'eval')
|-- Dockerfile                  # Docker container definition
|-- requirements.txt            # Python dependencies
`-- README.md                   # This file
```

---

## Dataset Composition

| Class | Count | Source |
|-------|-------|--------|
| **Real** | 300 | DFGC (250) + Nyakura (50) |
| **Deepfake** | 250 | DFGC fake_baseline |
| **AI-Generated** | 250 | Nyakura (50) + thispersondoesnotexist.com (200) |
| **Total** | **800** | Mixed sources |

**Split (v3.0):** 70% train (560) / 15% val (120) / 15% test (120) — stratified

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
pip install streamlit
```

### Option 1: Streamlit Web UI (Recommended)

```bash
cd svm_model
streamlit run app.py
```

Upload any face image in the browser. The app runs prediction and Grad-CAM simultaneously and displays:
- Predicted class and confidence score
- Per-class probability bars
- Grad-CAM heatmap overlay

### Option 2: SVM Model (CLI)

```bash
cd svm_model

# Train
python train_svm.py

# Evaluate (confusion matrix)
python evaluate_svm.py

# Predict on a single image
python predict_svm.py path/to/image.jpg

# Grad-CAM visualization
python gradcam_svm.py path/to/image.jpg
```

### Option 3: MLP Model (CLI)

```bash
cd mlp_model

# Train
python train_mlp.py

# Evaluate
python evaluate_mlp.py

# Grad-CAM visualization
python gradcam_mlp.py path/to/image.jpg
```

### Option 4: Docker

```bash
docker build -t deepfake-detector .
docker run -v ${PWD}:/app deepfake-detector python svm_model/train_svm.py
```

---

## Model Comparison

| Aspect | SVM Model (v3.0) | MLP Model |
|--------|-----------------|-----------|
| Honest Test Accuracy | **80.0%** | 94.2% (on train set) |
| Overfit Gap | 12.5% | Unknown |
| Training | ~30 sec | ~10 min |
| ML Course Compliant | Yes | No |
| GPU Required | Feature extraction only | Training |
| Web UI | Yes (Streamlit) | No |
| Interpretability | Grad-CAM + PCA | Grad-CAM |

---

## Project Status

| Phase | Status |
|-------|--------|
| Data Collection | 800 images |
| MTCNN Preprocessing | Completed |
| SVM Model (v3.0) | 80% honest test accuracy |
| Overfitting fixes | PCA + C=1.0 + eval mode + 70/15/15 split |
| MLP Model | 94.2% (train-set evaluation) |
| Evaluation | Confusion matrices |
| Grad-CAM | Both models |
| Streamlit UI | Completed |
| Documentation | Updated |

---

## Acknowledgements

### Dataset Sources
- **DFGC 2021** - Deepfake Game Competition (IJCB 2021)
- **Nyakura AI_Human_Face_Detection** - Hugging Face
- **thispersondoesnotexist.com** - AI-generated faces

### Libraries
- PyTorch, EfficientNet-PyTorch, scikit-learn
- MTCNN, Albumentations, OpenCV, Streamlit

---

## License

This project is for **educational purposes only**.

**Disclaimer:** Deepfake detection research involves sensitive data. Users should be aware of ethical implications.

---

**Last Updated:** March 2026
