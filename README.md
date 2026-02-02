# Multi-Branch Deepfake and AI-Generated Face Detection

A **3-class classification model** to detect and distinguish between **Real faces**, **Deepfake faces**, and **AI-Generated faces** using a dual-branch neural network combining RGB spatial features with frequency domain (FFT) analysis.

![Confusion Matrix](svm_model/confusion_matrix_svm.png)

---

##  Project Overview

| Property | Value |
|----------|-------|
| **Institution** | Sree Chitra Thirunal College of Engineering (SCTCE) |
| **Course** | B.Tech CSE - AI & ML Specialization |
| **Project Type** | Mini-project |
| **Date** | February 2026 |

### Key Features
-  **Multi-class classification:** Real / Deepfake / AI-Generated
-  **Dual-branch architecture:** EfficientNet-B0 (RGB) + FFT frequency analysis
-  **Two model variants:** SVM (classical ML) and MLP (deep learning)
-  **MTCNN preprocessing:** Robust face detection and alignment
-  **Class-balanced training:** Weighted loss + smart augmentation
-  **Explainability:** Grad-CAM visualizations for model interpretability
-  **Docker support:** Reproducible containerized environment

---

##  Recent Updates

### v2.0 - SVM Model Addition (February 2026)
-  Added **SVM-based classifier** using frozen EfficientNet features
-  Expanded AI-Gen dataset from 50 to **250 images** (scraped from thispersondoesnotexist.com)
-  Reorganized project into `mlp_model/` and `svm_model/` folders
-  Improved AI-Gen detection (F1: 0.59  0.99)
-  Overall accuracy: **97.2%** with SVM model

---

##  Architecture

\\\
Input Image (2242243)
         
         
                           
    [RGB Branch]       [FFT Branch]
                           
  EfficientNet-B0     2D FFT Magnitude
   (pretrained)         Pooled features
                           
   1280-dim features    64-dim features
                           
          Concat 
                   
            1344-dim vector
                   
         
                          
    [SVM Model]       [MLP Model]
     (sklearn)        (PyTorch)
                          
   3-class output    3-class output
\\\

---

##  Results

### SVM Model (Recommended)
| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 97.2% |
| Real F1-Score | 0.96 |
| Deepfake F1-Score | 0.97 |
| AI-Generated F1-Score | 0.99 |

### MLP Model (Original)
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 94.2% |
| Training Epochs | 20 |

---

##  Project Structure

\\\
MiniProject/
 svm_model/              #  SVM-based classifier (Classical ML)
    train_svm.py        # Feature extraction + SVM training
    evaluate_svm.py     # Evaluation with confusion matrix
    predict_svm.py      # Single image prediction
    README.md           # SVM model documentation

 mlp_model/              #  MLP-based classifier (Deep Learning)
    train_mlp.py        # End-to-end training
    evaluate_mlp.py     # Evaluation script
    gradcam_mlp.py      # Grad-CAM visualization
    README.md           # MLP model documentation

 dataset/
    cropped_dataset/    # MTCNN-preprocessed face images
       real/           # 300 images
       deepfake/       # 250 images
       ai_gen/         # 250 images (expanded!)
    download_aigen.py   #  Script to download AI-gen faces

 augmentdatting.py       # Dataset class with augmentation
 Dockerfile              # Docker container definition
 requirements.txt        # Python dependencies
 README.md               # This file
\\\

---

##  Dataset Composition

| Class | Count | Source |
|-------|-------|--------|
| **Real** | 300 | DFGC (250) + Nyakura (50) |
| **Deepfake** | 250 | DFGC fake_baseline |
| **AI-Generated** | 250 | Nyakura (50) + thispersondoesnotexist.com (200) |
| **Total** | **800** | Mixed sources |

---

##  How to Run

### Prerequisites
\\\ash
pip install -r requirements.txt
\\\

### Option 1: SVM Model (Recommended)

\\\ash
cd svm_model

# Train
python train_svm.py

# Evaluate
python evaluate_svm.py

# Predict on new image
python predict_svm.py path/to/image.jpg
\\\

### Option 2: MLP Model

\\\ash
cd mlp_model

# Train
python train_mlp.py

# Evaluate
python evaluate_mlp.py

# Grad-CAM visualization
python gradcam_mlp.py path/to/image.jpg
\\\

### Option 3: Docker

\\\ash
docker build -t deepfake-detector .
docker run -v \E:\Projects\VSC\MiniProject\svm_model:/app deepfake-detector python svm_model/train_svm.py
\\\

---

##  Model Comparison

| Aspect | SVM Model | MLP Model |
|--------|-----------|-----------|
| Accuracy | **97.2%** | 94.2% |
| Training Time | ~30 sec | ~10 min |
| ML Course Compliant |  Yes |  No |
| GPU Required | Feature extraction only | Training |
| Interpretability | Feature-based | Grad-CAM |

---

##  Project Status

| Phase | Status |
|-------|--------|
| Data Collection |  800 images |
| MTCNN Preprocessing |  Completed |
| SVM Model |  97.2% accuracy |
| MLP Model |  94.2% accuracy |
| Evaluation |  Confusion matrices |
| Documentation |  Updated |

---

##  Acknowledgements

### Dataset Sources

- **DFGC 2021** - Deepfake Game Competition (IJCB 2021)
- **Nyakura AI_Human_Face_Detection** - Hugging Face
- **thispersondoesnotexist.com** - AI-generated faces

### Libraries
- PyTorch, EfficientNet-PyTorch, scikit-learn
- MTCNN, Albumentations, OpenCV

---

##  License

This project is for **educational purposes only**.

 **Disclaimer:** Deepfake detection research involves sensitive data. Users should be aware of ethical implications.

---

**Last Updated:** February 2, 2026
