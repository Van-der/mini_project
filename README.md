# Multi-Branch Deepfake and AI-Generated Face Detection

A **3-class classification model** to detect and distinguish between **Real faces**, **Deepfake faces**, and **AI-Generated faces** using a dual-branch neural network combining RGB spatial features with frequency domain (FFT) analysis.

![Confusion Matrix](confusion_matrix.png)

---

## 📋 Project Overview

| Property | Value |
|----------|-------|
| **Institution** | Sree Chitra Thirunal College of Engineering (SCTCE) |
| **Course** | B.Tech CSE - AI & ML Specialization |
| **Project Type** | Mini-project |
| **Date** | January 2026 |

### Key Features
- ✅ **Multi-class classification:** Real / Deepfake / AI-Generated
- ✅ **Dual-branch architecture:** EfficientNet-B0 (RGB) + FFT frequency analysis
- ✅ **MTCNN preprocessing:** Robust face detection and alignment
- ✅ **Class-balanced training:** Weighted loss + smart augmentation for minority classes
- ✅ **Explainability:** Grad-CAM visualizations for model interpretability
- ✅ **Docker support:** Reproducible containerized environment

---

## 🏗️ Architecture

```
Input Image (224×224×3)
         │
         ├──────────────────┐
         │                  │
    [RGB Branch]       [FFT Branch]
         │                  │
  EfficientNet-B0     2D FFT Magnitude
   (pretrained)        → CNN layers
         │                  │
   1280-dim features    64-dim features
         │                  │
         └────── Concat ────┘
                   │
            [Fusion Head]
         FC: 1344 → 128 → 3
                   │
         3-class softmax output
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 94.2% |
| **Training Epochs** | 20 |
| **Best Val Loss** | 0.165 |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Real | High | High | High |
| Deepfake | High | High | High |
| AI-Generated | Moderate | Moderate | Moderate |

*Note: AI-Generated class has fewer samples (50) compared to Real (300) and Deepfake (250)*

---

## 📁 Project Structure

```
MiniProject/
├── train_model.py          # Main training script with dual-branch model
├── augmentdatting.py       # Dataset class with smart augmentation
├── evaluate_model.py       # Evaluation script with confusion matrix
├── gradcam_demo.py         # Grad-CAM visualization for explainability
├── best_model.pth          # Trained model weights
├── training_history.json   # Training/validation metrics per epoch
├── confusion_matrix.png    # Confusion matrix visualization
├── gradcam_result.png      # Sample Grad-CAM output
├── Dockerfile              # Docker container definition
├── requirements.txt        # Python dependencies
└── dataset/
    └── cropped_dataset/    # MTCNN-preprocessed face images
        ├── real/           # 300 images
        ├── deepfake/       # 250 images
        └── ai_gen/         # 50 images
```

---

## 📜 Script Descriptions

### `train_model.py`
**Main training script** containing:
- `DualBranchDeepfakeDetector` class - the dual-branch neural network
- RGB branch using pretrained EfficientNet-B0
- FFT branch for frequency domain feature extraction
- Training loop with class-weighted CrossEntropy loss
- Learning rate scheduling with ReduceLROnPlateau
- Saves `best_model.pth` and `training_history.json`

### `augmentdatting.py`
**Dataset and augmentation pipeline**:
- `BalancedFaceDataset` class for loading preprocessed faces
- **Light augmentation** for Real/Deepfake: horizontal flip, color jitter
- **Heavy augmentation** for AI-Generated (`aigen_*` files): rotation, blur, affine transforms
- Automatic detection based on filename prefix

### `evaluate_model.py`
**Model evaluation script**:
- Loads trained model and runs inference on full dataset
- Generates classification report (precision, recall, F1)
- Creates and saves confusion matrix as `confusion_matrix.png`

### `gradcam_demo.py`
**Explainability visualization**:
- `GradCAM` class for generating activation maps
- MTCNN face detection for preprocessing input images
- Overlays heatmap on original image to show model focus areas
- Saves output as `gradcam_result.png`

### `Dockerfile`
**Containerized environment**:
- Based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- Includes all dependencies for reproducible execution
- Supports both CPU and GPU inference

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Option 1: Direct Python Execution

**Train the model:**
```bash
python train_model.py
```

**Evaluate the model:**
```bash
python evaluate_model.py
```

**Run Grad-CAM on an image:**
```bash
python gradcam_demo.py path/to/image.jpg
```

### Option 2: Docker (Recommended)

**Build the container:**
```bash
docker build -t deepfake-detector .
```

**Train:**
```bash
docker run -v ${PWD}:/app deepfake-detector python train_model.py
```

**Evaluate:**
```bash
docker run -v ${PWD}:/app deepfake-detector python evaluate_model.py
```

**Grad-CAM inference:**
```bash
docker run -v ${PWD}:/app deepfake-detector python gradcam_demo.py "dataset/cropped_dataset/ai_gen/aigen_ai_00000.jpg"
```

---

## ✅ Project Status

| Phase | Status | Details |
|-------|--------|---------|
| Data Collection | ✅ Completed | 600 images from DFGC + Nyakura datasets |
| MTCNN Preprocessing | ✅ Completed | 100% face detection success rate |
| Augmentation Strategy | ✅ Completed | Light/heavy augmentation per class |
| Model Architecture | ✅ Completed | Dual-branch RGB + FFT fusion |
| Model Training | ✅ Completed | 94.2% validation accuracy |
| Evaluation | ✅ Completed | Confusion matrix generated |
| Grad-CAM Visualization | ✅ Completed | Explainability pipeline working |
| Docker Support | ✅ Completed | Containerized environment ready |

### 🔜 Future Improvements
- [ ] Collect more AI-Generated samples (currently only 50)
- [ ] Add video-level detection (frame aggregation)
- [ ] Implement attention mechanisms for better feature fusion
- [ ] Create web interface for demo
- [ ] Add support for newer diffusion model outputs (DALL-E 3, Midjourney v6)
- [ ] Cross-dataset evaluation (test on FaceForensics++, DFDC)

---

## 📊 Dataset Composition

| Class | Count | Source |
|-------|-------|--------|
| **Real** | 300 | DFGC (250) + Nyakura (50) |
| **Deepfake** | 250 | DFGC fake_baseline |
| **AI-Generated** | 50 | Nyakura AI-gen subset |
| **Total** | **600** | Mixed sources |

---

## 🙏 Acknowledgements

### Dataset Sources

#### DFGC 2021 (Deepfake Game Competition)
- **Source:** IJCB 2021 International Joint Conference on Biometrics
- **Base Dataset:** Celeb-DF v2
- **Link:** [DFGC 2021 Competition](https://competitions.codalab.org/competitions/29583)

```bibtex
@misc{peng2021dfgc,
    title={DFGC 2021: A DeepFake Game Competition},
    author={Bo Peng and Hongxing Fan and Wei Wang and Jing Dong and Yuezun Li and 
            Siwei Lyu and Qi Li and Zhenan Sun and Han Chen and Baoying Chen and 
            Yanjie Hu and Shenghai Luo and Junrui Huang and Yutong Yao and Boyuan Liu 
            and Hefei Ling and Guosheng Zhang and Zhiliang Xu and Changtao Miao and 
            Changlei Lu and Shan He and Xiaoyan Wu and Wanyi Zhuang},
    year={2021},
    eprint={2106.01217},
    archivePrefix={arXiv}
}
```

#### Nyakura AI_Human_Face_Detection
- **Source:** Hugging Face Datasets Hub
- **Link:** [nyakura/AI_Human_Face_Detection](https://huggingface.co/datasets/nyakura/AI_Human_Face_Detection)

```bibtex
@misc{nyakura2024ai_human_face_detection,
    author = {Nyakura},
    title = {AI Human Face Detection Dataset},
    howpublished = {\url{https://huggingface.co/datasets/nyakura/AI_Human_Face_Detection}},
    year = {2024}
}
```

### Libraries & Tools
- **PyTorch** - Deep learning framework
- **EfficientNet-PyTorch** - Pretrained backbone
- **MTCNN** - Face detection
- **Albumentations** - Image augmentation
- **Grad-CAM** - Model explainability

### Related Work
- EfficientNet: Tan & Le (2019)
- MTCNN: Zhang et al. (2016)
- Grad-CAM: Selvaraju et al. (2017)
- Celeb-DF: Li et al. (2020)

---

## 📄 License

This project is for **educational purposes only**.

- **DFGC 2021:** Refer to [competition page](https://competitions.codalab.org/competitions/29583)
- **Nyakura Dataset:** Hugging Face community license

⚠️ **Disclaimer:** Deepfake detection research involves sensitive data. Users should be aware of ethical implications and obtain proper consent before using any face data.

---

**Last Updated:** January 15, 2026
