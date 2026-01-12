# Multi-Branch Deepfake and AI-Generated Face Detection

**Project Title:** Multi-Branch Deepfake and AI-Generated Face Detection Using RGB and Frequency Domain Analysis

**Academic Institution:** Sree Chitra Thirunal College of Engineering (SCTE)  
**Specialization:** B.Tech CSE - Artificial Intelligence and Machine Learning  
**Project Scope:** Mini-project (Small-scale)  
**Date:** January 2026

---

## Project Overview

This project develops a **3-class classification model** to detect and distinguish between:
1. **Real faces** (authentic video frames)
2. **Deepfake faces** (face-swap synthetic videos)
3. **AI-Generated faces** (synthetic faces from diffusion models)

The model employs a **dual-branch neural network** combining RGB spatial features (EfficientNet-B0) with frequency domain analysis (2D FFT) to capture both spatial and frequency artifacts characteristic of deepfakes and AI-generated faces.

### Key Features
- ✅ **Multi-class classification:** 3-way classification (real/deepfake/AI-gen)
- ✅ **MTCNN preprocessing:** Robust face detection and alignment (100% success rate)
- ✅ **Intelligent augmentation:** Per-class light/heavy augmentation strategy
- ✅ **Dual-branch architecture:** RGB + FFT feature fusion
- ✅ **Explainability:** Grad-CAM visualizations for model interpretability
- ✅ **Class balance handling:** Class-weighted loss + strategic oversampling

---

## Dataset Composition

### Total Dataset: 600 Images (3-class)

| Class | Count | Source | Details |
|-------|-------|--------|---------|
| **Real** | 300 | DFGC (250) + Nyakura (50) | Authentic face frames |
| **Deepfake** | 250 | DFGC fake_baseline | Face-swap synthetic videos |
| **AI-Generated** | 50 | Nyakura AI-gen subset | Diffusion model outputs |
| **TOTAL** | **600** | Mixed sources | Effective 1:1:1 after augmentation |

### Data Sources

#### 1. DFGC 2021 (Deepfake Game Competition)

**Source:** IJCB 2021 International Joint Conference on Biometrics  
**Base Dataset:** Celeb-DF v2  
**Contents:**
- `real_fulls`: 1000 authentic frames from celebrity videos
- `fake_baseline`: 1000 deepfake frames created via face-swap methods
- **Sampled:** 250 real + 250 deepfake for this project

**Dataset Link:** [DFGC 2021 Competition](https://competitions.codalab.org/competitions/29583)

**Citation:**

```bibtex
@misc{peng2021dfgc,
    title={DFGC 2021: A DeepFake Game Competition},
    author={Bo Peng and Hongxing Fan and Wei Wang and Jing Dong and Yuezun Li and 
Siwei Lyu and Qi Li and Zhenan Sun and Han Chen and Baoying Chen and Yanjie Hu and 
Shenghai Luo and Junrui Huang and Yutong Yao and Boyuan Liu and Hefei Ling and 
Guosheng Zhang and Zhiliang Xu and Changtao Miao and Changlei Lu and Shan He and 
Xiaoyan Wu and Wanyi Zhuang},
    year={2021},
    eprint={2106.01217},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@inproceedings{Celeb_DF_cvpr20,
    author = {Yuezun Li and Xin Yang and Pu Sun and Honggang Qi and Siwei Lyu},
    title = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2020}
}
```

**Acknowledgement:**  
We thank the following DFGC-21 participants for sharing their created DeepFake datasets to the research community:  
*Zhiliang Xu, Quanwei Yang, Fengyuan Liu, Hang Cai, Shan He, Christian Rathgeb, Daniel Fischer, Binghao Zhao, Li Dongze.*

---

#### 2. Nyakura AI_Human_Face_Detection (Hugging Face)

**Source:** Hugging Face Datasets Hub  
**Dataset Name:** `nyakura/AI_Human_Face_Detection`  
**Contents:**
- 50 real faces (from FFHQ subset)
- 50 AI-generated faces (Flux1/SDXL diffusion models)
- Total: 100 images

**Dataset Link:** [nyakura/AI_Human_Face_Detection on Hugging Face](https://huggingface.co/datasets/nyakura/AI_Human_Face_Detection)

**Citation:**

```bibtex
@misc{nyakura2024ai_human_face_detection,
    author = {Nyakura},
    title = {AI Human Face Detection Dataset},
    howpublished = {\url{https://huggingface.co/datasets/nyakura/AI_Human_Face_Detection}},
    year = {2024},
    note = {Accessed: January 2026}
}
```

**Dataset Features:**
- Image resolution: 256×256 (minimum)
- Classes: 2 (real faces, AI-generated faces)
- Use: Extended 2-class DFGC dataset to 3-class (real/deepfake/AI-gen)
- License: As per Hugging Face community license

---

## Project Methodology

### Phase 1: Data Preparation ✅ COMPLETED

**Steps:**
1. Downloaded DFGC real_fulls (1000 images) and fake_baseline (1000 images)
2. Downloaded Nyakura dataset (100 images: 50 real + 50 AI-gen)
3. Stratified sampling:
   - Real: 250 DFGC + 50 Nyakura (total 300)
   - Deepfake: 250 DFGC fake_baseline
   - AI-Generated: 50 Nyakura AI-gen
4. File naming convention:
   - Nyakura files renamed as `aigen_*.jpg` for augmentation strategy
   - DFGC files kept as regular names

### Phase 2: Face Preprocessing ✅ COMPLETED

**MTCNN Face Detection:**
- Detected faces in all 600 images using MTCNN
- Extracted face regions with 20-pixel padding
- Standardized all faces to 224×224 resolution
- **Results:** 100% success rate (600/600 processed, 0 failures)
- **Output:** Cropped images saved to `cropped_dataset/` folder

### Phase 3: Data Augmentation Strategy ✅ COMPLETED

**Two-Tier Augmentation (Applied during training, not pre-computed):**

**Light Augmentation (Real + Deepfake: 500 images)**
- Horizontal flip (50% probability)
- Color jitter (brightness ±0.2, contrast ±0.2)
- ImageNet normalization
- Purpose: Prevent overfitting on 250-300 samples per class

**Heavy Augmentation (AI-Generated with `aigen_*` prefix: 50 images)**
- Horizontal flip (70% probability)
- Color jitter (brightness ±0.4, contrast ±0.4, saturation ±0.3)
- Random rotation (±20°)
- ResizedCrop with zoom (0.7-1.1 scale)
- Gaussian blur (σ ∈ [0.1, 2.0])
- Affine transformations (translate 10%)
- Purpose: Oversampling minority class (50 files → 250-500+ effective samples)

**Effective Training Distribution:**
- Real: 250 files × 20 epochs × 10-15 variations = 2,500-3,750 samples
- Deepfake: 250 files × 20 epochs × 10-15 variations = 2,500-3,750 samples
- AI-Gen: 50 files × 20 epochs × 100-200 variations = 1,000-10,000 samples
- **Result:** Effectively balanced 1:1:1 ratio

### Phase 4: Model Architecture (READY FOR TRAINING)

**Dual-Branch Multi-Class Detector**

**RGB Branch:**
- EfficientNet-B0 (pretrained on ImageNet)
- Output: 1280-dimensional feature vector
- Purpose: Spatial feature learning (textures, edges, blending artifacts)

**FFT Branch:**
- 2D FFT magnitude spectrum extraction
- Custom CNN (Conv→Conv→Conv→AvgPool)
- Output: 128-dimensional feature vector
- Purpose: Frequency domain artifact detection (compression, GAN patterns)

**Fusion Head:**
- Input: Concatenate [RGB (1280) + FFT (128)] = 1408 dims
- FC: 1408 → 512 → 256 → 3 logits
- Dropout (0.5, 0.3) for regularization
- Output: 3-class softmax probabilities

**Loss Function:**
- Cross-entropy with class weights
- Weight formula: w_c = 1/n_c (inverse class frequency)
- Effect: AI-gen errors 6× more expensive than real errors

### Phase 5: Training Configuration (NEXT STEP)

**Hyperparameters:**
- Batch size: 16
- Learning rate: 1×10⁻⁴ (Adam optimizer)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Epochs: 20
- Train/Val split: 80/20 (480 train, 120 val)
- Device: GPU (CUDA) or CPU

**Expected Training Time:** ~30-60 minutes

### Phase 6: Evaluation & Explainability (AFTER TRAINING)

**Metrics:**
- Per-class accuracy, precision, recall, F1-score
- Confusion matrix (misclassification patterns)
- ROC-AUC curves (one-vs-rest for each class)
- Loss curves (training vs validation)

**Grad-CAM Visualization:**
- Identify important facial regions for each class
- Highlight detection artifacts (boundaries, symmetry, etc.)
- Analyze failure cases

---

## File Structure

```
E:\Projects\VSC\MiniProject\dataset\

├── cropped_dataset/              (✅ 600 MTCNN-preprocessed faces)
│   ├── real/                     (300 images)
│   │   ├── img_00001.jpg        (DFGC real)
│   │   ├── aigen_real_00000.jpg (Nyakura real)
│   │   └── ...
│   ├── deepfake/                 (250 images)
│   │   ├── img_00001.jpg        (DFGC deepfake)
│   │   └── ...
│   └── ai_gen/                   (50 images)
│       ├── aigen_ai_00000.jpg   (Nyakura AI-gen)
│       └── ...
│
├── train_model.py                (✅ Training script)
├── augmentdatting.py             (✅ Dataset verification)
├── mtcnn_preprocessing.py        (✅ Face cropping script)
├── copy_files_renamed.py         (✅ Data organization script)
├── deepfake_report.tex           (✅ LaTeX report)
├── README.md                     (This file)
│
└── [Output files after training]
    ├── best_model.pth           (Best model weights)
    ├── training_history.json    (Loss/accuracy curves)
    └── grad_cam_*.png           (Explainability visualizations)
```

---

## How to Reproduce

### Prerequisites
```bash
pip install torch torchvision pytorch-lightning mtcnn opencv-python pillow numpy scipy scikit-learn matplotlib tqdm
```

### Step 1: Data Preparation (Already Done ✅)
```bash
python copy_files_renamed.py
```

### Step 2: MTCNN Preprocessing (Already Done ✅)
```bash
python mtcnn_preprocessing.py
```

### Step 3: Dataset Verification (Already Done ✅)
```bash
python augmentdatting.py
# Output: ✓ Total samples: 600
#         ✓ Breakdown: 300 real, 250 deepfake, 50 ai_gen
```

### Step 4: Train the Model (NEXT)
```bash
python train_model.py
# Expected: 20 epochs, ~30-60 minutes
# Output: best_model.pth, training_history.json
```

### Step 5: Generate Grad-CAM Visualizations (AFTER TRAINING)
```bash
python grad_cam_visualization.py  # (To be created)
# Output: grad_cam_real_*.png, grad_cam_deepfake_*.png, grad_cam_ai_gen_*.png
```

---

## Citation and Usage

If you use this project or dataset combination, please cite:

### Main Project
```bibtex
@misc{deepfake_detection_2026,
    author = {CSE Student, AI/ML Specialization},
    title = {Multi-Branch Deepfake and AI-Generated Face Detection Using RGB and Frequency Domain Analysis},
    school = {Sree Chitra Thirunal College of Engineering},
    year = {2026},
    month = {January},
    note = {Mini-project, B.Tech CSE}
}
```

### Datasets Used
Cite both DFGC and Nyakura as shown in the Dataset Composition section above.

### Related Work
- EfficientNet: Tan & Le (2019)
- MTCNN: Zhang et al. (2016)
- Grad-CAM: Selvaraju et al. (2017)
- Celeb-DF: Li et al. (2020)

---

## Acknowledgments

**Data Sources:**
- DFGC 2021 competition organizers and participants
- Nyakura for publishing the AI Human Face Detection dataset on Hugging Face
- Original Celeb-DF dataset creators (Li et al., 2020)

**Tools & Libraries:**
- PyTorch for deep learning framework
- MTCNN for face detection
- Hugging Face for dataset hosting
- EfficientNet pretrained models

**Institution:**
- Sree Chitra Thirunal College of Engineering, Thiruvananthapuram

---

## License

This project code is provided as-is for educational purposes. 

**Dataset Licenses:**
- DFGC 2021: Please refer to the [DFGC competition page](https://competitions.codalab.org/competitions/29583)
- Nyakura AI_Human_Face_Detection: Hugging Face community license (as specified on dataset page)

**Disclaimer:** Deepfake detection research is sensitive. Users should be aware of ethical implications and obtain proper consent before using any face data.

---

## Contact & Questions

For questions about this project:
1. Refer to the LaTeX report: `deepfake_report.tex`
2. Check dataset documentation (DFGC, Nyakura Hugging Face)
3. Review code comments in Python scripts

---

## Project Status

| Phase | Status | Completion |
|-------|--------|-----------|
| Data Assembly | ✅ Completed | 100% |
| MTCNN Preprocessing | ✅ Completed | 100% |
| Augmentation Strategy | ✅ Completed | 100% |
| Architecture Design | ✅ Completed | 100% |
| **Model Training** | 🔄 **Next Step** | 0% |
| Evaluation & Grad-CAM | ⏳ Pending | 0% |
| Final Report | ⏳ Pending | 0% |

---

**Last Updated:** January 13, 2026, 12:34 AM IST
