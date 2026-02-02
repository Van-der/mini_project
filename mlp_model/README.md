# MLP-Based Deepfake Detection (Original Model)

This folder contains the **end-to-end deep learning approach** using:
- **EfficientNet-B0** for RGB feature extraction (fine-tuned)
- **FFT CNN** for frequency domain analysis (trained)
- **MLP** for classification (trained)

## 🏗️ Architecture

```
Input Image (224×224×3)
         │
         ├──────────────────┐
         │                  │
    [RGB Branch]       [FFT Branch]
         │                  │
  EfficientNet-B0     2D FFT Magnitude
   (fine-tuned)          → CNN (trained)
         │                  │
   1280-dim features    64-dim features
         │                  │
         └────── Concat ────┘
                   │
            1344-dim vector
                   │
           ┌──────────────┐
           │  MLP Head     │
           │ 1344→128→3   │
           └──────────────┘
                   │
         3-class softmax output
```

## 📁 Files

| File | Description |
|------|-------------|
| `train_mlp.py` | End-to-end training script |
| `evaluate_mlp.py` | Full evaluation with confusion matrix |
| `gradcam_mlp.py` | Grad-CAM visualization |
| `best_model.pth` | Trained model weights |

## 🚀 How to Run

### Train the model
```bash
cd mlp_model
python train_mlp.py
```

### Evaluate
```bash
python evaluate_mlp.py
```

### Grad-CAM visualization
```bash
python gradcam_mlp.py ../dataset/cropped_dataset/ai_gen/aigen_ai_00001.jpg
```

## ⚠️ Note

This is a **pure deep learning** approach. All components (EfficientNet, FFT CNN, MLP) are trained end-to-end. 

For a **hybrid ML approach** using classical machine learning (SVM), see the `svm_model/` folder.
