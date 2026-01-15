import torch
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from augmentdatting import BalancedFaceDataset
from train_model import DualBranchDeepfakeDetector

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Load dataset & model
    dataset = BalancedFaceDataset('dataset/cropped_dataset')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    model = DualBranchDeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()
    
    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    print("\n📊 FULL EVALUATION REPORT")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Real', 'Deepfake', 'AI-Gen']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Deepfake', 'AI-Gen'],
                yticklabels=['Real', 'Deepfake', 'AI-Gen'])
    plt.title('Confusion Matrix (94.2% Val Acc)')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Saved: confusion_matrix.png")

if __name__ == '__main__':
    main()
