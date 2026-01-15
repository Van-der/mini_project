import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import json
from tqdm import tqdm
import os
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt

# Your existing imports + dataset
from augmentdatting import BalancedFaceDataset  # Your dataset class

class DualBranchDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # RGB Branch
        self.rgb_backbone = EfficientNet.from_pretrained('efficientnet-b0')
        rgb_features = self.rgb_backbone._fc.in_features
        self.rgb_backbone._fc = nn.Identity()
        
        # FFT Branch
        self.fft_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(rgb_features + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def fft_transform(self, x):
        # x: [B,3,H,W] -> [B,1,H,W]
        fft = torch.fft.fft2(x[:,0,:,:].unsqueeze(1))  # Use first channel
        magnitude = torch.abs(fft)
        return magnitude  # [B,1,H,W] ✓

    
    def forward(self, x):
        # RGB: [B,3,H,W] -> [B,1280]
        rgb_features = self.rgb_backbone(x)
        
        # FFT: [B,3,H,W] -> [B,64]
        fft_features = self.fft_conv(self.fft_transform(x))
        fft_features = fft_features.view(fft_features.size(0), -1)
        
        # Fuse: [B,1280+64] -> [B,3]
        combined = torch.cat([rgb_features, fft_features], dim=1)
        return self.classifier(combined)


# Training config
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Dataset (your existing class)
    dataset = BalancedFaceDataset('dataset/cropped_dataset')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Class weights (real:300, deepfake:250, aigen:50)
    class_weights = torch.tensor([1/300, 1/250, 1/50]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = DualBranchDeepfakeDetector().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss, train_correct = 0, 0
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for imgs, labels in train_iter:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            
            train_iter.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*predicted.eq(labels).sum().item()/len(labels):.1f}%'
            })
        
        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
        
        # Metrics
        train_acc = 100. * train_correct / len(train_ds)
        val_acc = 100. * val_correct / len(val_ds)
        
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss/len(val_loader))
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss/len(val_loader))
        
        print(f'Epoch {epoch+1}: Train {train_acc:.1f}% | Val {val_acc:.1f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    print(f'✅ Training complete! Best val acc: {best_val_acc:.1f}%')
    print(f'📁 Saved: best_model.pth + training_history.json')

if __name__ == '__main__':
    main()
