import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
from mtcnn import MTCNN
import cv2
import numpy as np

class BalancedFaceDataset(Dataset):
    def __init__(self, data_dir="cropped_dataset"):

        self.data_dir = Path(data_dir)
        self.samples = []
        
        # Load all samples with their paths and labels
        # 0=real, 1=deepfake, 2=ai_gen
        label_map = {"real": 0, "deepfake": 1, "ai_gen": 2}
        
        for label_name, label_id in label_map.items():
            folder_path = self.data_dir / label_name
            if not folder_path.exists():
                print(f"⚠️  Folder not found: {folder_path}")
                continue
            
            for img_path in folder_path.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                    self.samples.append((str(img_path), label_id, img_path.name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, filename = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Check if filename starts with aigen_ → HEAVY augmentation
        if filename.startswith("aigen_"):
            transform = self.get_heavy_transform()
        else:
            transform = self.get_light_transform()
        
        image = transform(image)
        return image, label
    
    @staticmethod
    def get_light_transform():
        """Light augmentation for real + deepfake images"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_heavy_transform():
        """Heavy augmentation for aigen_* files (minority class)"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# First verify dataset structure
print("=== DATASET VERIFICATION ===\n")
dataset = BalancedFaceDataset()

# ADD THIS DEBUG
print(f"Data dir: {dataset.data_dir}")
print(f"Real folder: {dataset.data_dir / 'real'} exists? {(dataset.data_dir / 'real').exists()}")
print(f"Files in real/: {len(list((dataset.data_dir / 'real').glob('*')))}")
print(f"Files in deepfake/: {len(list((dataset.data_dir / 'deepfake').glob('*')))}")
print(f"Files in ai_gen/: {len(list((dataset.data_dir / 'ai_gen').glob('*')))}")

print(f"✓ Total samples: {len(dataset)}")


real_count = sum(1 for _, label, _ in dataset.samples if label == 0)
deepfake_count = sum(1 for _, label, _ in dataset.samples if label == 1)
ai_gen_count = sum(1 for _, label, _ in dataset.samples if label == 2)

print(f"✓ Breakdown: {real_count} real, {deepfake_count} deepfake, {ai_gen_count} ai_gen")

if len(dataset) == 0:
    print("\n❌ ERROR: No images found!")
    print("Did you run setup_dataset_final.py first?")
    exit(1)

# Now create DataLoader for augmentation preview
print("\n=== CREATING DATALOADER ===\n")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# Test: Load one batch and show augmentation is working
batch_images, batch_labels = next(iter(dataloader))
print(f"✓ Batch shape: {batch_images.shape}")
print(f"✓ Labels in batch: {batch_labels.tolist()}")
print(f"✓ Augmentation pipeline working!")

print("\n✅ Dataset ready for training!")
print("\nNext steps:")
print("  1. Import this dataset in your training script")
print("  2. Use with EfficientNet backbone")
print("  3. Add FFT features + Grad-CAM visualization")
