import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np


class BalancedFaceDataset(Dataset):
    def __init__(self, data_dir="data/cropped_dataset", mode='train'):
        """
        mode='train' : applies per-class augmentation (for MLP training)
        mode='eval'  : no augmentation — deterministic resize+normalize only
                       (use this for SVM feature extraction)
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.samples = []

        label_map = {"real": 0, "deepfake": 1, "ai_gen": 2}

        for label_name, label_id in label_map.items():
            folder_path = self.data_dir / label_name
            if not folder_path.exists():
                print(f"Folder not found: {folder_path}")
                continue

            for img_path in folder_path.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                    self.samples.append((str(img_path), label_id, img_path.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, filename = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.mode == 'eval':
            transform = self.get_inference_transform()
        elif filename.startswith("aigen_"):
            transform = self.get_heavy_transform()
        else:
            transform = self.get_light_transform()

        image = transform(image)
        return image, label

    @staticmethod
    def get_inference_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_light_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_heavy_transform():
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


def get_transforms(train=True):
    """Albumentations pipeline for CLI scripts (predict.py, gradcam.py)"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


if __name__ == '__main__':
    dataset = BalancedFaceDataset()
    real_count     = sum(1 for _, label, _ in dataset.samples if label == 0)
    deepfake_count = sum(1 for _, label, _ in dataset.samples if label == 1)
    ai_gen_count   = sum(1 for _, label, _ in dataset.samples if label == 2)
    print(f"Total: {len(dataset)}  |  real={real_count}  deepfake={deepfake_count}  ai_gen={ai_gen_count}")
