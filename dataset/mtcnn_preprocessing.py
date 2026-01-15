import cv2
import os
from pathlib import Path
from mtcnn import MTCNN
from PIL import Image
import numpy as np

# Initialize MTCNN
detector = MTCNN()

# Paths
FINAL_DATASET = Path("final_dataset")
CROPPED_DATASET = Path("cropped_dataset")

# Create output folders
for folder in ["real", "deepfake", "ai_gen"]:
    (CROPPED_DATASET / folder).mkdir(parents=True, exist_ok=True)

total_processed = 0
total_failed = 0

print("=== MTCNN FACE CROPPING ===\n")

# Process each folder
for label_folder in ["real", "deepfake", "ai_gen"]:
    src_folder = FINAL_DATASET / label_folder
    dst_folder = CROPPED_DATASET / label_folder
    
    files = list(src_folder.glob("*"))
    processed = 0
    failed = 0
    
    print(f"Processing {label_folder}/ ({len(files)} files)...")
    
    for idx, img_path in enumerate(files):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            
            # Convert BGR to RGB for MTCNN
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector.detect_faces(img_rgb)
            
            if len(faces) == 0:
                # No face detected, skip
                failed += 1
                continue
            
            # Get largest face (or first if only one)
            face = max(faces, key=lambda f: f['box'][2] * f['box'][3])  # max by area
            x, y, w, h = face['box']
            
            # Add padding and crop
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            face_crop = img[y1:y2, x1:x2]
            
            # Resize to 224x224
            face_resized = cv2.resize(face_crop, (224, 224))
            
            # Save cropped face
            output_path = dst_folder / img_path.name
            cv2.imwrite(str(output_path), face_resized)
            
            processed += 1
            
            # Progress
            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{len(files)} processed...")
        
        except Exception as e:
            failed += 1
            continue
    
    print(f"  ✓ {processed} successful, {failed} failed\n")
    total_processed += processed
    total_failed += failed

print("="*50)
print(f"✅ PREPROCESSING COMPLETE!")
print(f"Total processed: {total_processed}")
print(f"Total failed: {total_failed}")
print(f"\n📁 Cropped images saved to cropped_dataset/")
print("\nNext: Update DataLoader to use cropped_dataset/")
