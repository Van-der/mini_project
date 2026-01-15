from datasets import load_dataset
from PIL import Image
import os

repo_id = "nyakura/AI_Human_Face_Detection"

# Load in streaming mode
dataset = load_dataset(repo_id, split="train", streaming=True)

# Create folders
os.makedirs("downloaded_real", exist_ok=True)
os.makedirs("downloaded_ai", exist_ok=True)

real_count = 0
ai_count = 0

# Iterate and save first 50 from each class
for row in dataset:
    # Print first row structure to debug (run once, then comment out)
    if real_count == 0 and ai_count == 0:
        print("Dataset keys:", row.keys())
        print("Sample label:", row["label"])
    
    label = row["label"]  # 0=ai, 1=real based on your screenshot
    
    # FIXED: row["image"] is already PIL Image, not a path/bytes
    image = row["image"]  
    
    if label == 1 and real_count < 50:  # 1=real
        image.save(f"downloaded_real/real_{real_count:03d}.jpg")
        real_count += 1
        print(f"Saved real {real_count}")
    elif label == 0 and ai_count < 50:  # 0=ai
        image.save(f"downloaded_ai/ai_{ai_count:03d}.jpg")
        ai_count += 1
        print(f"Saved ai {ai_count}")
    
    if real_count >= 50 and ai_count >= 50:
        break

print("Downloaded 50 real + 50 AI images!")
