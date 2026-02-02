"""
Download AI-generated faces from thispersondoesnotexist.com
Each page load gives a new random AI-generated face.
"""
import requests
import time
import os
from pathlib import Path
from tqdm import tqdm

def download_aigen_faces(num_images=200, output_dir="cropped_dataset/ai_gen"):
    """Download AI-generated faces"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Count existing aigen images to continue numbering
    existing = list(output_path.glob("aigen_*.jpg")) + list(output_path.glob("aigen_*.png"))
    start_idx = len(existing)
    print(f"Found {start_idx} existing AI-gen images")
    print(f"Downloading {num_images} new images...")
    
    url = "https://thispersondoesnotexist.com"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    downloaded = 0
    failed = 0
    
    for i in tqdm(range(num_images), desc="Downloading"):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Save image
                img_path = output_path / f"aigen_tpdne_{start_idx + downloaded:05d}.jpg"
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
            else:
                failed += 1
                print(f"\n⚠️ Failed: HTTP {response.status_code}")
            
            # Be nice to the server - wait between requests
            time.sleep(0.5)
            
        except Exception as e:
            failed += 1
            print(f"\n❌ Error: {e}")
            time.sleep(1)
    
    print(f"\n✅ Downloaded: {downloaded} images")
    print(f"❌ Failed: {failed}")
    print(f"📁 Saved to: {output_path.absolute()}")
    
    # Show new total
    total = len(list(output_path.glob("aigen_*")))
    print(f"\n📊 Total AI-Gen images now: {total}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200, help="Number of images to download")
    args = parser.parse_args()
    
    download_aigen_faces(num_images=args.count)
