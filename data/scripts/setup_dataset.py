import shutil
from pathlib import Path

base = Path(".")

DFGC_REAL = base / "real_fulls"
DFGC_DEEPFAKE = base / "fake_baseline"
NYAKURA_REAL = base / "downloaded_real"
NYAKURA_AI = base / "downloaded_ai"

FINAL_REAL = base / "final_dataset" / "real"
FINAL_DEEPFAKE = base / "final_dataset" / "deepfake"
FINAL_AI_GEN = base / "final_dataset" / "ai_gen"

print("=== COPYING & RENAMING ===\n")

# 1. DEEPFAKE: 250 from fake_baseline (no rename)
deepfake_files = list(DFGC_DEEPFAKE.glob("*"))[:250]
print(f"Copying {len(deepfake_files)} → deepfake/...")
for f in deepfake_files:
    shutil.copy(f, FINAL_DEEPFAKE)

# 2. REAL: 250 from real_fulls + 50 from nyakura (renamed)
real_files = list(DFGC_REAL.glob("*"))[:250]
print(f"Copying {len(real_files)} → real/...")
for f in real_files:
    shutil.copy(f, FINAL_REAL)

nyakura_real_files = list(NYAKURA_REAL.glob("*"))[:50]
print(f"Copying {len(nyakura_real_files)} → real/ (as aigen_*.jpg)...")
for idx, f in enumerate(nyakura_real_files):
    dst = FINAL_REAL / f"aigen_real_{idx:05d}.jpg"
    shutil.copy(f, dst)

# 3. AI_GEN: 50 from nyakura (renamed)
nyakura_ai_files = list(NYAKURA_AI.glob("*"))[:50]
print(f"Copying {len(nyakura_ai_files)} → ai_gen/ (as aigen_*.jpg)...")
for idx, f in enumerate(nyakura_ai_files):
    dst = FINAL_AI_GEN / f"aigen_ai_{idx:05d}.jpg"
    shutil.copy(f, dst)

# Verify
print("\n✅ DONE!\n")
real = len(list(FINAL_REAL.glob("*")))
deepfake = len(list(FINAL_DEEPFAKE.glob("*")))
ai_gen = len(list(FINAL_AI_GEN.glob("*")))
aigen_real = len(list(FINAL_REAL.glob("aigen_*")))
aigen_ai = len(list(FINAL_AI_GEN.glob("aigen_*")))

print(f"real/: {real} files (including {aigen_real} aigen_*)")
print(f"deepfake/: {deepfake} files")
print(f"ai_gen/: {ai_gen} files (all aigen_*)")
print(f"\nTotal: {real + deepfake + ai_gen} images ✓")
