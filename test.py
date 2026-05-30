from PIL import Image
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

folder = "data/train"
for f in os.listdir(folder):
    path = os.path.join(folder, f)
    try:
        Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Hibás: {f} → {e}")
print("Kész!")