from PIL import Image
import os

INPUT  = "data/train"
OUTPUT = "data/train_720p"
os.makedirs(OUTPUT, exist_ok=True)

files = [f for f in os.listdir(INPUT) if f.lower().endswith((".png",".jpg",".jpeg"))]

for i, fname in enumerate(files):
    img = Image.open(os.path.join(INPUT, fname)).convert("RGB")
    img = img.resize((1280, 720), Image.LANCZOS)
    img.save(os.path.join(OUTPUT, fname), quality=95)
    if (i+1) % 50 == 0:
        print(f"{i+1}/{len(files)} kész...")

print("Minden kép átméretezve!")