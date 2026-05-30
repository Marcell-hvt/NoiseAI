from PIL import Image
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

img = Image.open("kep.jpg").convert("RGB")   # ← írd át a fájlnévre
arr = np.array(img, dtype=np.float32)

noise = np.random.normal(0, 50, arr.shape)   # 25 = ugyanaz mint tanításkor
noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)

Image.fromarray(noisy).save("kep_zajos.jpg")
print("Kész! -> kep_zajos.jpg")