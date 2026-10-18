from PIL import Image
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

arr = np.array(img, dtype=np.float32)

noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)

