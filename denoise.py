import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import sys
import os
from model import UNet
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# BEÁLLÍTÁSOK
# ─────────────────────────────────────────────
MODEL_PATH = "checkpoints/unet_final.pth"
PATCH_SIZE = 256   # ugyanakkora mint tanításkor
OVERLAP    = 128    # átfedés a patchek között (simább illesztés)
# ─────────────────────────────────────────────


def denoise_image(model, device, image_path, output_path):
    """
    Teljes 720p kép denoise-olása patch-alapon:
    1. Felvágjuk PATCH_SIZE×PATCH_SIZE darabokra OVERLAP átfedéssel
    2. Minden patchet külön denoise-olunk
    3. Visszaillesztjük (az átfedésnél átlagolunk → nincs látható varrat)
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_tensor = TF.to_tensor(img)  # [3, H, W], értékek [0,1]

    output = torch.zeros_like(img_tensor)
    count  = torch.zeros_like(img_tensor)  # hány patch fedte ezt a pixelt

    step = PATCH_SIZE - OVERLAP

    model.eval()
    with torch.no_grad():
        y = 0
        while y < H:
            y_end = min(y + PATCH_SIZE, H)
            y_start = y_end - PATCH_SIZE  # ha a szélen vagyunk, visszalépünk

            x = 0
            while x < W:
                x_end = min(x + PATCH_SIZE, W)
                x_start = x_end - PATCH_SIZE

                patch = img_tensor[:, y_start:y_end, x_start:x_end]
                patch = patch.unsqueeze(0).to(device)  # [1, 3, 256, 256]

                denoised_patch = model(patch).squeeze(0).cpu()

                output[:, y_start:y_end, x_start:x_end] += denoised_patch
                count[:, y_start:y_end, x_start:x_end]  += 1

                if x_end == W:
                    break
                x += step

            if y_end == H:
                break
            y += step

    # Átlagolás az átfedő területeken
    output = output / count
    output = torch.clamp(output, 0.0, 1.0)

    result = TF.to_pil_image(output)
    result.save(output_path)
    print(f"Kész! Mentve: {output_path}")


def main():

    input_path = sinput_path  = sys.argv[1] if len(sys.argv) > 1 else "kep_zajos.jpg"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "kep_tiszta.jpg"

    if not os.path.exists(input_path):
        print(f"Nem talalhato: {input_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eszkoz: {device}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Modell betoltve: {MODEL_PATH}")

    denoise_image(model, device, input_path, output_path)


if __name__ == "__main__":
    main()