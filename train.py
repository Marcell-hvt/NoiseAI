import os
from pyexpat import model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNet
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# BEÁLLÍTÁSOK – csak ezt kell módosítani
# ─────────────────────────────────────────────
TRAIN_DIR   = "data/train"   # ide tedd a tiszta képeket
EPOCHS      = 500
BATCH_SIZE  = 8              # ha elfogy a VRAM, csökkentsd 2-re
PATCH_SIZE  = 256            # 256×256 patcheket tanítunk (nem egész 720p-t)
LR          = 1e-4           # tanulási ráta
SAVE_EVERY  = 5              # minden 5. epoch után elmenti a modellt
RESUME_FROM = "checkpoints/unet_epoch160.pth"  # vagy az utolsó mentett
RESUME_EPOCH = 160  # honnan folytatjuk
# ─────────────────────────────────────────────


class DenoiseDataset(Dataset):
    """
    Beolvassa a tiszta képeket, véletlen 256×256 patchet vág ki,
    majd Gauss-zajt ad hozzá → (zajos, tiszta) párokat ad vissza.
    """
    def __init__(self, folder, patch_size=256, noise_sigma=25):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.patch_size = patch_size
        self.noise_sigma = noise_sigma

        # Véletlen patch kivágás + tensor konverzió
        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),   # adatbővítés
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),               # [0,255] → [0,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            clean = self.transform(img)
            random_sigma = torch.FloatTensor(1).uniform_(5, 200).item()
            noise = torch.randn_like(clean) * (random_sigma / 255.0)
            noisy = torch.clamp(clean + noise, 0.0, 1.0)
            return noisy, clean
        except Exception:
            # Hibás kép esetén veszi a következőt
            return self.__getitem__((idx + 1) % len(self.paths))


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eszköz: {device}")

    # Adathalmaz és DataLoader
    dataset = DenoiseDataset(TRAIN_DIR, PATCH_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=6, pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=2)  
    print(f"Tanítóképek száma: {len(dataset)}")

    # Modell, veszteségfüggvény, optimalizáló
    model = UNet().to(device)
    if os.path.exists(RESUME_FROM):
        model.load_state_dict(torch.load(RESUME_FROM, map_location=device))
        print(f"Checkpoint betöltve: {RESUME_FROM}")
    else:
        print("Nincs checkpoint, nulláról indul...")
    criterion = nn.L1Loss()          # L1 élesebb képeket ad mint MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    os.makedirs("checkpoints", exist_ok=True)
    with open("adatok.txt", "w", encoding="utf-8") as fajl:
        fajl.write("")
    print("Fájl kész: adatok.txt")

    for epoch in range(RESUME_EPOCH + 1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (noisy, clean) in enumerate(loader):
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)          # háló futtatása
            loss   = criterion(output, clean)  # hiba számítása
            loss.backward()                # gradiens (PDF 40. dia!)
            optimizer.step()               # paraméter frissítés

            total_loss += loss.item()

            # Haladás kiírása
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} | "
                      f"Batch {batch_idx+1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        szoveg = f"Epoch {epoch}/{EPOCHS} befejezve | Átlag loss: {avg_loss:.4f}\n"
        print(szoveg)
        with open("adatok.txt", "a", encoding="utf-8") as fajl:
            fajl.write(szoveg)
        scheduler.step()

        # Mentés
        if epoch % SAVE_EVERY == 0:
            path = f"checkpoints/unet_epoch{epoch}.pth"
            torch.save(model.state_dict(), path)
            print(f"  → Mentve: {path}")

    # Végső mentés
    torch.save(model.state_dict(), "checkpoints/unet_final.pth")
    print("Tanítás kész! Modell elmentve: checkpoints/unet_final.pth")


if __name__ == "__main__":
    train()