import torch
import torch.nn as nn
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ── Egy "blokk" = két konvolúció egymás után, BatchNorm és ReLU aktiváció ──
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── U-Net architektúra ──
# Encoder: leszűkíti a képet, megtanulja a struktúrát
# Decoder: visszanagyítja, visszaállítja a részleteket
# Skip connections: az encoder minden szintjéről átad infót a decodernek
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (leszűkítés)
        self.enc1 = ConvBlock(3, 64)       # 720×1280 → 64 csatorna
        self.enc2 = ConvBlock(64, 128)     # 360×640
        self.enc3 = ConvBlock(128, 256)    # 180×320
        self.enc4 = ConvBlock(256, 512)    # 90×160

        # Bottleneck (legkisebb szint)
        self.bottleneck = ConvBlock(512, 1024)  # 45×80

        # Decoder (visszanagyítás)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)   # 512 (up) + 512 (skip) = 1024 bemenet

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Kimenet: visszaalakítás 3 csatornás (RGB) képpé
        self.out = nn.Conv2d(64, 3, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder (skip connection = torch.cat)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))  # [0, 1] tartományba szorítjuk