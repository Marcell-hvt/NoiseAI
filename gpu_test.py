import torch
import time

device = torch.device("cuda")
model = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)

print("GPU teszt indul...")
for i in range(10):
    x = torch.randn(4, 3, 256, 256).to(device)
    y = model(x)
    print(f"Batch {i+1}: {y.shape} - GPU: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

print("GPU rendben működik!")