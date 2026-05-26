import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features  # 512 na ResNet18
model.fc = nn.Linear(num_features, 37)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

print(f"Device: {device}")
print(f"Última camada: {model.fc}")