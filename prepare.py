import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
 
dataset = datasets.ImageFolder(
    root="/Users/ghabrielcalado/Documents/deeplearning_pets/images",
    transform=train_transforms
)
 
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
 
val_dataset.dataset.transform = val_transforms
 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
 
print(f"Total de imagens: {len(dataset)}")
print(f"Treino: {train_size} | Validação: {val_size}")
print(f"Classes: {dataset.classes}")
