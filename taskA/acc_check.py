import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from dataset_loader import val_loader  # your validation DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate resnet34 model
model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: real vs synthetic

# Load trained weights
model.load_state_dict(torch.load("resnet34_best_weights.pth", map_location=device))

model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()
val_loss, val_correct, val_total = 0.0, 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

val_loss /= len(val_loader)
val_acc = 100 * val_correct / val_total

print(f"ResNet34 Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
