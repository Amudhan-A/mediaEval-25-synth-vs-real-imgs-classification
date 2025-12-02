
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from data_loader import MediaEvalDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau



# 🔹 Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 🔹 Hyperparameters
num_epochs = 5
learning_rate = 1e-4

# 🔹 Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



train_dataset = MediaEvalDataset("data/train", transform=train_transform, mask_transform=None, use_masks=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


val_dataset = MediaEvalDataset("data/val", transform=val_transform, mask_transform=None, use_masks=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)





# 🔹 Model
model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)  # binary classification
model = model.to(device)

# 🔹 Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor= 0.1, patience = 2)


best_val_loss = float("inf")
patience = 3
counter = 0

# 🔹 Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    print("Epoch {epoch + 1} started...")

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)
    
    # 🔹 Validation
    model.eval()
    val_correct, val_total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    if (val_loss < best_val_loss):
        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "classification_models/efficientnetb0_taskB.pth")
        print("Best model saved to classification_models/efficientnetb0_taskB.pth")
    else:
        counter+= 1
        if (patience <= counter) : 
            print("Early stopping: No improvement")
            break 


    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


