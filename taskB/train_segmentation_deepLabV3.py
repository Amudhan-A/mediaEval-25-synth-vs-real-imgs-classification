# train_segmentation.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from segmentation_data_loaders import train_loader, val_loader  
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

from torchvision.models.segmentation import deeplabv3_resnet50

num_classes = 1  # binary segmentation
model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
model = model.to(device)

# ---------------- Loss & Optimizer ----------------
# BCE + Dice loss is common for masks
bce_loss = nn.BCEWithLogitsLoss()  # use logits for numerical stability
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# ---------------- Dice Loss ----------------
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # convert logits into prob using sigmoid func
    pred_flat = pred.view(-1)   # flatten it into 1d arr
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()  # intersection basically tells us how good of an overlap there is
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# ---------------- Training Loop ----------------
num_epochs = 10

patience = 5  # stop if no improvement for 5 epochs
counter = 0   # counts consecutive non-improvements
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']  # [B, 1, H, W]
        loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

    # Adjust learning rate
    scheduler.step(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "segmentation_models/best_model.pth")
        counter = 0
        print("Saved best model.")
    else :
        counter += 1
        if (counter >= patience):
            print("Early stopping triggered...")
            break
print("Training completed.")
