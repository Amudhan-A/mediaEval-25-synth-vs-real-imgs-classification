# Run this first, loading and processing datasets

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim



print("hello")
# Transformations (resize + convert to tensor + normalize) Data augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # random flip to left or right
    transforms.RandomRotation(10),  # rotate randomly +- 10 deg
    transforms.ColorJitter(  # slightly change brightness/contrast/saturation
        brightness = 0.2,
        contrast = 0.2,
        saturation = 0.2,
        hue = 0.1
    ),

    transforms.ToTensor(),  # converts [0,255] to [0,1] automatically ( [H,W,C] to [C,H,W])
])

val_transform = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
])

# Custom Dataset  (doing ts so that we dont load the dataset all at once)

class RealFakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        # Walk over categories
        for category in os.listdir(root_dir):
            cat_path = os.path.join(root_dir, category)
            if not os.path.isdir(cat_path):
                continue

            for label_folder, label in [("0_real", 0), ("1_fake", 1)]:
                folder_path = os.path.join(cat_path, label_folder)
                if not os.path.exists(folder_path):
                    continue

                for fname in os.listdir(folder_path):
                    if fname.endswith((".jpg", ".png", ".jpeg")):
                        self.img_paths.append(os.path.join(folder_path, fname))
                        self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")  # Loads one image from disk at index idx.
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Create dataset and DataLoader
train_dir = "ExtractedDataset/train_raw/progan"
train_dataset = RealFakeDataset(train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)



# Check one batch
images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
print("Labels:", labels)



# Path to validation dataset
val_dir = "/ExtractedDataset/val_raw/progan"  # change if needed

# Create validation dataset & loader (same transform but NO random augmentations)
val_dataset = RealFakeDataset(val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Quick check
images, labels = next(iter(val_loader))
print("Validation batch shape:", images.shape)
print("Validation labels:", labels)



# RESNET-18

model = models.resnet18(pretrained = True)  # pretrained so the weights are already kinda adjusted
                                            # will be able to intrepret basic img features like edges,shapes and textures

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2) #(inpt_size, output_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

model = model.to(device )   # move all the models weights to cpu or gpu


# the model basically has 512 inuput neurons and outputs 2 neurons (from 1000)

print(model)


criterion = nn.CrossEntropyLoss() # loss function

optimizer = optim.Adam(model.parameters(), lr = 1e-4) # actually updates weights ( with gradient descent )



num_epochs = 5 #  increase later if needed

for epoch in range(num_epochs):

    model.train()  # set model to training mode
    running_loss = 0.0
    correct, total= 0,0

    for images, labels in train_loader: # loop for one batch
        images, labels = images.to(device), labels.to(device) # moves to GPU or CPU

        optimizer.zero_grad()        # clear gradients
        outputs = model(images)      # forward pass ( feed the current batch to the model)
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()              # backward pass  (gradient descent happens here)
        optimizer.step()             # update weights

        running_loss += loss.item() # for tracking how the model is improving

        # checking training accuracy
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss  = running_loss/ len(train_loader)
    train_acc = 100 * correct / total

    # VALIDATION run
    model.eval()  # set model to evaluation mode
    val_loss = 0.0
    val_correct, val_total = 0, 0

    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # get the predicted labels from output
        _, predicted = torch.max(outputs, 1)
        val_total += labels.size(0) # count total samples
        val_correct += (predicted == labels).sum().item() # get how much was actually correct

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    




#DENSENET -121
model = models.densenet121(pretrained = True)

num_features = model.classifier.in_features # last layer features
model.classifier = nn.Linear(num_features, 2) # 2 classes , real or fake
model = model.to(device)

num_epochs = 5  # you can increase later

for epoch in range(num_epochs):

    model.train()  # set model to training mode
    running_loss = 0.0
    correct, total= 0,0

    for images, labels in train_loader: # loop for one batch
        images, labels = images.to(device), labels.to(device) # moves to GPU or CPU

        optimizer.zero_grad()        # clear gradients
        outputs = model(images)      # forward pass ( feed the current batch to the model)
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()              # backward pass  (gradient descent happens here)
        optimizer.step()             # update weights

        running_loss += loss.item() # for tracking how the model is improving

        # checking training accuracy
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss  = running_loss/ len(train_loader)
    train_acc = 100 * correct / total

    # VALIDATION run
    model.eval()  # set model to evaluation mode
    val_loss = 0.0
    val_correct, val_total = 0, 0

    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        # get the predicted labels from output
        _, predicted = torch.max(outputs, 1)
        val_total += labels.size(0) # count total samples
        val_correct += (predicted == labels).sum().item() # get how much was actually correct

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    



# RESNET - 34

model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  # pretrained ImageNet
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)  # 2 classes: real/fake
model = model.to(device)


num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
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

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")