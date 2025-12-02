# Run this first, loading and processing datasets

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

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
train_dir = "D:/mediaEval/ExtractedDataset/train_raw/progan"
train_dataset = RealFakeDataset(train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)



# Check one batch
images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
print("Labels:", labels)



# Path to validation dataset
val_dir = "D:/mediaEval/ExtractedDataset/val_raw"  # change if needed

# Create validation dataset & loader (same transform but NO random augmentations)
val_dataset = RealFakeDataset(val_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Quick check
images, labels = next(iter(val_loader))
print("Validation batch shape:", images.shape)
print("Validation labels:", labels)

print("Total validation images:", len(val_dataset))
print("Number of 0_real:", val_dataset.labels.count(0))
print("Number of 1_fake:", val_dataset.labels.count(1))






if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory Allocated:", torch.cuda.memory_allocated(0))
    print("GPU Memory Cached:", torch.cuda.memory_reserved(0))