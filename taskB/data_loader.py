import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 🔹 Custom Dataset
class MediaEvalDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None, use_masks=False):
        self.root_dir = root_dir
        self.use_masks = use_masks
        self.transform = transform
        self.mask_transform = mask_transform

        # Collect authentic + fake image paths
        self.image_paths = []
        self.labels = []

        for label, cls in enumerate(["authentic", "fake"]):
            cls_dir = os.path.join(root_dir, "images", cls)
            for root, _, files in os.walk(cls_dir):   # 🔹 walk through subdirs too
                for fname in files:
                    if fname.endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(root, fname))
                        self.labels.append(label)

        # Sort for consistency
        self.image_paths, self.labels = zip(*sorted(zip(self.image_paths, self.labels)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Load mask if needed
        if self.use_masks:
            mask_name = os.path.basename(img_path)  # assume same filename
            mask_path = os.path.join(self.root_dir, "masks", mask_name)
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            return image, mask, label

        return image, label


# 🔹 Transforms
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225]) # for normalizing the pixel vals

])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # mask will be [0,1]
])

# 🔹 Create datasets
train_dataset = MediaEvalDataset("data/train", transform=img_transform, mask_transform=mask_transform, use_masks=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


val_dataset = MediaEvalDataset("data/val", transform=img_transform, mask_transform=mask_transform, use_masks=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)





# 🔹 Example usage
for images, labels in train_loader:
    print("Image batch:", images.shape)   # [B, 3, 128, 128]
    print("Labels:", labels)              # [B]
    break
