import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MediaEvalSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None, use_masks=True):
        self.root_dir = root_dir
        self.use_masks = use_masks
        self.transform = transform
        self.mask_transform = mask_transform

        # Base folder containing fake images
        images_base = os.path.join(root_dir, "images", "fake")

        self.image_paths = []
        # Walk through all nested subfolders
        for current_root, _, files in os.walk(images_base):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(current_root, fname)
                    self.image_paths.append(full_path)

        if len(self.image_paths) == 0:
            print("Warning: No images found! Check your paths.")
        else:
            print(f"Found {len(self.image_paths)} images.")

        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.use_masks:
            # Extract the folder name under images/fake/... (e.g., 'apple')
            parts = img_path.split(os.sep)
            folder_name = parts[-2]

            # Extract numeric ID from image filename
            img_filename = os.path.basename(img_path)  # e.g., 60855_mask_bbox.png_ps_0.png
            numeric_id = img_filename.split("_")[0]

            # Build mask filename and path
            mask_filename = f"{numeric_id}_mask_512.png"
            mask_path = os.path.join(self.root_dir, "masks", folder_name, mask_filename)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            mask = Image.open(mask_path).convert("L")

            if self.mask_transform:
                mask = self.mask_transform(mask)

            return image, mask

        return image


# ---------------- Transforms ----------------
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # [0,1]
])

# ---------------- Datasets & Loaders ----------------
train_dataset = MediaEvalSegmentationDataset(
    "data/train",
    transform=img_transform,
    mask_transform=mask_transform,
    use_masks=True
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = MediaEvalSegmentationDataset(
    "data/val",
    transform=img_transform,
    mask_transform=mask_transform,
    use_masks=True
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ---------------- Example usage ----------------
for images, masks in train_loader:
    print("Image batch:", images.shape)   # [B, 3, 224, 224]
    print("Mask batch:", masks.shape)     # [B, 1, 224, 224]
    break
