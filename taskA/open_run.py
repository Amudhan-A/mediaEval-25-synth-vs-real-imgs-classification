import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import pandas as pd
import zipfile

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Open test dataset (ProGAN test folder)
# -------------------------------
# Make a custom dataset to keep track of file paths
class TestDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)  # ignore label
        path = self.imgs[index][0]
        return img, path

# Root should be one level above all categories
test_root = r"D:\mediaEval\taskA\progan_testset"
test_dataset = TestDataset(root=test_root, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------------
# Load DenseNet-121
# -------------------------------
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load("densenet121_best_weights.pth", map_location=device))
model.to(device)
model.eval()

# -------------------------------
# Generate predictions
# -------------------------------
image_ids = []
probs = []

with torch.no_grad():
    for images, paths in test_loader:
        images = images.to(device)
        outputs = model(images)
        softmax_probs = nn.functional.softmax(outputs, dim=1)
        probs.extend(softmax_probs[:,1].cpu().numpy())  # probability for synthetic
        for path in paths:
            image_ids.append(os.path.basename(path))

# -------------------------------
# Prepare submission CSV
# -------------------------------
threshold = 0.5
labels_pred = [1 if p >= threshold else 0 for p in probs]

df = pd.DataFrame({
    "image_id": image_ids,
    "prob": probs,
    "label": labels_pred,
    "threshold": [threshold]*len(probs)
})

# Save CSV
csv_path = "teamname_open.csv"
df.to_csv(csv_path, index=False)

# Optional: compress to ZIP
with zipfile.ZipFile("teamname_open.zip", 'w') as zipf:
    zipf.write(csv_path, arcname=os.path.basename(csv_path))

print("Open run CSV & ZIP ready for submission!")
