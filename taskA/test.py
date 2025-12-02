import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os, csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.files = sorted(os.listdir(img_dir))  # keep file names consistent

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return fname, img

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

test_dir = "taska_test/"
test_dataset = TestDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)



if __name__ == "__main__":
    # Recreate model exactly as training
    model = models.densenet121(weights=None)  # use weights=None instead of pretrained=False
    model.classifier = nn.Linear(model.classifier.in_features, 2)


    # Load weights (match filename from training)
    model.load_state_dict(torch.load("densenet121_best_weights.pth", map_location=device))

    model.to(device)
    model.eval()

    results = []
    threshold = 0.5
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for fnames, imgs in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)   # shape: (batch_size, 2)
            probs = softmax(outputs)[:, 1].cpu().numpy()  # probability of class 1

            for fname, prob in zip(fnames, probs):
                label = 1 if prob >= threshold else 0
                results.append([fname, float(prob), label, threshold])

    # Save to CSV
    with open("bitStop_constrained.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "prob", "label", "threshold"])
        writer.writerows(results)

    print("Predictions saved to bitStop_constrained.csv")
