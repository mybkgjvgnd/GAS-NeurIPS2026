import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

device = torch.device("cuda")

data_root = "food101_split"
model_path = "./weights_360/efficientnet_b7_best_top1_0.8604.pth"

# ---------- Data ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    os.path.join(data_root, "Training"),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(data_root, "Test"),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = len(train_dataset.classes)

# ---------- Model ----------
model = models.efficientnet_b7(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

def extract(loader):
    feats = []
    labels = []
    probs = []

    with torch.no_grad():
        for imgs, labs in tqdm(loader):
            imgs = imgs.to(device)
            outputs = model(imgs)

            prob = torch.softmax(outputs, dim=1)

            # penultimate layer feature
            feat = model.features(imgs)
            feat = model.avgpool(feat)
            feat = torch.flatten(feat, 1)

            feats.append(feat.cpu().numpy())
            labels.append(labs.numpy())
            probs.append(prob.cpu().numpy())

    return (
        np.concatenate(feats),
        np.concatenate(labels),
        np.concatenate(probs)
    )

print("Extracting train features...")
train_feats, train_labels, _ = extract(train_loader)

print("Extracting test features...")
test_feats, test_labels, test_probs = extract(test_loader)

np.save("food101_train_feats.npy", train_feats)
np.save("food101_train_labels.npy", train_labels)
np.save("food101_test_feats.npy", test_feats)
np.save("food101_test_labels.npy", test_labels)
np.save("food101_test_probs.npy", test_probs)

print("✅ Feature extraction done.")