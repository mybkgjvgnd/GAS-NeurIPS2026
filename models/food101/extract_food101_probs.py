import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

device = torch.device("cuda")

data_root = "food101_split"

# 模型权重路径
model_paths = {
    "resnet": "./weights_360/resnet50_best_top1_0.8400.pth",
    "efficientnet": "./weights_360/efficientnet_b7_best_top1_0.8604.pth",
    "vit": "./weights_360/vit_b_16_best_top1_0.8559.pth"
}

# 数据
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_dataset = datasets.ImageFolder(
    os.path.join(data_root,"Test"),
    transform=transform
)

test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False)

num_classes = len(test_dataset.classes)

# -----------------------------
# 加载模型函数
# -----------------------------

def load_resnet(path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features,num_classes)
    model.load_state_dict(torch.load(path))
    return model.to(device)

def load_efficientnet(path):
    model = models.efficientnet_b7(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features,num_classes)
    model.load_state_dict(torch.load(path))
    return model.to(device)

def load_vit(path):
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features,num_classes)
    model.load_state_dict(torch.load(path))
    return model.to(device)

# -----------------------------
# 提取函数
# -----------------------------

def extract_probs(model):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs,dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)

# -----------------------------
# 运行
# -----------------------------

print("Loading ResNet...")
resnet = load_resnet(model_paths["resnet"])
resnet_probs, labels = extract_probs(resnet)
np.save("food101_resnet_probs.npy",resnet_probs)

print("Loading EfficientNet...")
efficientnet = load_efficientnet(model_paths["efficientnet"])
eff_probs,_ = extract_probs(efficientnet)
np.save("food101_efficientnet_probs.npy",eff_probs)

print("Loading ViT...")
vit = load_vit(model_paths["vit"])
vit_probs,_ = extract_probs(vit)
np.save("food101_vit_probs.npy",vit_probs)

np.save("food101_test_labels.npy",labels)

print("\n✅ All probability files saved.")