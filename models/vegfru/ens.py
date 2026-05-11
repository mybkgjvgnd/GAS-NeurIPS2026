#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =========================
# 配置
# =========================

DATA_ROOT = "/mnt/ljh-21/organized_data_simple/vegfru/vegfru_split"
NUM_CLASSES = 292
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重文件名（已把 resnet50 → densenet201）
DENSENET_WEIGHT = "densenet201_vegfru_best.pth"
EFFNET_WEIGHT = "efficientnet_b4_vegfru_best.pth"
VIT_WEIGHT    = "vitb16_vegfru_best.pth"

print("Using device:", DEVICE)

# =========================
# 数据
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT,"test"),
                                    transform=transform)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=8)

# =========================
# 加载模型（已替换成 DenseNet201）
# =========================

def load_densenet():
    model = models.densenet201(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(DENSENET_WEIGHT, map_location=DEVICE))
    return model.to(DEVICE).eval()

def load_effnet():
    model=models.efficientnet_b4(weights=None)
    in_features=model.classifier[1].in_features
    model.classifier[1]=nn.Linear(in_features,NUM_CLASSES)
    model.load_state_dict(torch.load(EFFNET_WEIGHT,map_location=DEVICE))
    return model.to(DEVICE).eval()

def load_vit():
    model=models.vit_b_16(weights=None)
    in_features=model.heads.head.in_features
    model.heads.head=nn.Linear(in_features,NUM_CLASSES)
    model.load_state_dict(torch.load(VIT_WEIGHT,map_location=DEVICE))
    return model.to(DEVICE).eval()

# 加载三个模型
densenet = load_densenet()
effnet   = load_effnet()
vit     = load_vit()

# =========================
# 集成推理
# =========================

all_logits=[]
all_labels=[]

with torch.no_grad():
    for images,labels in tqdm(test_loader):
        images=images.to(DEVICE)

        out_densenet = densenet(images)
        out_eff = effnet(images)
        out_vit = vit(images)

        # 三个模型输出平均集成
        ensemble_logits = (out_densenet + out_eff + out_vit) / 3.0

        all_logits.append(ensemble_logits.cpu())
        all_labels.append(labels)

all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)

probs = torch.softmax(all_logits, dim=1)
top5_probs, top5_preds = torch.topk(probs, k=5, dim=1)
top1_preds = top5_preds[:, 0]

top1_acc = (top1_preds == all_labels).float().mean().item()
top3_acc = ((top5_preds[:, :3] == all_labels.unsqueeze(1)).any(dim=1)).float().mean().item()
top5_acc = ((top5_preds == all_labels.unsqueeze(1)).any(dim=1)).float().mean().item()

print("\n📊 FINAL TEST DATASET ACCURACY")
print("Top‑1 Accuracy:", round(top1_acc, 4))
print("Top‑3 Accuracy:", round(top3_acc, 4))
print("Top‑5 Accuracy:", round(top5_acc, 4))