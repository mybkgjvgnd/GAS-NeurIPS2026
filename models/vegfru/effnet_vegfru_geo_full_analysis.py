#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# =========================
# 基础配置
# =========================

DATA_ROOT = "/mnt/ljh-21/organized_data_simple/vegfru/vegfru_split"
WEIGHT_PATH = "efficientnet_b4_vegfru_best.pth"
NUM_CLASSES = 292
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# 数据加载
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=8, pin_memory=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=8, pin_memory=True)

idx_to_class = {v:k for k,v in train_dataset.class_to_idx.items()}

# =========================
# 构建模型
# =========================

model = models.efficientnet_b4(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 提取特征层
feature_extractor = nn.Sequential(*list(model.children())[:-1])

# =========================
# 提取 TRAIN 特征
# =========================

print("\n📌 Extracting TRAIN features...")
train_features = []
train_labels = []

with torch.no_grad():
    for images, labels in tqdm(train_loader):
        images = images.to(DEVICE)
        feats = feature_extractor(images)
        feats = feats.view(feats.size(0), -1)
        train_features.append(feats.cpu())
        train_labels.append(labels)

train_features = torch.cat(train_features)
train_labels = torch.cat(train_labels)

# =========================
# 计算 Prototype
# =========================

print("\n📌 Computing Prototypes...")
feature_dim = train_features.size(1)
prototypes = torch.zeros(NUM_CLASSES, feature_dim)

for c in range(NUM_CLASSES):
    mask = (train_labels == c)
    prototypes[c] = train_features[mask].mean(dim=0)

# =========================
# 提取 TEST 特征 + Logits
# =========================

print("\n📌 Extracting TEST features...")
test_features = []
test_logits = []
test_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        feats = feature_extractor(images)
        feats = feats.view(feats.size(0), -1)

        test_logits.append(outputs.cpu())
        test_features.append(feats.cpu())
        test_labels.append(labels)

test_logits = torch.cat(test_logits)
test_features = torch.cat(test_features)
test_labels = torch.cat(test_labels)

# =========================
# Top‑k 计算
# =========================

probs = torch.softmax(test_logits, dim=1)
top5_probs, top5_preds = torch.topk(probs, k=5, dim=1)
top1_preds = top5_preds[:,0]

top1_acc = (top1_preds == test_labels).float().mean().item()
top3_acc = ((top5_preds[:,:3] == test_labels.unsqueeze(1)).any(dim=1)).float().mean().item()
top5_acc = ((top5_preds == test_labels.unsqueeze(1)).any(dim=1)).float().mean().item()

print("\n📊 FINAL TEST DATASET ACCURACY")
print("Top‑1 Accuracy:", round(top1_acc,4))
print("Top‑3 Accuracy:", round(top3_acc,4))
print("Top‑5 Accuracy:", round(top5_acc,4))

# =========================
# Conf Margin
# =========================

top2_probs, _ = torch.topk(probs, k=2, dim=1)
conf_margin = (top2_probs[:,0] - top2_probs[:,1]).numpy()

# =========================
# Geo Margin
# =========================

print("\n📌 Computing Geo Margin...")
dist_matrix = torch.cdist(test_features, prototypes)
top2_dist, _ = torch.topk(dist_matrix, k=2, largest=False)
geo_margin = (top2_dist[:,1] - top2_dist[:,0]).numpy()

# =========================
# AUROC 计算
# =========================

error = (top1_preds != test_labels).numpy().astype(int)

geo_auc  = roc_auc_score(error, -geo_margin)
conf_auc = roc_auc_score(error, -conf_margin)

print("\n📈 ERROR DETECTION AUROC")
print("Geo AUROC :", round(geo_auc,4))
print("Conf AUROC:", round(conf_auc,4))

# =========================
# Selective Oracle Simulation
# =========================

def selective_simulation(margin, ratio):
    N = len(test_labels)
    k = int(N * ratio)
    idx = np.argsort(margin)[:k]

    final_correct = (top1_preds == test_labels).numpy().astype(int)
    final_correct[idx] = 1
    return final_correct.mean()

print("\n📌 SELECTIVE ORACLE SIMULATION")
ratios = [0.05, 0.1, 0.15, 0.2]
for r in ratios:
    geo_acc = selective_simulation(geo_margin, r)
    conf_acc = selective_simulation(conf_margin, r)
    print(f"Call Ratio {int(r*100)}% → Geo: {geo_acc:.4f}, Conf: {conf_acc:.4f}")

# =========================
# 保存 CSV
# =========================

print("\n📌 Saving CSV...")

rows = []

for i in range(len(test_labels)):
    row = {
        "true_label": idx_to_class[int(test_labels[i])]
    }
    for k in range(5):
        row[f"top{k+1}_pred"] = idx_to_class[int(top5_preds[i,k])]
        row[f"top{k+1}_conf"] = float(top5_probs[i,k])
    row["conf_margin"] = float(conf_margin[i])
    row["geo_margin"]  = float(geo_margin[i])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("effnet_vegfru_geo_full_analysis.csv", index=False)

print("✅ Saved: effnet_vegfru_geo_full_analysis.csv")