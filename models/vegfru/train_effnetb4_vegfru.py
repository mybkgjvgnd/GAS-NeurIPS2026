#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# =========================
# 基础配置
# =========================

DATA_ROOT = "/mnt/ljh-21/organized_data_simple/vegfru/vegfru_split"
NUM_CLASSES = 292
BATCH_SIZE = 64
NUM_EPOCHS = 60
LEARNING_RATE = 0.01
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# 数据增强
# =========================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# 加载数据
# =========================

train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"),
                                     transform=train_transform)

val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),
                                   transform=val_transform)

test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),
                                    transform=val_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          pin_memory=True)

val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=True)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS,
                         pin_memory=True)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))

# =========================
# 构建模型
# =========================

model = models.efficientnet_b4(
    weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# =========================
# 损失 & 优化器
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)

scaler = GradScaler()

# =========================
# 训练函数
# =========================

def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, 1)
        correct += torch.sum(preds == labels)

    epoch_loss = total_loss / len(train_dataset)
    epoch_acc = correct.double().item() / len(train_dataset)

    return epoch_loss, epoch_acc


def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, 1)
            correct += torch.sum(preds == labels)

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = correct.double().item() / len(loader.dataset)

    return epoch_loss, epoch_acc


# =========================
# 开始训练
# =========================

best_val_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate(val_loader)

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "efficientnet_b4_vegfru_best.pth")

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Time: {time.time()-start_time:.1f}s")

print("\n✅ Training Finished")
print("Best Val Acc:", best_val_acc)

# =========================
# 测试
# =========================

model.load_state_dict(best_weights)

test_loss, test_acc = evaluate(test_loader)

print("\n🎯 Test Top-1 Accuracy:", test_acc)

# =========================
# 计算 Top-3 / Top-5
# =========================

model.eval()
correct_top3 = 0
correct_top5 = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        top5_probs, top5_preds = torch.topk(probs, k=5, dim=1)

        correct_top3 += (top5_preds[:, :3] == labels.unsqueeze(1)).any(dim=1).sum().item()
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

print("🎯 Test Top-3 Accuracy:", correct_top3 / total)
print("🎯 Test Top-5 Accuracy:", correct_top5 / total)