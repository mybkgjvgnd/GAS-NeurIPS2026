#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 配置
# ============================================================

DATA_ROOT = "/mnt/ljh-21/organized_data_simple/vegfru/vegfru_split"

NUM_CLASSES = 292
BATCH_SIZE = 64
NUM_WORKERS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESNET_WEIGHT = "resnet50_vegfru_best.pth"
EFFNET_WEIGHT = "efficientnet_b4_vegfru_best.pth"
VIT_WEIGHT    = "vitb16_vegfru_best.pth"

# 是否对特征做 L2 normalize
# 与论文中的 normalized feature space 更一致
FEATURE_L2_NORM = True

# Ensemble 模式：
# "logit" 复现你之前 Raw Ensemble 的结果
# "prob"  对应论文里的概率平均
ENSEMBLE_MODE = "logit"

# 计算 cdist 的 chunk size，显存不足可调小
GEO_CHUNK_SIZE = 4096

print("=" * 80)
print("VegFru Multi-Model Geo-Margin Analysis")
print("=" * 80)
print("Using device:", DEVICE)
print("Feature L2 Norm:", FEATURE_L2_NORM)
print("Ensemble Mode:", ENSEMBLE_MODE)
print("=" * 80)


# ============================================================
# 数据
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"),  transform=transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
test_paths = [p for p, _ in test_dataset.samples]

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))
print("Num classes:", len(train_dataset.classes))


# ============================================================
# 模型加载
# ============================================================

def safe_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_resnet():
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(safe_load(RESNET_WEIGHT))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_effnet():
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(safe_load(EFFNET_WEIGHT))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_vit():
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(safe_load(VIT_WEIGHT))
    model = model.to(DEVICE)
    model.eval()
    return model


print("\nLoading models...")
resnet = load_resnet()
effnet = load_effnet()
vit = load_vit()
print("✅ All models loaded")


# ============================================================
# Forward with features
# ============================================================

def forward_resnet_with_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    feat = torch.flatten(x, 1)
    logits = model.fc(feat)
    return logits, feat


def forward_effnet_with_features(model, x):
    x = model.features(x)
    x = model.avgpool(x)
    feat = torch.flatten(x, 1)
    logits = model.classifier(feat)
    return logits, feat


def forward_vit_with_features(model, x):
    n = x.shape[0]

    # torchvision ViT internal processing
    x = model._process_input(x)
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = model.encoder(x)
    feat = x[:, 0]
    logits = model.heads(feat)

    return logits, feat


# ============================================================
# 特征提取
# ============================================================

@torch.inference_mode()
def extract_features_and_logits(loader, split_name="train", need_logits=False):
    print(f"\n📌 Extracting {split_name.upper()} features...")

    feats_res_list = []
    feats_eff_list = []
    feats_vit_list = []

    logits_res_list = []
    logits_eff_list = []
    logits_vit_list = []

    labels_list = []

    for images, labels in tqdm(loader):
        images = images.to(DEVICE, non_blocking=True)

        logits_res, feat_res = forward_resnet_with_features(resnet, images)
        logits_eff, feat_eff = forward_effnet_with_features(effnet, images)
        logits_vit, feat_vit = forward_vit_with_features(vit, images)

        feat_res = feat_res.float()
        feat_eff = feat_eff.float()
        feat_vit = feat_vit.float()

        if FEATURE_L2_NORM:
            feat_res = F.normalize(feat_res, dim=1)
            feat_eff = F.normalize(feat_eff, dim=1)
            feat_vit = F.normalize(feat_vit, dim=1)

        feats_res_list.append(feat_res.cpu())
        feats_eff_list.append(feat_eff.cpu())
        feats_vit_list.append(feat_vit.cpu())

        if need_logits:
            logits_res_list.append(logits_res.float().cpu())
            logits_eff_list.append(logits_eff.float().cpu())
            logits_vit_list.append(logits_vit.float().cpu())

        labels_list.append(labels.cpu())

    out = {
        "res_feat": torch.cat(feats_res_list),
        "eff_feat": torch.cat(feats_eff_list),
        "vit_feat": torch.cat(feats_vit_list),
        "labels": torch.cat(labels_list)
    }

    if need_logits:
        out["res_logits"] = torch.cat(logits_res_list)
        out["eff_logits"] = torch.cat(logits_eff_list)
        out["vit_logits"] = torch.cat(logits_vit_list)

    return out


# ============================================================
# Prototype 计算
# ============================================================

def compute_prototypes(features, labels, num_classes=NUM_CLASSES):
    feat_dim = features.size(1)
    prototypes = torch.zeros(num_classes, feat_dim)

    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = features[mask].mean(dim=0)

    if FEATURE_L2_NORM:
        prototypes = F.normalize(prototypes, dim=1)

    return prototypes


# ============================================================
# Geo Margin 计算
# ============================================================

def compute_geo_margin_chunked(features, prototypes, name="model"):
    print(f"\n📌 Computing Geo Margin: {name}")

    margins = []

    prototypes_gpu = prototypes.to(DEVICE)

    for start in tqdm(range(0, features.size(0), GEO_CHUNK_SIZE)):
        end = min(start + GEO_CHUNK_SIZE, features.size(0))

        feat = features[start:end].to(DEVICE)

        dist = torch.cdist(feat, prototypes_gpu)
        top2_dist, _ = torch.topk(dist, k=2, largest=False)

        margin = top2_dist[:, 1] - top2_dist[:, 0]
        margins.append(margin.cpu())

    return torch.cat(margins).numpy()


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


# ============================================================
# Top-k Accuracy
# ============================================================

def compute_topk_accuracy(probs, labels):
    top5_probs, top5_preds = torch.topk(probs, k=5, dim=1)

    top1 = (top5_preds[:, 0] == labels).float().mean().item()
    top3 = ((top5_preds[:, :3] == labels.unsqueeze(1)).any(dim=1)).float().mean().item()
    top5 = ((top5_preds == labels.unsqueeze(1)).any(dim=1)).float().mean().item()

    return top1, top3, top5, top5_probs, top5_preds


# ============================================================
# Selective Oracle Simulation
# ============================================================

def selective_oracle_simulation(margin, correct, ratios):
    results = []

    N = len(correct)

    for r in ratios:
        k = int(N * r)
        idx = np.argsort(margin)[:k]

        selected_errors = int((1 - correct[idx]).sum())
        selected_correct = int(correct[idx].sum())
        selected_error_rate = selected_errors / max(k, 1)

        final_correct = correct.copy()
        final_correct[idx] = 1

        final_acc = final_correct.mean()

        results.append({
            "call_ratio": r,
            "selected_samples": k,
            "selected_errors": selected_errors,
            "selected_correct": selected_correct,
            "selected_error_rate": selected_error_rate,
            "oracle_final_acc": final_acc
        })

    return results


# ============================================================
# 主流程
# ============================================================

def main():

    # -------------------------
    # 1. Train features -> prototypes
    # -------------------------
    train_data = extract_features_and_logits(
        train_loader,
        split_name="train",
        need_logits=False
    )

    train_labels = train_data["labels"]

    print("\n📌 Computing prototypes...")
    proto_res = compute_prototypes(train_data["res_feat"], train_labels)
    proto_eff = compute_prototypes(train_data["eff_feat"], train_labels)
    proto_vit = compute_prototypes(train_data["vit_feat"], train_labels)

    # -------------------------
    # 2. Test features + logits
    # -------------------------
    test_data = extract_features_and_logits(
        test_loader,
        split_name="test",
        need_logits=True
    )

    labels = test_data["labels"]

    # -------------------------
    # 3. Ensemble probabilities
    # -------------------------
    probs_res = torch.softmax(test_data["res_logits"], dim=1)
    probs_eff = torch.softmax(test_data["eff_logits"], dim=1)
    probs_vit = torch.softmax(test_data["vit_logits"], dim=1)

    if ENSEMBLE_MODE == "prob":
        ensemble_probs = (probs_res + probs_eff + probs_vit) / 3.0
    else:
        ensemble_logits = (
            test_data["res_logits"] +
            test_data["eff_logits"] +
            test_data["vit_logits"]
        ) / 3.0
        ensemble_probs = torch.softmax(ensemble_logits, dim=1)

    # -------------------------
    # 4. Ensemble Accuracy
    # -------------------------
    ens_top1, ens_top3, ens_top5, ens_top5_probs, ens_top5_preds = compute_topk_accuracy(
        ensemble_probs,
        labels
    )

    print("\n" + "=" * 80)
    print("📊 RAW ENSEMBLE TEST ACCURACY")
    print("=" * 80)
    print("Top-1 Accuracy:", round(ens_top1, 4))
    print("Top-3 Accuracy:", round(ens_top3, 4))
    print("Top-5 Accuracy:", round(ens_top5, 4))

    # -------------------------
    # 5. Individual model top-k
    # -------------------------
    res_top1, res_top3, res_top5, res_top5_probs, res_top5_preds = compute_topk_accuracy(probs_res, labels)
    eff_top1, eff_top3, eff_top5, eff_top5_probs, eff_top5_preds = compute_topk_accuracy(probs_eff, labels)
    vit_top1, vit_top3, vit_top5, vit_top5_probs, vit_top5_preds = compute_topk_accuracy(probs_vit, labels)

    print("\n📊 SINGLE MODEL TEST ACCURACY")
    print(f"ResNet50       Top-1: {res_top1:.4f} | Top-3: {res_top3:.4f} | Top-5: {res_top5:.4f}")
    print(f"EfficientNetB4 Top-1: {eff_top1:.4f} | Top-3: {eff_top3:.4f} | Top-5: {eff_top5:.4f}")
    print(f"ViT-B/16       Top-1: {vit_top1:.4f} | Top-3: {vit_top3:.4f} | Top-5: {vit_top5:.4f}")

    # -------------------------
    # 6. Confidence margin
    # -------------------------
    top2_probs, _ = torch.topk(ensemble_probs, k=2, dim=1)
    conf_margin = (top2_probs[:, 0] - top2_probs[:, 1]).numpy()

    # -------------------------
    # 7. Geo margins
    # -------------------------
    margin_res = compute_geo_margin_chunked(test_data["res_feat"], proto_res, name="ResNet50")
    margin_eff = compute_geo_margin_chunked(test_data["eff_feat"], proto_eff, name="EfficientNetB4")
    margin_vit = compute_geo_margin_chunked(test_data["vit_feat"], proto_vit, name="ViT-B/16")

    geo_margin_multi_raw = (margin_res + margin_eff + margin_vit) / 3.0
    geo_margin_multi_z = (zscore(margin_res) + zscore(margin_eff) + zscore(margin_vit)) / 3.0

    # -------------------------
    # 8. Error detection AUROC
    # -------------------------
    ensemble_top1_preds = ens_top5_preds[:, 0]
    correct = (ensemble_top1_preds == labels).numpy().astype(int)
    error = 1 - correct

    conf_auc = roc_auc_score(error, -conf_margin)

    geo_res_auc = roc_auc_score(error, -margin_res)
    geo_eff_auc = roc_auc_score(error, -margin_eff)
    geo_vit_auc = roc_auc_score(error, -margin_vit)

    geo_multi_raw_auc = roc_auc_score(error, -geo_margin_multi_raw)
    geo_multi_z_auc = roc_auc_score(error, -geo_margin_multi_z)

    print("\n" + "=" * 80)
    print("📈 ERROR DETECTION AUROC")
    print("=" * 80)
    print(f"Conf Margin AUROC        : {conf_auc:.4f}")
    print(f"Geo ResNet AUROC         : {geo_res_auc:.4f}")
    print(f"Geo EffNet AUROC         : {geo_eff_auc:.4f}")
    print(f"Geo ViT AUROC            : {geo_vit_auc:.4f}")
    print(f"Geo-Multi Raw AUROC      : {geo_multi_raw_auc:.4f}")
    print(f"Geo-Multi ZScore AUROC   : {geo_multi_z_auc:.4f}")

    # -------------------------
    # 9. Selective Oracle Simulation
    # -------------------------
    ratios = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]

    sim_conf = selective_oracle_simulation(conf_margin, correct, ratios)
    sim_geo_raw = selective_oracle_simulation(geo_margin_multi_raw, correct, ratios)
    sim_geo_z = selective_oracle_simulation(geo_margin_multi_z, correct, ratios)

    print("\n" + "=" * 80)
    print("📌 SELECTIVE ORACLE SIMULATION")
    print("=" * 80)
    print(f"{'Ratio':<8} {'Conf':<10} {'Geo-Raw':<10} {'Geo-Z':<10} "
          f"{'Conf Err%':<12} {'Geo-Z Err%':<12}")
    print("-" * 80)

    selective_rows = []

    for i, r in enumerate(ratios):
        row_conf = sim_conf[i]
        row_raw = sim_geo_raw[i]
        row_z = sim_geo_z[i]

        print(f"{r:<8.2f} "
              f"{row_conf['oracle_final_acc']:<10.4f} "
              f"{row_raw['oracle_final_acc']:<10.4f} "
              f"{row_z['oracle_final_acc']:<10.4f} "
              f"{row_conf['selected_error_rate']:<12.4f} "
              f"{row_z['selected_error_rate']:<12.4f}")

        selective_rows.append({
            "call_ratio": r,

            "conf_oracle_acc": row_conf["oracle_final_acc"],
            "conf_selected_error_rate": row_conf["selected_error_rate"],
            "conf_selected_errors": row_conf["selected_errors"],

            "geo_raw_oracle_acc": row_raw["oracle_final_acc"],
            "geo_raw_selected_error_rate": row_raw["selected_error_rate"],
            "geo_raw_selected_errors": row_raw["selected_errors"],

            "geo_z_oracle_acc": row_z["oracle_final_acc"],
            "geo_z_selected_error_rate": row_z["selected_error_rate"],
            "geo_z_selected_errors": row_z["selected_errors"],
        })

    # -------------------------
    # 10. 保存 summary
    # -------------------------
    summary = {
        "ensemble_top1": ens_top1,
        "ensemble_top3": ens_top3,
        "ensemble_top5": ens_top5,

        "resnet_top1": res_top1,
        "resnet_top3": res_top3,
        "resnet_top5": res_top5,

        "effnet_top1": eff_top1,
        "effnet_top3": eff_top3,
        "effnet_top5": eff_top5,

        "vit_top1": vit_top1,
        "vit_top3": vit_top3,
        "vit_top5": vit_top5,

        "conf_auc": conf_auc,
        "geo_res_auc": geo_res_auc,
        "geo_eff_auc": geo_eff_auc,
        "geo_vit_auc": geo_vit_auc,
        "geo_multi_raw_auc": geo_multi_raw_auc,
        "geo_multi_z_auc": geo_multi_z_auc,
    }

    pd.DataFrame([summary]).to_csv("vegfru_multi_geo_summary.csv", index=False)
    pd.DataFrame(selective_rows).to_csv("vegfru_multi_geo_selective_oracle.csv", index=False)

    print("\n✅ Saved summary:")
    print("vegfru_multi_geo_summary.csv")
    print("vegfru_multi_geo_selective_oracle.csv")

    # -------------------------
    # 11. 保存完整 CSV
    # -------------------------
    print("\n📌 Saving full prediction CSV...")

    rows = []

    for i in tqdm(range(len(labels))):
        row = {
            "image_path": test_paths[i],
            "true_label": idx_to_class[int(labels[i])],
            "baseline_correct": int(correct[i]),

            "conf_margin": float(conf_margin[i]),

            "geo_margin_resnet": float(margin_res[i]),
            "geo_margin_effnet": float(margin_eff[i]),
            "geo_margin_vit": float(margin_vit[i]),
            "geo_margin_multi_raw": float(geo_margin_multi_raw[i]),
            "geo_margin_multi_z": float(geo_margin_multi_z[i]),
        }

        # Ensemble Top-5
        for k in range(5):
            row[f"ensemble_top{k+1}_pred"] = idx_to_class[int(ens_top5_preds[i, k])]
            row[f"ensemble_top{k+1}_conf"] = float(ens_top5_probs[i, k])

        # ResNet Top-5
        for k in range(5):
            row[f"resnet_top{k+1}_pred"] = idx_to_class[int(res_top5_preds[i, k])]
            row[f"resnet_top{k+1}_conf"] = float(res_top5_probs[i, k])

        # EffNet Top-5
        for k in range(5):
            row[f"effnet_top{k+1}_pred"] = idx_to_class[int(eff_top5_preds[i, k])]
            row[f"effnet_top{k+1}_conf"] = float(eff_top5_probs[i, k])

        # ViT Top-5
        for k in range(5):
            row[f"vit_top{k+1}_pred"] = idx_to_class[int(vit_top5_preds[i, k])]
            row[f"vit_top{k+1}_conf"] = float(vit_top5_probs[i, k])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("vegfru_multi_geo_predictions.csv", index=False)

    print("✅ Saved full CSV:")
    print("vegfru_multi_geo_predictions.csv")

    # -------------------------
    # 12. 保存 tensor 文件，便于后续 VLM / GAS 使用
    # -------------------------
    torch.save({
        "labels": labels,
        "ensemble_probs": ensemble_probs,
        "ensemble_top5_preds": ens_top5_preds,
        "ensemble_top5_probs": ens_top5_probs,

        "resnet_top5_preds": res_top5_preds,
        "resnet_top5_probs": res_top5_probs,

        "effnet_top5_preds": eff_top5_preds,
        "effnet_top5_probs": eff_top5_probs,

        "vit_top5_preds": vit_top5_preds,
        "vit_top5_probs": vit_top5_probs,

        "conf_margin": torch.tensor(conf_margin),
        "geo_margin_resnet": torch.tensor(margin_res),
        "geo_margin_effnet": torch.tensor(margin_eff),
        "geo_margin_vit": torch.tensor(margin_vit),
        "geo_margin_multi_raw": torch.tensor(geo_margin_multi_raw),
        "geo_margin_multi_z": torch.tensor(geo_margin_multi_z),

        "class_to_idx": train_dataset.class_to_idx,
        "idx_to_class": idx_to_class,
        "test_paths": test_paths,
    }, "vegfru_multi_geo_data.pt")

    print("✅ Saved tensor data:")
    print("vegfru_multi_geo_data.pt")

    print("\n" + "=" * 80)
    print("✅ Multi-Model Geo Analysis Finished")
    print("=" * 80)


if __name__ == "__main__":
    main()