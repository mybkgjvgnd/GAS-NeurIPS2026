# import os
# import csv
# import argparse
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import torchvision.models as models
# import numpy as np
# from tqdm import tqdm
# from PIL import ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True


# def build_model(model_name, num_classes):
#     if model_name == "resnet50":
#         model = models.resnet50(weights=None)
#         model.fc = nn.Linear(model.fc.in_features, num_classes)

#     elif model_name == "efficientnet_b7":
#         model = models.efficientnet_b7(weights=None)
#         in_features = model.classifier[1].in_features
#         model.classifier[1] = nn.Linear(in_features, num_classes)

#     elif model_name == "vit_b_16":
#         model = models.vit_b_16(weights=None)
#         in_features = model.heads.head.in_features
#         model.heads.head = nn.Linear(in_features, num_classes)

#     else:
#         raise ValueError(f"Unknown model: {model_name}")

#     return model


# def topk_accuracy(outputs, labels, ks=(1, 3, 5)):
#     with torch.no_grad():
#         maxk = max(ks)
#         _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
#         pred = pred.t()
#         correct = pred.eq(labels.view(1, -1).expand_as(pred))

#         res = {}
#         for k in ks:
#             correct_k = correct[:k].reshape(-1).float().sum(0)
#             res[k] = (correct_k / labels.size(0)).item()
#         return res


# def main():
#     parser = argparse.ArgumentParser(description="Food-101 inference -> CSV")
#     parser.add_argument("--model_name", type=str, required=True,
#                         choices=["resnet50", "efficientnet_b7", "vit_b_16"])
#     parser.add_argument("--weight", type=str, required=True)
#     parser.add_argument("--data_root", type=str, default="/mnt/ljh-21/organized_data_simple/food101_split")
#     parser.add_argument("--out_csv", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--num_workers", type=int, default=4)
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     test_dir = os.path.join(args.data_root, "Test")
#     assert os.path.isdir(test_dir), f"Missing Test dir: {test_dir}"

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])

#     test_dataset = datasets.ImageFolder(test_dir, transform=transform)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=True
#     )

#     class_names = test_dataset.classes
#     num_classes = len(class_names)
#     print(f"Number of classes: {num_classes}")
#     print(f"Test samples: {len(test_dataset)}")

#     # build model
#     model = build_model(args.model_name, num_classes)
#     state_dict = torch.load(args.weight, map_location=device)
#     model.load_state_dict(state_dict, strict=True)
#     model = model.to(device)
#     model.eval()
#     print(f"✅ Model loaded: {args.weight}")

#     os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

#     top1_list, top3_list, top5_list = [], [], []

#     sample_paths = [s[0] for s in test_dataset.samples]

#     with open(args.out_csv, "w", newline="", encoding="utf-8-sig") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "image_path", "true_label",
#             "top1_pred", "top1_conf",
#             "top2_pred", "top2_conf",
#             "top3_pred", "top3_conf",
#             "top4_pred", "top4_conf",
#             "top5_pred", "top5_conf"
#         ])

#         for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
#             images = images.to(device)
#             labels = labels.to(device)

#             with torch.no_grad():
#                 outputs = model(images)
#                 probs = torch.softmax(outputs, dim=1)

#             accs = topk_accuracy(outputs, labels, ks=(1, 3, 5))
#             top1_list.append(accs[1])
#             top3_list.append(accs[3])
#             top5_list.append(accs[5])

#             top5_probs, top5_indices = probs.topk(5, dim=1)
#             top5_probs = top5_probs.cpu().numpy()
#             top5_indices = top5_indices.cpu().numpy()
#             labels_np = labels.cpu().numpy()

#             start = batch_idx * args.batch_size
#             end = start + images.size(0)
#             batch_paths = sample_paths[start:end]

#             for i in range(len(batch_paths)):
#                 row = [
#                     str(batch_paths[i]),
#                     class_names[labels_np[i]],

#                     class_names[top5_indices[i][0]], f"{top5_probs[i][0]:.6f}",
#                     class_names[top5_indices[i][1]], f"{top5_probs[i][1]:.6f}",
#                     class_names[top5_indices[i][2]], f"{top5_probs[i][2]:.6f}",
#                     class_names[top5_indices[i][3]], f"{top5_probs[i][3]:.6f}",
#                     class_names[top5_indices[i][4]], f"{top5_probs[i][4]:.6f}"
#                 ]
#                 writer.writerow(row)

#     print("\n" + "=" * 60)
#     print(f"Model: {args.model_name}")
#     print(f"Top-1: {np.mean(top1_list):.4f}")
#     print(f"Top-3: {np.mean(top3_list):.4f}")
#     print(f"Top-5: {np.mean(top5_list):.4f}")
#     print("=" * 60)
#     print(f"✅ CSV saved to: {args.out_csv}")


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Food-101 GAS Final Version
Calibration + Margin-based Selective + Qwen Arbitration
"""

import os
import re
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# =====================================
# 参数设置（核心超参数）
# =====================================

TEMPERATURE = 2.0             # 🔥 温度缩放
MARGIN_THRESHOLD = 0.05       # 🔥 margin 触发阈值
QWEN_CONF_THRESHOLD = 50      # Qwen 采纳阈值
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# 加载概率文件
# =====================================

resnet = np.load("food101_resnet_probs.npy")
eff    = np.load("food101_efficientnet_probs.npy")
vit    = np.load("food101_vit_probs.npy")
labels = np.load("food101_test_labels.npy")

ensemble_probs = (resnet + eff + vit) / 3

# =====================================
# Temperature Scaling
# =====================================

def apply_temperature(probs, T):
    logits = np.log(probs + 1e-12)
    logits = logits / T
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

ensemble_probs = apply_temperature(ensemble_probs, TEMPERATURE)

baseline_preds = np.argmax(ensemble_probs, axis=1)
baseline_acc = (baseline_preds == labels).mean()

print("Baseline Accuracy (after calibration):", baseline_acc)

# =====================================
# 加载图像路径
# =====================================

from torchvision.datasets import ImageFolder
dataset = ImageFolder("food101_split/Test")
class_names = dataset.classes
image_paths = [dataset.samples[i][0] for i in range(len(dataset))]

# =====================================
# 加载 Qwen
# =====================================

model_path = "/mnt/ljh-21/ljh/Qwen/Qwen2.5-VL-7B-Instruct"

print("Loading Qwen...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_path)

# =====================================
# VLM 调用函数
# =====================================

def call_qwen(image_path, candidates):

    image = Image.open(image_path).convert("RGB")

    candidate_text = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(candidates)]
    )

    prompt = f"""
你是一位专业食物图像分类专家。
请严格基于图像视觉细节判断类别。

候选类别:
{candidate_text}

输出 JSON:
{{"ranking":["类别1","类别2",...], "confidence":0-100}}
"""

    messages = [{
        "role":"user",
        "content":[
            {"type":"image","image":image},
            {"type":"text","text":prompt}
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    response = processor.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]

    try:
        data = json.loads(re.search(r"\{.*\}", response).group())
        ranking = data.get("ranking", candidates)
        conf = int(data.get("confidence", 0))
        return ranking, conf
    except:
        return candidates, 0

# =====================================
# Margin-based Selective GAS
# =====================================

final_preds = baseline_preds.copy()

trigger = 0
correction = 0
degradation = 0

sorted_idx = np.argsort(ensemble_probs, axis=1)
top1 = ensemble_probs[np.arange(len(ensemble_probs)), sorted_idx[:, -1]]
top2 = ensemble_probs[np.arange(len(ensemble_probs)), sorted_idx[:, -2]]
margin = top1 - top2

for i in tqdm(range(len(labels))):

    if margin[i] >= MARGIN_THRESHOLD:
        continue

    trigger += 1

    topk_indices = sorted_idx[i][-TOP_K:][::-1]
    candidates = [class_names[c] for c in topk_indices]

    ranking, qconf = call_qwen(image_paths[i], candidates)

    if qconf < QWEN_CONF_THRESHOLD:
        continue

    if ranking and ranking[0] in class_names:
        new_pred = class_names.index(ranking[0])
    else:
        continue

    baseline_pred = baseline_preds[i]

    if baseline_pred != labels[i] and new_pred == labels[i]:
        correction += 1

    if baseline_pred == labels[i] and new_pred != labels[i]:
        degradation += 1

    final_preds[i] = new_pred

# =====================================
# 统计结果
# =====================================

final_acc = (final_preds == labels).mean()

print("\n===== GAS Final Results =====")
print("Temperature:", TEMPERATURE)
print("Margin Threshold:", MARGIN_THRESHOLD)
print("Baseline Accuracy:", baseline_acc)
print("Final Accuracy:", final_acc)
print("Total Gain:", final_acc - baseline_acc)
print("Trigger Rate:", trigger / len(labels))
print("Correction:", correction)
print("Degradation:", degradation)
print("Net Gain:", correction - degradation)