import os
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# =========================
# 参数（核心）
# =========================

MARGIN_THRESHOLD = 0.05   # 🔥 关键：margin-based 触发阈值
QWEN_CONF_THRESHOLD = 50
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 加载数据
# =========================

resnet_probs = np.load("food101_resnet_probs.npy")
eff_probs    = np.load("food101_efficientnet_probs.npy")
vit_probs    = np.load("food101_vit_probs.npy")
labels       = np.load("food101_test_labels.npy")

ensemble_probs = (resnet_probs + eff_probs + vit_probs) / 3
baseline_preds = np.argmax(ensemble_probs, axis=1)

# 加载图像路径
from torchvision.datasets import ImageFolder
dataset = ImageFolder("food101_split/Test")
class_names = dataset.classes
image_paths = [dataset.samples[i][0] for i in range(len(dataset))]

baseline_acc = (baseline_preds == labels).mean()
print("Baseline Accuracy:", baseline_acc)

# =========================
# 加载 Qwen
# =========================

model_path = "/mnt/ljh-21/ljh/Qwen/Qwen2.5-VL-7B-Instruct"

print("Loading Qwen...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_path)

# =========================
# VLM 调用
# =========================

def call_qwen(image_path, candidates):

    image = Image.open(image_path).convert("RGB")

    candidate_text = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(candidates)]
    )

    prompt = f"""
你是一位专业食物分类专家。
请从以下候选类别中选择最匹配图像的类别：

{candidate_text}

请进行视觉分析后输出 JSON：
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
        conf = int(data.get("confidence",0))
        return ranking, conf
    except:
        return candidates, 0

# =========================
# Margin-based Selective GAS
# =========================

final_preds = baseline_preds.copy()

trigger = 0
correction = 0
degradation = 0

for i in tqdm(range(len(labels))):

    probs = ensemble_probs[i]

    sorted_idx = np.argsort(probs)[::-1]
    top1 = probs[sorted_idx[0]]
    top2 = probs[sorted_idx[1]]

    margin = top1 - top2

    if margin >= MARGIN_THRESHOLD:
        continue

    trigger += 1

    topk_indices = sorted_idx[:TOP_K]
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

# =========================
# 统计结果
# =========================

final_acc = (final_preds == labels).mean()

print("\n===== Margin-based GAS =====")
print("Baseline Accuracy:", baseline_acc)
print("Final Accuracy:", final_acc)
print("Trigger Rate:", trigger / len(labels))
print("Correction:", correction)
print("Degradation:", degradation)
print("Net Gain:", correction - degradation)
print("Total Gain:", final_acc - baseline_acc)