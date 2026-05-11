import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torchvision.datasets import ImageFolder
import json
import os

# =========================
# 参数设置
# =========================

lambda_thresh = 0.6            # 低置信度阈值
qwen_conf_threshold = 50       # Qwen 采纳阈值
TOP_K = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 加载 Qwen
# =========================

model_path = "/mnt/ljh-21/ljh/Qwen/Qwen2.5-VL-7B-Instruct"

print("Loading Qwen model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_path)

# =========================
# 加载数据
# =========================

print("Loading ensemble probability files...")

resnet_probs = np.load("food101_resnet_probs.npy")
eff_probs    = np.load("food101_efficientnet_probs.npy")
vit_probs    = np.load("food101_vit_probs.npy")
labels       = np.load("food101_test_labels.npy")

ensemble_probs = (resnet_probs + eff_probs + vit_probs) / 3
baseline_preds = np.argmax(ensemble_probs, axis=1)

dataset = ImageFolder("food101_split/Test")
class_names = dataset.classes
image_paths = [dataset.samples[i][0] for i in range(len(dataset))]

baseline_acc = (baseline_preds == labels).mean()
print("Baseline Accuracy:", baseline_acc)

# =========================
# Qwen 调用函数
# =========================

def call_qwen(image_path, candidates):

    image = Image.open(image_path).convert("RGB")

    candidate_str = "\n".join([f"{i+1}. {c}" for i,c in enumerate(candidates)])

    prompt = (
        "You are a professional food classifier.\n"
        "From the following candidates, choose the best matching class.\n\n"
        f"{candidate_str}\n\n"
        "Output strictly in JSON format:\n"
        '{"ranking": ["class1","class2",...], "confidence": 0-100}'
    )

    messages = [
        {"role":"user","content":[
            {"type":"image","image":image},
            {"type":"text","text":prompt}
        ]}
    ]

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
    ).to(device)

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
        data = json.loads(response)
        ranking = data.get("ranking", candidates)
        conf = int(data.get("confidence",0))
        return ranking, conf
    except:
        return candidates, 0

# =========================
# 只处理低置信度样本
# =========================

top1_conf_all = np.max(ensemble_probs, axis=1)
low_conf_mask = top1_conf_all < lambda_thresh
low_indices = np.where(low_conf_mask)[0]

print(f"Total low-confidence samples (<{lambda_thresh}): {len(low_indices)}")

final_preds = baseline_preds.copy()

trigger = 0
correction = 0
degradation = 0

for i in tqdm(low_indices):

    trigger += 1

    probs = ensemble_probs[i]
    topk_indices = np.argsort(probs)[-TOP_K:][::-1]
    candidates = [class_names[c] for c in topk_indices]

    ranking, qconf = call_qwen(image_paths[i], candidates)

    if qconf < qwen_conf_threshold:
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

print("\n===== Low-Confidence Qwen Selective =====")
print("Baseline Accuracy:", baseline_acc)
print("Final Accuracy:", final_acc)
print("Trigger Rate:", trigger / len(labels))
print("Correction:", correction)
print("Degradation:", degradation)
print("Net Gain:", correction - degradation)
print("Total Gain:", final_acc - baseline_acc)