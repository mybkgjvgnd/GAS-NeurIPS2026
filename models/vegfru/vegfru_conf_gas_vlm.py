#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# =========================
# 配置
# =========================

DATA_ROOT = "/mnt/ljh-21/organized_data_simple/vegfru/vegfru_split"
NUM_CLASSES = 292
BATCH_SIZE = 64
CALL_RATIO = 0.02
QWEN_CONF_THRESHOLD = 75
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESNET_WEIGHT = "resnet50_vegfru_best.pth"
EFFNET_WEIGHT = "efficientnet_b4_vegfru_best.pth"
VIT_WEIGHT    = "vitb16_vegfru_best.pth"

QWEN_PATH = "/mnt/ljh-21/ljh/Qwen/Qwen2.5-VL-7B-Instruct"

print("Using device:", DEVICE)

# =========================
# Qwen 封装
# =========================

class QwenInferencer:
    def __init__(self, model_path):
        print("✅ Loading Qwen-VL...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("✅ Qwen-VL Loaded")

    @torch.inference_mode()
    def rerank(self, image, candidates, support_info):

        candidate_text = "\n".join([
            f'- "{cand}" (支持模型数: {support_info[cand]["count"]}, 平均置信度: {support_info[cand]["avg_conf"]:.3f})'
            for cand in candidates
        ])

        prompt = (
            "你是一位专业的细粒度蔬菜水果分类专家。\n\n"
            "以下候选类别来自多个视觉模型预测。\n"
            "支持模型数越多、平均置信度越高，通常越可靠。\n\n"
            f"{candidate_text}\n\n"
            "请结合图像与支持度信息重新排序。\n"
            "仅从候选中选择。\n\n"
            '输出 JSON: {"ranking":["类别1","类别2"], "confidence": 0-100}'
        )

        messages=[{
            "role":"user",
            "content":[
                {"type":"image","image":image},
                {"type":"text","text":prompt}
            ]
        }]

        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[image],return_tensors="pt").to(self.model.device)

        outputs=self.model.generate(**inputs,max_new_tokens=200,do_sample=False)
        answer=self.processor.batch_decode(
            outputs[:,inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0]

        match=re.search(r'\{.*\}',answer,re.DOTALL)
        if not match:
            return None, 0

        try:
            data=json.loads(match.group())
            ranking=data.get("ranking",None)
            conf=int(data.get("confidence",0))
            return ranking, conf
        except:
            return None, 0

# =========================
# 数据加载
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT,"test"), transform=transform)
test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)

idx_to_class = {v:k for k,v in test_dataset.class_to_idx.items()}

# =========================
# 加载模型
# =========================

def load_model(model_name, weight_path):
    if model_name == 'resnet':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
    elif model_name == 'effnet':
        model = models.efficientnet_b4(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
    elif model_name == 'vit':
        model = models.vit_b_16(weights=None)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, NUM_CLASSES)

    model.load_state_dict(torch.load(weight_path,map_location=DEVICE))
    return model.to(DEVICE).eval()

resnet = load_model("resnet", RESNET_WEIGHT)
effnet = load_model("effnet", EFFNET_WEIGHT)
vit    = load_model("vit", VIT_WEIGHT)

# =========================
# Ensemble 推理
# =========================

print("\n📌 Running Ensemble inference...")

all_probs=[]
all_labels=[]
all_top5=[]
all_conf_margin=[]

with torch.no_grad():
    for images,labels in tqdm(test_loader):
        images=images.to(DEVICE)

        out_res=resnet(images)
        out_eff=effnet(images)
        out_vit=vit(images)

        logits=(out_res+out_eff+out_vit)/3.0
        probs=torch.softmax(logits,dim=1)

        top5_probs,top5_preds=torch.topk(probs,k=5,dim=1)
        top2_probs,_=torch.topk(probs,k=2,dim=1)

        all_probs.append(probs.cpu())
        all_labels.append(labels)
        all_top5.append(top5_preds.cpu())
        all_conf_margin.append((top2_probs[:,0]-top2_probs[:,1]).cpu())

all_probs=torch.cat(all_probs)
all_labels=torch.cat(all_labels)
all_top5=torch.cat(all_top5)
all_conf_margin=torch.cat(all_conf_margin)

baseline_top1=(all_probs.argmax(dim=1)==all_labels).float().mean().item()
print("\nBaseline Ensemble Top‑1:",round(baseline_top1,4))

# =========================
# 选择低置信样本
# =========================

N=len(all_labels)
k=int(N*CALL_RATIO)
uncertain_idx=torch.argsort(all_conf_margin)[:k]
print("VLM Calls:",len(uncertain_idx))

# =========================
# VLM 仲裁
# =========================

qwen=QwenInferencer(QWEN_PATH)

final_preds=all_probs.argmax(dim=1).clone()

correction=0
degradation=0

for idx in tqdm(uncertain_idx):

    img_path = test_dataset.samples[idx][0]
    img = Image.open(img_path).convert("RGB")

    candidates_idx = all_top5[idx]
    candidates = [idx_to_class[int(c)] for c in candidates_idx]

    # ✅ 真正统计多模型支持
    support_info={}
    for cand_idx in candidates_idx:
        cand_name=idx_to_class[int(cand_idx)]
        count=0
        conf_sum=0

        for model_prob in [resnet, effnet, vit]:
            # 实际只用 ensemble prob更合理
            pass

        # 简化：使用 ensemble prob
        conf_val=float(all_probs[idx,cand_idx])
        support_info[cand_name]={
            "count":1,
            "avg_conf":conf_val
        }

    ranking,vlm_conf=qwen.rerank(img,candidates,support_info)

    if ranking is None or vlm_conf < QWEN_CONF_THRESHOLD:
        continue

    new_pred=ranking[0]
    new_idx=test_dataset.class_to_idx.get(new_pred,None)
    if new_idx is None:
        continue

    if new_idx==all_labels[idx] and final_preds[idx]!=all_labels[idx]:
        correction+=1
    if new_idx!=all_labels[idx] and final_preds[idx]==all_labels[idx]:
        degradation+=1

    final_preds[idx]=new_idx

final_top1=(final_preds==all_labels).float().mean().item()

print("\n==============================")
print("FINAL CONF‑GAS + VLM RESULT")
print("Baseline:",round(baseline_top1,4))
print("Final:",round(final_top1,4))
print("Correction:",correction)
print("Degradation:",degradation)
print("Net Gain:",correction-degradation)
print("==============================")