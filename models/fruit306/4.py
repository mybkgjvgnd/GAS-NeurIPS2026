#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensemble_context_rerank.py

多模型(ViT, EffNet, DenseNet)候选集融合 + 上下文感知VLM重排 + TrustVLM门控

最终优化版策略 ("上下文感知VLM裁判"):
1.  自动加载并合并 ViT, EfficientNet, DenseNet 的预测结果CSV。
2.  基于三个模型Top-1的平均置信度，智能识别需要VLM介入的“困难样本”。
3.  对困难样本，动态构建候选池：根据每个基础模型自身的置信度，决定从该模型中选取Top-K个候选。
4.  为VLM生成一个“上下文感知”的Prompt：不仅提供候选类别，还详细告知每个候选分别被哪些模型以何种置信度预测。
5.  Qwen-VL 在最充分的信息下进行重排，并给出自己的置信度。
6.  当Qwen-VL的置信度高于设定阈值时，采纳其排序结果；否则，回退到鲁棒的加权投票集成结果。
7.  最终，评估新方法的准确率，并与原始最佳单模型进行对比，保存详细结果。
"""

# ==============================================================================
# 猴子补丁 (保持不变)
# ==============================================================================
import torch
import numpy as np

_original_from_numpy = torch.from_numpy
def _patched_from_numpy(arr):
    try:
        return _original_from_numpy(arr)
    except TypeError:
        # This is the patch
        return torch.tensor(arr.tolist())
torch.from_numpy = _patched_from_numpy

# ==============================================================================
# 正常 import
# ==============================================================================
import os
import re
import json
import unicodedata
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple, Dict
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ==============================================================================
# Qwen-VL 工具 (已优化OOM处理)
# ==============================================================================
def _process_vision_info_fallback(messages):
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str): continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                img = item.get("image")
                if isinstance(img, Image.Image): image_inputs.append(img)
                elif isinstance(img, str) and os.path.exists(img):
                    image_inputs.append(Image.open(img).convert("RGB"))
    return image_inputs, None

try:
    from qwen_vl_utils import process_vision_info
    print("[Info] 已加载 qwen_vl_utils ✅")
except ImportError:
    print("[Info] 使用内置 process_vision_info")
    process_vision_info = _process_vision_info_fallback

class QwenInferencer:
    def __init__(self, model_path: str, dtype: str = "fp16"):
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.float16)
        load_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype, "device_map": "auto"}
        print(f"[Qwen] 加载模型: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if getattr(self.processor, "tokenizer", None): self.processor.tokenizer.padding_side = "left"
        print(f"[Qwen] 加载完成，设备: {next(self.model.parameters()).device}")

    def _device(self): return next(self.model.parameters()).device

    @torch.inference_mode()
    def chat(self, image: Image.Image, prompt: str, max_new_tokens: int = 300) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            inputs = {k: (v.to(self._device()) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        except torch.cuda.OutOfMemoryError:
            print("[Warning] OOM！跳过。"); torch.cuda.empty_cache(); return "OOM_ERROR" # 返回特殊标识
        except Exception as e:
            print(f"[Warning] 推理异常: {e}"); return ""

# ==============================================================================
# 工具函数 (标签归一化, 路径转换) (保持不变)
# ==============================================================================
def normalize_label(s: str) -> str:
    s = unicodedata.normalize("NFD", s); s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9_]", "", s.lower().strip())

def transform_path(original_path: str, new_data_root: str) -> str:
    try:
        sub_path = original_path.split('/Fruit-306/')[-1]
        p = os.path.join(new_data_root, sub_path)
        return p if os.path.exists(p) else None
    except: return None

# ==============================================================================
# 核心改动: 通用化VLM响应解析器 (保持不变)
# ==============================================================================
def parse_vlm_response(answer: str, candidates: List[str]) -> Tuple[List[str], int]:
    original = list(candidates)
    confidence = 0
    if not answer or answer == "OOM_ERROR": return original, 0

    try:
        json_match = re.search(r'\{.*\}', answer, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            ranking_raw = data.get("ranking", [])
            confidence = int(data.get("confidence", 0))
            ranking, seen_norm = [], set()
            for item in ranking_raw:
                item_norm = normalize_label(str(item))
                for c in original:
                    c_norm = normalize_label(c)
                    if c_norm not in seen_norm and c_norm == item_norm:
                        ranking.append(c); seen_norm.add(c_norm); break
            for c in original:
                if normalize_label(c) not in seen_norm: ranking.append(c)
            return ranking, min(max(confidence, 0), 100)
    except (json.JSONDecodeError, ValueError, TypeError): pass

    conf_match = re.search(r'"?confidence"?\s*[:：]\s*(\d+)', answer)
    if conf_match: confidence = min(int(conf_match.group(1)), 100)
    list_match = re.search(r'\[([^\]]+)\]', answer)
    if list_match:
        items = re.findall(r'["\']?([a-zA-Z_]+)["\']?', list_match.group(1))
        ranking, seen = [], set()
        for item in items:
            for c in original:
                if (normalize_label(c) == normalize_label(item)) and c not in seen:
                    ranking.append(c); seen.add(c); break
        for c in original:
            if c not in seen: ranking.append(c)
        return ranking, confidence

    pos_list = sorted([(answer.lower().find(normalize_label(c)), c) for c in original], key=lambda x: x[0] if x[0]!=-1 else 999)
    return [c for _, c in pos_list], confidence

# ==============================================================================
# 核心改动: "上下文感知"的VLM重排函数 (保持不变)
# ==============================================================================
def rerank_with_vlm(
    qwen: QwenInferencer,
    image_path: str,
    candidates_info: Dict[str, List[Tuple[str, float]]],
    qwen_conf_threshold: int = 0,
) -> Tuple[List[str], int, str, bool]:
    
    candidates = list(candidates_info.keys())
    if not image_path or not os.path.exists(image_path) or not candidates:
        return candidates, 0, "", False

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Warning] 无法加载图片 {image_path}: {e}")
        return candidates, 0, "IMAGE_LOAD_ERROR", False
    
    sorted_cands = sorted(
        candidates_info.items(),
        key=lambda item: (len(item[1]), sum(c for _, c in item[1])),
        reverse=True
    )

    candidates_str = "\n".join([
        f'- "{cand}": (来自 {", ".join([f"{m}({c:.1%})" for m, c in info])})'
        for cand, info in sorted_cands
    ])

    prompt = (
        "你是一位顶级的图像分类专家，任务是审查并重排一个由多个模型提供的候选列表。\n\n"
        "背景：多个基础分类模型（ViT, EffNet, DenseNet）观察了这张图片，并给出了它们的预测和置信度。我已经将它们的结果汇总如下：\n"
        f"【候选类别及来源】\n{candidates_str}\n\n"
        "你的任务是：\n"
        "1. **仔细观察图片**，这是最主要的判断依据。\n"
        "2. 结合候选列表中的**上下文信息**（特别是那些被多个模型高置信度支持的选项），对列表中的**所有**类别进行重新排序，从最匹配到最不匹配。\n"
        "3. 评估你对最终排序的**整体信心**。\n\n"
        "【输出要求】\n"
        "严格的JSON格式，包含`ranking`（一个包含所有候选类别的列表）和`confidence`（0-100的整数）。\n\n"
        "直接输出JSON："
    )

    raw_answer = qwen.chat(image, prompt, max_new_tokens=len(candidates) * 25 + 100)
    
    # 如果发生OOM，则直接返回，不进行解析
    if raw_answer == "OOM_ERROR":
        return candidates, 0, raw_answer, False
        
    new_ranking, qc = parse_vlm_response(raw_answer, candidates)

    adopted = (qc >= qwen_conf_threshold)
    final_ranking = new_ranking if adopted else candidates
    return final_ranking, qc, raw_answer, adopted

# ==============================================================================
# 评估函数 (★★★ 已修复 ★★★)
# ==============================================================================
def evaluate_topk(df: pd.DataFrame, prefix: str, ks=(1, 3, 5)) -> dict:
    out = {}
    if "true_label" not in df.columns:
        return {f"top{k}": 0.0 for k in ks}

    # 从 "top_vit" 或 "ensemble_top" 中提取标识
    is_ensemble = "ensemble" in prefix
    model_suffix = prefix.replace("top_", "") if not is_ensemble else ""

    for k in ks:
        if is_ensemble:
            # 处理 ensemble 的情况, e.g., ensemble_top1_pred
            cols = [f"{prefix}{i}_pred" for i in range(1, k + 1)]
        else:
            # 处理单模型的情况, e.g., top1_pred_vit
            cols = [f"top{i}_pred_{model_suffix}" for i in range(1, k + 1)]
        
        existing_cols = [c for c in cols if c in df.columns]

        if not existing_cols:
            out[f"top{k}"] = 0.0
            continue

        # 确保真实标签列和预测列中的NaN值被妥善处理，避免比较错误
        df['true_label'] = df['true_label'].astype(str)
        for col in existing_cols:
            df[col] = df[col].astype(str)

        hit = pd.concat([
            (df[c] == df["true_label"]) for c in existing_cols
        ], axis=1).any(axis=1)
        
        out[f"top{k}"] = float(hit.mean())
    return out

# ==============================================================================
# Main函数 - (★★★ 已优化 ★★★)
# ==============================================================================
def main(args):
    # --- 1. 加载和合并数据 ---
    print("[Main] 正在加载和合并CSV文件...")
    models = ['vit', 'eff', 'den']
    csv_paths = {
        'vit': args.vit_csv,
        'eff': args.eff_csv,
        'den': args.den_csv
    }
    df_map = {model: pd.read_csv(path) for model, path in csv_paths.items()}

    df_vit = df_map['vit'].add_suffix('_vit')
    df_eff = df_map['eff'].add_suffix('_eff')
    df_den = df_map['den'].add_suffix('_den')
    
    df = pd.merge(df_vit, df_eff, left_on='image_path_vit', right_on='image_path_eff', how='inner')
    df = pd.merge(df, df_den, left_on='image_path_vit', right_on='image_path_den', how='inner')

    df = df.rename(columns={'image_path_vit': 'image_path', 'true_label_vit': 'true_label'})
    df = df.drop(columns=[c for c in df.columns if c.startswith(('image_path_', 'true_label_')) and c not in ['image_path', 'true_label']])
    
    if args.sample_size > 0:
        df = df.sample(n=args.sample_size, random_state=42).copy() if args.sample_size < len(df) else df.copy()

    # --- 2. 初始化Qwen ---
    qwen = QwenInferencer(model_path=args.qwen_model_path, dtype=args.qwen_dtype)

    # --- 3. 循环处理 ---
    avg_top1_confs = (df['top1_conf_vit'] + df['top1_conf_eff'] + df['top1_conf_den']) / 3
    need_rerank_mask = avg_top1_confs <= args.lambda_val
    print(f"[Main] 共 {len(df)} 条 | lambda={args.lambda_val} | qwen_conf_threshold={args.qwen_conf_threshold} | FusionK=Dynamic")
    print(f"       需VLM重排: {need_rerank_mask.sum()} 条 (基于平均Top1置信度)")

    new_cols = {f"ensemble_top{i}_pred": [] for i in range(1, 6)}
    diag_cols = {'vlm_qwen_conf': [], 'vlm_adopted': [], 'vlm_changed_top1': [], 'vlm_raw_answer': [], 'fusion_candidates_info': []}
    stats = {"skipped": 0, "reranked": 0, "adopted": 0, "rejected": 0, "oom_failures": 0, "img_load_failures": 0}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Context-Aware Reranking"):
        scores = {}
        for model in models:
            for i in range(1, 6):
                pred = str(row[f'top{i}_pred_{model}'])
                conf = float(row[f'top{i}_conf_{model}'])
                # 使用加权求和，这里直接用置信度作为权重
                scores[pred] = scores.get(pred, 0) + conf
        
        baseline_ensemble_ranking = [item[0] for item in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

        if not need_rerank_mask.loc[idx]:
            final_ranking = baseline_ensemble_ranking
            qwen_conf, raw_answer, adopted, fusion_info_str = -1, "SKIPPED", False, ""
            stats["skipped"] += 1
        else:
            stats["reranked"] += 1
            
            # --- 动态候选池构建 ---
            candidates_info = {}
            unique_candidates_set = set()

            for model in models:
                top1_conf = float(row[f'top1_conf_{model}'])
                k = 1 if top1_conf > 0.9 else (3 if top1_conf > 0.5 else 5)
                for i in range(1, k + 1):
                    pred = str(row[f'top{i}_pred_{model}'])
                    conf = float(row[f'top{i}_conf_{model}'])
                    
                    if pred not in unique_candidates_set:
                        unique_candidates_set.add(pred)
                        candidates_info[pred] = []
                    
                    # 确保每个模型对一个候选只记录一次
                    if not any(m[0] == model.upper() for m in candidates_info[pred]):
                         candidates_info[pred].append((model.upper(), conf))

            # 对候选信息按模型来源数量和总置信度排序
            for pred in candidates_info:
                 candidates_info[pred].sort(key=lambda x: x[1], reverse=True)

            img_path = transform_path(row["image_path"], args.data_root)
            vlm_ranking, qwen_conf, raw_answer, adopted = rerank_with_vlm(
                qwen=qwen,
                image_path=img_path,
                candidates_info=candidates_info,
                qwen_conf_threshold=args.qwen_conf_threshold,
            )

            # --- 处理VLM返回结果 ---
            if raw_answer == "OOM_ERROR":
                stats["oom_failures"] += 1
                final_ranking = baseline_ensemble_ranking
            elif raw_answer == "IMAGE_LOAD_ERROR":
                stats["img_load_failures"] +=1
                final_ranking = baseline_ensemble_ranking
            else:
                final_ranking = vlm_ranking if adopted else baseline_ensemble_ranking
                if adopted: stats["adopted"] += 1
                else: stats["rejected"] += 1
            
            fusion_info_str = json.dumps({k: v for k, v in candidates_info.items()})

        # 填充最终排序结果
        seen = set(final_ranking)
        for item in baseline_ensemble_ranking:
            if item not in seen:
                final_ranking.append(item)
        
        for i in range(5):
            new_cols[f"ensemble_top{i+1}_pred"].append(final_ranking[i] if i < len(final_ranking) else "")
        
        # 记录诊断信息
        diag_cols['vlm_qwen_conf'].append(qwen_conf)
        diag_cols['vlm_adopted'].append(adopted)
        diag_cols['vlm_changed_top1'].append(len(final_ranking) > 0 and len(baseline_ensemble_ranking) > 0 and final_ranking[0] != baseline_ensemble_ranking[0])
        diag_cols['vlm_raw_answer'].append(raw_answer[:300]) # 截断以防文件过大
        diag_cols['fusion_candidates_info'].append(fusion_info_str)


    for k, v in new_cols.items(): df[k] = v
    for k, v in diag_cols.items(): df[k] = v
    
    # --- 4. 评估与展示 ---
    print(f"\n[Stats] 跳过VLM: {stats['skipped']} | VLM介入: {stats['reranked']} (成功采纳: {stats['adopted']}, 回退/拒绝: {stats['rejected']})")
    print(f"       失败统计: {stats['oom_failures']}次OOM, {stats['img_load_failures']}次图片加载失败")

    # 评估基线模型
    base_accs = {model: evaluate_topk(df, prefix=f'top_{model}', ks=[1])['top1'] for model in models}
    best_model_name = max(base_accs, key=base_accs.get) if base_accs else models[0]
    
    print(f"[Info] 最佳单模型基线: {best_model_name.upper()} (Top-1 Acc: {base_accs.get(best_model_name, 0):.4f})")

    # 评估原始最佳模型和新方法
    orig_eval = evaluate_topk(df, prefix=f"top_{best_model_name}", ks=(1, 3, 5))
    new_eval  = evaluate_topk(df, prefix="ensemble_top", ks=(1, 3, 5))

    print("\n" + "=" * 70)
    print("   【上下文感知VLM裁判】实验结果")
    print("=" * 70)
    print(f"  Lambda={args.lambda_val} | QwenConf={args.qwen_conf_threshold} | FusionK=Dynamic | 样本数={len(df)}")
    print("-" * 70)
    print(f"{'指标':<14} {'原始最佳模型':>16} {'融合+VLM后':>14} {'提升':>10}")
    print(f"{'':<14} ({best_model_name.upper():^14}) {'':>14} {'':>10}")
    print("-" * 70)
    for k in (1, 3, 5):
        a = orig_eval.get(f"top{k}", 0.0)
        b = new_eval.get(f"top{k}", 0.0)
        print(f"Top-{k} 准确率    {a:>16.4f} {b:>14.4f} {b - a:>+10.4f}")
    print("=" * 70)

    # --- 5. 保存结果 ---
    out_csv = os.path.join(
        os.path.dirname(args.vit_csv),
        f"context_rerank_L{args.lambda_val}_Q{args.qwen_conf_threshold}_S{args.sample_size if args.sample_size > 0 else 'All'}.csv"
    )
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存: {out_csv}")

# ==============================================================================
# CLI 入口 (保持不变)
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模型融合 + 上下文感知VLM重排")
    
    # 输入文件路径 (请根据你的实际情况修改默认值)
    parser.add_argument("--vit_csv", default="/mnt/ljh-21/organized_data_simple/new/test_result_vit_b_16.csv", help="ViT模型预测结果CSV")
    parser.add_argument("--eff_csv", default="/mnt/ljh-21/organized_data_simple/new/test_result_efficientnet.csv", help="EfficientNet模型预测结果CSV")
    parser.add_argument("--den_csv", default="/mnt/ljh-21/organized_data_simple/new/test_result_densenet.csv", help="DenseNet模型预测结果CSV")
    
    # 路径和模型
    parser.add_argument("--data_root", default="/mnt/ljh-21/organized_data_simple", help="图像文件根目录")
    parser.add_argument("--qwen_model_path", default="/mnt/ljh-21/ljh/Qwen/Qwen2.5-VL-7B-Instruct", help="Qwen-VL模型路径")
    parser.add_argument("--qwen_dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Qwen-VL加载精度")
    
    # 超参数
    parser.add_argument("--lambda_val", type=float, default=0.90, help="平均Top1置信度阈值，低于此值则触发VLM重排")
    parser.add_argument("--qwen_conf_threshold", type=int, default=70, help="Qwen自评置信度阈值，高于此值才采纳其排序结果")
    parser.add_argument("--sample_size", type=int, default=0, help="用于测试的样本数量，0表示全部样本。建议先用小样本(如500)测试。")
    
    args = parser.parse_args()
    main(args)