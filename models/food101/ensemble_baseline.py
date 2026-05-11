#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ensemble_baseline.py

多模型(ViT, EffNet, DenseNet) 纯置信度加权融合基线
策略:
1. 自动加载并合并 ViT, EfficientNet, DenseNet 的预测结果CSV。
2. 对三个模型的Top-K预测进行加权融合: w_conf * conf_score + w_dist * dist_score。
3. 得到融合后的 Top-5 排序结果。
4. 评估融合基线准确率，并与原始最佳单模型进行对比，保存详细结果。
"""

# ==============================================================================
# 猴子补丁 (防止潜在版本冲突报错)
# ==============================================================================
import torch
import numpy as np

_original_from_numpy = torch.from_numpy
def _patched_from_numpy(arr):
    try:
        return _original_from_numpy(arr)
    except TypeError:
        return torch.tensor(arr.tolist())
torch.from_numpy = _patched_from_numpy

# ==============================================================================
# 基础依赖
# ==============================================================================
import os
import argparse
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# 评估函数
# ==============================================================================
def evaluate_topk(df: pd.DataFrame, prefix: str, ks=(1, 3, 5)) -> dict:
    out = {}
    if "true_label" not in df.columns:
        return {f"top{k}": 0.0 for k in ks}

    is_ensemble = "ensemble" in prefix
    model_suffix = prefix.replace("top_", "") if not is_ensemble else ""

    for k in ks:
        if is_ensemble:
            cols = [f"{prefix}{i}_pred" for i in range(1, k + 1)]
        else:
            cols = [f"top{i}_pred_{model_suffix}" for i in range(1, k + 1)]
        
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            out[f"top{k}"] = 0.0
            continue

        df['true_label'] = df['true_label'].astype(str)
        for col in existing_cols:
            df[col] = df[col].astype(str)

        hit = pd.concat([(df[c] == df["true_label"]) for c in existing_cols], axis=1).any(axis=1)
        out[f"top{k}"] = float(hit.mean())
    return out

# ==============================================================================
# Main函数 - 纯融合基线版
# ==============================================================================
def main(args):
    # --- 1. 加载和合并数据 ---
    print("[Main] 正在加载和合并CSV文件...")
    models = ['vit', 'eff', 'den']
    csv_paths = {'vit': args.vit_csv, 'eff': args.eff_csv, 'den': args.den_csv}
    df_map = {model: pd.read_csv(path) for model, path in csv_paths.items()}

    # 加后缀以区分不同模型的列 (原始CSV为 top1_pred -> 合并后为 top1_pred_vit)
    df_vit = df_map['vit'].add_suffix('_vit')
    df_eff = df_map['eff'].add_suffix('_eff')
    df_den = df_map['den'].add_suffix('_den')
    
    df = pd.merge(df_vit, df_eff, left_on='image_path_vit', right_on='image_path_eff', how='inner')
    df = pd.merge(df, df_den, left_on='image_path_vit', right_on='image_path_den', how='inner')

    # 统一 image_path 和 true_label 列名，并清理多余的重复列
    df = df.rename(columns={'image_path_vit': 'image_path', 'true_label_vit': 'true_label'})
    df = df.drop(columns=[c for c in df.columns if c.startswith(('image_path_', 'true_label_')) and c not in ['image_path', 'true_label']])
    
    if args.sample_size > 0:
        df = df.sample(n=args.sample_size, random_state=42).copy() if args.sample_size < len(df) else df.copy()

    print(f"[Main] 共加载 {len(df)} 条测试数据")
    print(f"[Main] 融合公式: w_conf({args.w_conf}) * conf + w_dist({args.w_dist}) * dist (当前先跑纯conf基线)")

    # --- 2. 循环处理: 加权融合 ---
    new_cols = {f"ensemble_top{i}_pred": [] for i in range(1, 6)}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Ensemble Baseline Fusion"):
        scores = {}
        
        # 遍历三个模型的 Top-5 结果
        for model in models:
            for i in range(1, 6):
                pred = str(row[f'top{i}_pred_{model}'])
                conf = float(row[f'top{i}_conf_{model}'])
                
                # ==========================================
                # 核心融合公式
                # ==========================================
                conf_score = conf
                dist_score = 0.0  # 预留位置：后续加上特征距离时替换此行即可 (注意距离要转为越大越好的形式)
                
                total_score = (args.w_conf * conf_score) + (args.w_dist * dist_score)
                
                # 如果多个模型预测了同一个类别，分数累加
                scores[pred] = scores.get(pred, 0.0) + total_score

        # 根据融合总分降序排序，得到 Baseline Top-5
        baseline_ensemble_ranking = [item[0] for item in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

        # 填充最终排序结果
        for i in range(5):
            new_cols[f"ensemble_top{i+1}_pred"].append(baseline_ensemble_ranking[i] if i < len(baseline_ensemble_ranking) else "")

    for k, v in new_cols.items(): 
        df[k] = v
    
    # --- 3. 评估与展示 ---
    base_accs = {model: evaluate_topk(df, prefix=f'top_{model}', ks=[1])['top1'] for model in models}
    best_model_name = max(base_accs, key=base_accs.get) if base_accs else models[0]
    
    print(f"\n[Info] 最佳单模型基线: {best_model_name.upper()} (Top-1 Acc: {base_accs.get(best_model_name, 0):.4f})")

    orig_eval = evaluate_topk(df, prefix=f"top_{best_model_name}", ks=(1, 3, 5))
    new_eval  = evaluate_topk(df, prefix="ensemble_top", ks=(1, 3, 5))

    print("\n" + "=" * 70)
    print("   【多模型加权融合 Baseline】实验结果")
    print("=" * 70)
    print(f"  w_conf={args.w_conf} | w_dist={args.w_dist} | 样本数={len(df)}")
    print("-" * 70)
    print(f"{'指标':<14} {'原始最佳模型':>16} {'融合 Baseline':>14} {'提升':>10}")
    print(f"{'':<14} ({best_model_name.upper():^14}) {'':>14} {'':>10}")
    print("-" * 70)
    for k in (1, 3, 5):
        a = orig_eval.get(f"top{k}", 0.0)
        b = new_eval.get(f"top{k}", 0.0)
        print(f"Top-{k} 准确率    {a:>16.4f} {b:>14.4f} {b - a:>+10.4f}")
    print("=" * 70)

    # --- 4. 保存结果 ---
    out_csv = os.path.join(
        os.path.dirname(args.vit_csv),
        f"baseline_fusion_wC{args.w_conf}_wD{args.w_dist}_S{args.sample_size if args.sample_size > 0 else 'All'}.csv"
    )
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Baseline结果已保存: {out_csv}")

# ==============================================================================
# CLI 入口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模型融合 Baseline (无VLM)")
    
    parser.add_argument("--vit_csv", default="/mnt/ljh-21/organized_data_simple/new/test_result_vit_b_16.csv")
    parser.add_argument("--eff_csv", default="/mnt/ljh-21/organized_data_simple/new/test_result_efficientnet.csv")
    parser.add_argument("--den_csv", default="/mnt/ljh-21/organized_data_simple/new/test_result_densenet.csv")
    
    parser.add_argument("--w_conf", type=float, default=1.0, help="置信度分数权重")
    parser.add_argument("--w_dist", type=float, default=0.0, help="距离分数权重 (预留)")
    parser.add_argument("--sample_size", type=int, default=0, help="测试样本数，0表示全部")
    
    args = parser.parse_args()
    main(args)