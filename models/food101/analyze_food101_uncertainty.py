import numpy as np

# Load ensemble probs
resnet_probs = np.load("food101_resnet_probs.npy")
eff_probs    = np.load("food101_efficientnet_probs.npy")
vit_probs    = np.load("food101_vit_probs.npy")
labels       = np.load("food101_test_labels.npy")

ensemble_probs = (resnet_probs + eff_probs + vit_probs) / 3
preds = np.argmax(ensemble_probs, axis=1)

correct_mask = (preds == labels)

top1_conf = np.max(ensemble_probs, axis=1)

# 统计不同置信区间的错误率
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

print("\nConfidence Interval Analysis\n")

for i in range(len(bins)-1):
    low = bins[i]
    high = bins[i+1]

    mask = (top1_conf >= low) & (top1_conf < high)

    if mask.sum() == 0:
        continue

    acc = (correct_mask[mask]).mean()
    print(f"{low:.1f}-{high:.1f}: Samples={mask.sum()}, Acc={acc:.4f}")

# 计算低置信区域比例
low_conf_mask = top1_conf < 0.6
print("\nLow confidence (<0.6) ratio:", low_conf_mask.mean())