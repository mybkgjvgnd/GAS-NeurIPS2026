import numpy as np

print("Loading probability files...")

# 加载三个模型的 softmax 输出
resnet_probs = np.load("food101_resnet_probs.npy")
eff_probs    = np.load("food101_efficientnet_probs.npy")
vit_probs    = np.load("food101_vit_probs.npy")
labels       = np.load("food101_test_labels.npy")

num_samples = len(labels)

# -----------------------------
# Stage 1: Raw Softmax Ensemble
# -----------------------------

ensemble_probs = (resnet_probs + eff_probs + vit_probs) / 3
ensemble_preds = np.argmax(ensemble_probs, axis=1)

baseline_acc = (ensemble_preds == labels).mean()

print("\n===== Raw Ensemble =====")
print("Top1 Accuracy:", round(baseline_acc, 6))

# -----------------------------
# Stage 2: Weighted Ensemble
# -----------------------------

# 根据模型性能分配权重
w_resnet = 0.30
w_eff    = 0.40   # 最强
w_vit    = 0.30

weighted_probs = (
    w_resnet * resnet_probs +
    w_eff    * eff_probs +
    w_vit    * vit_probs
)

weighted_preds = np.argmax(weighted_probs, axis=1)
weighted_acc = (weighted_preds == labels).mean()

print("\n===== Weighted Ensemble =====")
print("Top1 Accuracy:", round(weighted_acc, 6))

# -----------------------------
# Stage 3: Top5 Majority Voting
# -----------------------------

correct_top5 = 0

for i in range(num_samples):
    combined = ensemble_probs[i]
    top5 = np.argsort(combined)[-5:]
    if labels[i] in top5:
        correct_top5 += 1

top5_acc = correct_top5 / num_samples

print("\n===== Ensemble Top5 =====")
print("Top5 Accuracy:", round(top5_acc, 6))

print("\nDone.")