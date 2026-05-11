import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 和训练保持一致
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 中国官方镜像
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models
import numpy as np
from tqdm import tqdm
import timm

# 允许加载损坏图片
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# python test.py -m vit_b_16 -w ./Pytorch_weights/best_vit_b_16_top1_0.7156_top3_0.8648_top5_0.9014.pt

# ====================== 1. 命令行参数 ======================
parser = argparse.ArgumentParser(description='Test Dataset Top1/Top3/Top5 Accuracy (306 Classes)')
parser.add_argument('-m', '--model_name', type=str, required=True, help='Model name (efficientnet, resnet50, vit_b_16...)')
parser.add_argument('-w', '--weight', type=str, required=True, help='Path to trained weight (.pt)')
parser.add_argument('-d', '--data_root', type=str, default='/gemini/data-1/Fruit-306', help='Dataset root (with test folder)')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
args = parser.parse_args()

# ====================== 2. 设备配置 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ====================== 3. 图像预处理（和训练完全一致） ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.455, 0.406], [0.229, 0.224, 0.225])
])

# ====================== 4. 加载测试集 ======================
test_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'test'), transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
num_classes = len(test_dataset.classes)
print(f'Number of test classes: {num_classes}')
print(f'Number of test images: {len(test_dataset)}')

# ====================== 5. 构建模型（和训练代码完全一致） ======================
def build_model(model_name, num_classes):
    if model_name == 'squeezenet':
        model = torchvision.models.squeezenet1_0()
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    elif model_name == 'alexnet':
        model = torchvision.models.alexnet()
        model.classifier[-1] = nn.Linear(4096, num_classes)
    elif model_name == 'googlenet':
        model = torchvision.models.googlenet()
        model.fc = nn.Linear(1024, num_classes)
    elif model_name == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0()
        model.fc = nn.Linear(1024, num_classes)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16()
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg19':
        model = torchvision.models.vgg19()
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2()
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50()
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101()
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d()
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'densenet161':
        model = torchvision.models.densenet161()
        model.classifier = nn.Linear(2208, num_classes)
    elif model_name == 'densenet':
        model = torchvision.models.densenet201()
        model.classifier = nn.Linear(1920, num_classes)
    elif model_name == 'inception_v3':
        model = torchvision.models.inception_v3()
        model.aux_logits = False
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'efficientnet':
        model = torchvision.models.efficientnet_b7()
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16()
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model

# 初始化模型 + 加载权重
model = build_model(args.model_name, num_classes)
model.load_state_dict(torch.load(args.weight, map_location=device))
model = model.to(device)
model.eval()

print(f'✅ Model loaded successfully: {args.weight}')

# ====================== 6. Top-K 准确率计算 ======================
def calculate_accuracy(outputs, labels, topk=(1, 3, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size).item())
        return res

# ====================== 7. 测试整个测试集 ======================
def test_all():
    top1_acc = []
    top3_acc = []
    top5_acc = []

    pbar = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            t1, t3, t5 = calculate_accuracy(outputs, labels)

            top1_acc.append(t1)
            top3_acc.append(t3)
            top5_acc.append(t5)

            # 实时显示进度
            pbar.set_postfix({
                'top1': np.mean(top1_acc),
                'top3': np.mean(top3_acc),
                'top5': np.mean(top5_acc)
            })

    # 最终结果
    final_top1 = np.mean(top1_acc)
    final_top3 = np.mean(top3_acc)
    final_top5 = np.mean(top5_acc)

    print('\n' + '=' * 60)
    print('📊 FINAL TEST DATASET ACCURACY')
    print(f'Top-1 Accuracy: {final_top1:.4f}')
    print(f'Top-3 Accuracy: {final_top3:.4f}')
    print(f'Top-5 Accuracy: {final_top5:.4f}')
    print('=' * 60 + '\n')

    return final_top1, final_top3, final_top5

# 开始测试
if __name__ == '__main__':
    test_all()