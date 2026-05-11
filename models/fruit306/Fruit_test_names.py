import os
import csv
import glob
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models
import numpy as np
from tqdm import tqdm
import timm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ====================== 1. 命令行参数 ======================
parser = argparse.ArgumentParser(description='Test Dataset Top1/Top3/Top5 Accuracy + Save to CSV')
parser.add_argument('-m', '--model_name', type=str, required=True, help='Model name')
parser.add_argument('-w', '--weight', type=str, required=True, help='Weight path')
parser.add_argument('-d', '--data_root', type=str, default='/gemini/data-1/Fruit-306')
parser.add_argument('-b', '--batch_size', type=int, default=16)
args = parser.parse_args()

# ====================== 2. 设备 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ====================== 3. 图像预处理 ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.455, 0.406], [0.229, 0.224, 0.225])
])

# ====================== 4. 加载测试集 ======================
test_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'test'), transform)
test_loader = DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
)
class_names = test_dataset.classes
num_classes = len(class_names)
print(f'Classes: {num_classes}')
print(f'Test images: {len(test_dataset)}')

# ====================== 5. 模型 ======================
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

    elif model_name == 'inceptionresnetv2':
        # import pretrainedmodels
        # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        # model.last_linear = nn.Linear(1536, num_classes)
        import timm
        model = timm.create_model('inception_resnet_v2',num_classes=num_classes, pretrained=True)
    elif model_name == 'efficientnet':
        model = torchvision.models.efficientnet_b7()
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    elif model_name == 'nasnetalarge':
        import timm
        # import pretrainedmodels
        # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        # model.avg_pool = torch.nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
        # model.last_linear = torch.nn.Linear(in_features=4032, out_features=nb_out)
        model = timm.create_model('nasnetalarge',num_classes=num_classes, pretrained=True)
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16()
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    return model

model = build_model(args.model_name, num_classes)
model.load_state_dict(torch.load(args.weight, map_location=device))
model = model.to(device)
model.eval()
print(f'✅ Model loaded: {args.weight}')

# ====================== 6. 计算函数 ======================
def calculate_accuracy(outputs, labels, topk=(1,3,5)):
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

# ====================== 7. 测试 + 保存CSV ======================
def test_all_and_save_csv():
    # CSV 保存路径
    csv_path = f'test_result_{args.model_name}.csv'
    top1_acc, top3_acc, top5_acc = [], [], []

    # 写入表头
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_path', 'true_label',
            'top1_pred', 'top1_conf',
            'top2_pred', 'top2_conf',
            'top3_pred', 'top3_conf',
            'top4_pred', 'top4_conf',
            'top5_pred', 'top5_conf'
        ])

        pbar = tqdm(test_loader, desc='Testing')
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                # 计算准确率
                t1, t3, t5 = calculate_accuracy(outputs, labels)
                top1_acc.append(t1)
                top3_acc.append(t3)
                top5_acc.append(t5)

                # 获取当前batch的图片路径
                start_idx = batch_idx * args.batch_size
                end_idx = start_idx + images.size(0)
                img_paths = [test_dataset.samples[i][0] for i in range(start_idx, end_idx)]

                # 获取TOP5预测与置信度
                top5_probs, top5_indices = probs.topk(5, dim=1)
                top5_probs = top5_probs.cpu().numpy()
                top5_indices = top5_indices.cpu().numpy()
                labels_np = labels.cpu().numpy()

                # 逐张保存
                for i in range(len(img_paths)):
                    row = [
                        img_paths[i],
                        class_names[labels_np[i]],
                        class_names[top5_indices[i][0]], f"{top5_probs[i][0]:.4f}",
                        class_names[top5_indices[i][1]], f"{top5_probs[i][1]:.4f}",
                        class_names[top5_indices[i][2]], f"{top5_probs[i][2]:.4f}",
                        class_names[top5_indices[i][3]], f"{top5_probs[i][3]:.4f}",
                        class_names[top5_indices[i][4]], f"{top5_probs[i][4]:.4f}"
                    ]
                    writer.writerow(row)

                pbar.set_postfix({
                    'top1': np.mean(top1_acc),
                    'top3': np.mean(top3_acc),
                    'top5': np.mean(top5_acc)
                })

    # 最终指标
    final_top1 = np.mean(top1_acc)
    final_top3 = np.mean(top3_acc)
    final_top5 = np.mean(top5_acc)

    print('\n' + '='*60)
    print('📊 TEST RESULT')
    print(f'Top1: {final_top1:.4f}')
    print(f'Top3: {final_top3:.4f}')
    print(f'Top5: {final_top5:.4f}')
    print('='*60)
    print(f'💾 结果已保存到 CSV: {csv_path}\n')

    return final_top1, final_top3, final_top5

if __name__ == '__main__':
    test_all_and_save_csv()