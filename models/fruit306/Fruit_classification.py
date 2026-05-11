import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="3";
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models
from tqdm import tqdm
import numpy as np
# import timm

# ====================== 只加这一行修复图片报错，不改动你任何逻辑 ======================
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# python Fruit_classification.py -e 1000 -b 16 -m efficientnet -w ./Pytorch_weights/weights_2_50000_efficientnet.pt
# --------------------------
# 1. Command Line Arguments
# --------------------------
parser = argparse.ArgumentParser(description='Image Classification Training (306 Categories)')
parser.add_argument('-e', '--epochs', type=int, default=1000, help='Total training epochs')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('-m', '--model_name', type=str, required=True, help='Model name (e.g., efficientnet, resnet50)')
parser.add_argument('-w', '--weight', type=str, default='', help='Path to pre-trained weights (optional)')
parser.add_argument('-d', '--data_root', type=str, default='/gemini/data-1/Fruit-306', help='Root folder with train/valid/test subfolders')
parser.add_argument('-s', '--save_dir', type=str, default='./Pytorch_weights', help='Weight save directory')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate')
args = parser.parse_args()

# Create save directory
os.makedirs(args.save_dir, exist_ok=True)

# --------------------------
# 2. Device Configuration (GPU/CPU)
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --------------------------
# 3. Data Loading & Augmentation
# --------------------------
# Image transforms (ImageNet standard normalization)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Auto-load datasets from train/valid/test folders
image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_root, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                             shuffle=True if x == 'train' else False,
                             num_workers=4, pin_memory=True)
               for x in ['train', 'val', 'test']}

# Auto-detect number of classes (306 in your case)
nb_out = len(image_datasets['train'].classes)
print(f'Number of classes: {nb_out}')

# --------------------------
# 4. Model Definition (YOUR FULL MODEL LIST)
# --------------------------
def build_model(model_name, num_classes):
    if args.weight:
        # Load custom weights
        model = torch.load(args.weight, weights_only=False)
        print(f'Loaded weights from: {args.weight}')
        return model.to(device)

    # 18+ ImageNet models (your original code)
    if model_name == 'squeezenet':
        model = torchvision.models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    
    elif model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    
    elif model_name == 'googlenet':
        model = torchvision.models.googlenet(pretrained=True)
        model.fc = nn.Linear(1024, num_classes)
    
    elif model_name == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(1024, num_classes)
    
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
    
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    
    elif model_name == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(1280, num_classes)
    
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    
    elif model_name == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(2048, num_classes)
    
    elif model_name == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Linear(2208, num_classes)
    
    elif model_name == 'densenet':
        model = torchvision.models.densenet201(pretrained=True)
        model.classifier = nn.Linear(1920, num_classes)
    
    elif model_name == 'inception_v3':
        model = torchvision.models.inception_v3(pretrained=True)
        model.aux_logits = False
        model.fc = nn.Linear(2048, num_classes)
    
    elif model_name == 'xception':
        import timm
        model = timm.create_model('xception65', num_classes=num_classes, pretrained=True)
    
    elif model_name == 'inceptionresnetv2':
        # import pretrainedmodels
        # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        # model.last_linear = nn.Linear(1536, num_classes)
        import timm
        model = timm.create_model('inception_resnet_v2',num_classes=num_classes, pretrained=True)
    elif model_name == 'nasnetalarge':
        import timm
        # import pretrainedmodels
        # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        # model.avg_pool = torch.nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
        # model.last_linear = torch.nn.Linear(in_features=4032, out_features=nb_out)
        model = timm.create_model('nasnetalarge',num_classes=num_classes, pretrained=True)
        
    elif model_name == 'efficientnet':
        # import timm
        # 换成 timm 版 EfficientNet，速度超快，不卡顿！
        # model = timm.create_model('efficientnet_b7', pretrained=True, num_classes=num_classes)
        # model = timm.create_model('tf_efficientnet_b7', pretrained=True, num_classes=num_classes)
    # elif model_name == 'efficientnet':
    #     from efficientnet_pytorch import EfficientNet
    #     model = EfficientNet.from_pretrained('efficientnet-b7')
    #     model._fc = nn.Linear(2560, num_classes)

        model = torchvision.models.efficientnet_b7(weights='IMAGENET1K_V1')
        # 替换分类层为你的 306 类
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif model_name == 'beit_large':
        import timm
        model = timm.create_model('beit_large_patch16_512', num_classes=num_classes, pretrained=True)


    elif model_name == 'vit_b_16':
        import timm
        # ViT-B/16 (ImageNet pretrained, official, same as torchvision)
        # model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)    

        model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')  # 🔥 这里是关键
        # 替换分类头为你的306类
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError('##### Please define your model #####')
    
    return model.to(device)

# Initialize model
model = build_model(args.model_name, nb_out)

# --------------------------
# 5. Loss & Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

# --------------------------
# 6. Metric Calculation (Top-1, Top-3, Top-5 Accuracy)
# --------------------------
def calculate_accuracy(outputs, labels, topk=(1, 3, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        # Get top-k predictions
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        # Calculate top-k accuracy
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / batch_size).item())
        return res

# --------------------------
# 7. Training & Validation Function
# --------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    total = 0

    pbar = tqdm(dataloaders['train'], desc=f'Train Epoch {epoch}')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        pbar.set_postfix({'loss': running_loss / total})

    return running_loss / total

def validate(epoch):
    model.eval()
    running_loss = 0.0
    top1_acc, top3_acc, top5_acc = [], [], []

    with torch.no_grad():
        # ====================== 这里把 valid 改成 val ======================
        pbar = tqdm(dataloaders['val'], desc=f'Valid Epoch {epoch}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            t1, t3, t5 = calculate_accuracy(outputs, labels)
            top1_acc.append(t1)
            top3_acc.append(t3)
            top5_acc.append(t5)

            pbar.set_postfix({
                # ====================== 这里也把 valid 改成 val ======================
                'loss': running_loss / len(dataloaders['val'].dataset),
                'top1': np.mean(top1_acc)
            })

    # ====================== 这里也把 valid 改成 val ======================
    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    return epoch_loss, np.mean(top1_acc), np.mean(top3_acc), np.mean(top5_acc)

# --------------------------
# 8. Test Function (Final Evaluation)
# --------------------------
def test_model():
    print('\n' + '-'*50)
    print('Starting Final Test Evaluation...')
    print('-'*50)

    model.eval()
    top1_acc, top3_acc, top5_acc = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloaders['test'], desc='Testing')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            t1, t3, t5 = calculate_accuracy(outputs, labels)
            top1_acc.append(t1)
            top3_acc.append(t3)
            top5_acc.append(t5)

    # Final test metrics
    final_top1 = np.mean(top1_acc)
    final_top3 = np.mean(top3_acc)
    final_top5 = np.mean(top5_acc)

    print('\n' + '='*50)
    print('TEST RESULTS (306 Categories)')
    print(f'Overall Top-1 Accuracy: {final_top1:.4f}')
    print(f'Top-3 Accuracy:        {final_top3:.4f}')
    print(f'Top-5 Accuracy:        {final_top5:.4f}')
    print('='*50 + '\n')

    return final_top1, final_top3, final_top5

# --------------------------
# 9. Main Training Loop
# --------------------------
best_acc = 0.0  # Track best validation accuracy

for epoch in range(1, args.epochs + 1):
    print(f'\n========== Epoch {epoch}/{args.epochs} ==========')

    # Train
    train_loss = train_one_epoch(epoch)

    # Validate
    val_loss, val_top1, val_top3, val_top5 = validate(epoch)

    # Update learning rate
    scheduler.step(val_loss)

    # Print epoch summary
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
    print(f'Val Top-1: {val_top1:.4f} | Val Top-3: {val_top3:.4f} | Val Top-5: {val_top5:.4f}')

    # # 1. Save BEST model (based on validation top-1 accuracy)
    # if val_top1 > best_acc:
    #     best_acc = val_top1
    #     best_weight_path = os.path.join(args.save_dir, f'best_{args.model_name}.pt')
    #     torch.save(model, best_weight_path)
    #     print(f'✅ Best model saved to: {best_weight_path}')

    # # 2. Save model EVERY 10 epochs
    # if epoch % 10 == 0:
    #     epoch_weight_path = os.path.join(args.save_dir, f'epoch_{epoch}_{args.model_name}.pt')
    #     torch.save(model, epoch_weight_path)
    #     print(f'📌 Epoch {epoch} model saved to: {epoch_weight_path}')

    # 1. Save BEST model (based on validation top-1 accuracy) + ADD ACCURACY TO FILENAME
    if val_top1 > best_acc:
        best_acc = val_top1
        # Save filename with top1, top3, top5 accuracy (4 decimal places)
        best_weight_path = os.path.join(
            args.save_dir, 
            f'best_{args.model_name}_top1_{val_top1:.4f}_top3_{val_top3:.4f}_top5_{val_top5:.4f}.pt'
        )
        # Save only state dict (standard, safe, compatible)
        torch.save(model.state_dict(), best_weight_path)
        print(f'✅ Best model saved to: {best_weight_path}')
    
    # 2. Save model EVERY 10 epochs + ADD ACCURACY TO FILENAME
    if epoch % 10 == 0:
        epoch_weight_path = os.path.join(
            args.save_dir,              f'epoch_{epoch}_{args.model_name}_top1_{val_top1:.4f}_top3_{val_top3:.4f}_top5_{val_top5:.4f}.pt'
        )
        torch.save(model.state_dict(), epoch_weight_path)
        print(f'📌 Epoch {epoch} model saved to: {epoch_weight_path}')
    
# --------------------------
# 10. Final Test on Test Dataset
# --------------------------
# Load the BEST model before testing
# best_model_path = os.path.join(args.save_dir, f'best_{args.model_name}.pt')
model = torch.load(best_weight_path)
print(f'\nLoaded best model from: {best_model_path} for testing')

# Run test
test_model()