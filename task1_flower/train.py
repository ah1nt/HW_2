import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
from model import get_resnet18, ResNet18_SE, get_vit

def train(args):
    # 构建有意义的 run name，方便在 wandb 中对比曲线
    run_name = f"{args.model}_pre{args.pretrained}_lr{args.lr}_ep{args.epochs}"
    
    try:
        wandb.init(project="hw2-flower-classification", name=run_name, config=args)
    except Exception as e:
        print("W&B login failed. Falling back to offline mode for logging.")
        wandb.init(mode="offline", project="hw2-flower-classification", name=run_name, config=args)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 遵循官方划分：
    # 训练集：使用官方的 train 划分 (1020张)
    # 验证集：使用官方的 val 划分 (1020张)
    train_dataset = datasets.Flowers102(root='./data', split='train', download=True, transform=train_transform)
    val_dataset = datasets.Flowers102(root='./data', split='val', download=True, transform=val_transform)
    
    # Windows 环境下 num_workers 建议设为 0 以避免多进程报错
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    if args.model == 'resnet18':
        model = get_resnet18(pretrained=args.pretrained)
        fc_name = 'fc'
    elif args.model == 'resnet18_se':
        model = ResNet18_SE(pretrained=args.pretrained)
        fc_name = 'fc'
    elif args.model == 'vit':
        model = get_vit(pretrained=args.pretrained)
        fc_name = 'head'
    else:
        raise ValueError("Invalid model name")
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.pretrained:
        # 提取新的输出层(全连接层)的参数和骨干网络的参数
        head_params = list(getattr(model, fc_name).parameters())
        head_params_ids = list(map(id, head_params))
        base_params = filter(lambda p: id(p) not in head_params_ids, model.parameters())
        
        # 分层学习率设置：
        # 新的输出层从零开始训练，使用较大的学习率 args.lr
        # 骨干网络使用在 ImageNet 上预训练的参数进行初始化，只做微调，因此使用较小的学习率 (args.lr * 0.1)
        optimizer = optim.Adam([
            {'params': base_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr}
        ], weight_decay=1e-4)
    else:
        # 如果不使用预训练模型(随机初始化)，所有参数一视同仁，使用统一学习率
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        wandb.log({
            "train_loss": train_loss, 
            "train_acc": train_acc, 
            "val_loss": val_loss, 
            "val_acc": val_acc, 
            "learning_rate": optimizer.param_groups[-1]['lr'],
            "epoch": epoch + 1
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/{run_name}_best.pth')
            
    print(f"Run {run_name} completed. Best Val Acc: {best_acc:.2f}%")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet18_se', 'vit'])
    parser.add_argument('--pretrained', action='store_true', help='使用 ImageNet 预训练权重')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3, help='新输出层的学习率')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    train(args)
