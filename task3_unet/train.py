import argparse
import os
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import wandb

from model import UNet
from loss import DiceLoss, CombinedLoss


DATASET_ARCHIVE_NAME = "iccv09Data.tar.gz"
DATASET_FOLDER_NAME = "iccv09Data"


def dataset_layout_exists(root_dir):
    root_dir = Path(root_dir)
    return (root_dir / "images").is_dir() and (root_dir / "labels").is_dir()


def ensure_dataset_ready(data_dir):
    data_dir = Path(data_dir)
    if data_dir.is_file():
        archive_path = data_dir
        extract_root = data_dir.parent
    else:
        archive_path = data_dir / DATASET_ARCHIVE_NAME
        extract_root = data_dir

    direct_root = data_dir
    extracted_root = extract_root / DATASET_FOLDER_NAME

    if dataset_layout_exists(direct_root):
        return direct_root
    if dataset_layout_exists(extracted_root):
        return extracted_root

    if archive_path.is_file():
        print(f"Extracting dataset archive from {archive_path} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_root)
        if dataset_layout_exists(extracted_root):
            return extracted_root

    raise FileNotFoundError(
        "Stanford Background dataset not found. Expected either "
        f"'{data_dir}/images' + '{data_dir}/labels', or archive '{archive_path}'."
    )


class StanfordBackgroundDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_dir = self.root_dir / "images"
        self.label_dir = self.root_dir / "labels"

        image_files = sorted(self.image_dir.glob("*.jpg"))
        self.samples = []
        for image_path in image_files:
            label_path = self.label_dir / f"{image_path.stem}.regions.txt"
            if label_path.is_file():
                self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No valid image/label pairs found under {self.root_dir}")

        split_idx = int(0.8 * len(self.samples))
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        mask_np = np.loadtxt(mask_path, dtype=np.int64)
        mask = Image.fromarray(mask_np.astype(np.uint8))

        image = image.resize((256, 256))
        mask = mask.resize((256, 256), Image.NEAREST)

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).long()
        mask[mask >= 8] = 7

        return image, mask

def calculate_miou(preds, labels, num_classes=8):
    preds = preds.argmax(dim=1)
    miou = 0.0
    valid_classes = 0
    
    for c in range(num_classes):
        pred_c = (preds == c)
        label_c = (labels == c)
        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()
        
        # 只计算图片中实际存在，或模型预测出的类别的 IoU
        if union > 0:
            miou += intersection / union
            valid_classes += 1
            
    # 如果图片全是背景且预测全是背景，避免除以 0
    return (miou / valid_classes).item() if valid_classes > 0 else 0.0

def train(args):
    run_name = f"unet_{args.loss}_lr{args.lr}_ep{args.epochs}"
    try:
        wandb.init(project="hw2-unet-segmentation", name=run_name, config=args)
    except Exception as e:
        print("W&B login failed. Falling back to offline mode for logging.")
        wandb.init(mode="offline", project="hw2-unet-segmentation", name=run_name, config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_root = ensure_dataset_ready(args.data_dir)
    print(f"Using dataset root: {dataset_root}")

    train_dataset = StanfordBackgroundDataset(root_dir=dataset_root, split="train")
    val_dataset = StanfordBackgroundDataset(root_dir=dataset_root, split="val")

    # Windows 环境下 num_workers 建议设为 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = UNet(in_channels=3, out_channels=8).to(device)
    
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'dice':
        criterion = DiceLoss()
    elif args.loss == 'combined':
        criterion = CombinedLoss()
        
    # 降低初始学习率至 2e-4，并配合正则化
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.2, weight_decay=1e-4)
    
    # 使用 ReduceLROnPlateau：当验证集 mIoU 不再提升时，自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
    best_miou = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 引入梯度裁剪，防止梯度爆炸导致的震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        total_miou = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                total_miou += calculate_miou(outputs, labels, 8)
                
        val_loss = val_loss / len(val_loader)
        val_miou = total_miou / len(val_loader)
        
        # 将验证集 mIoU 传给调度器，用于自适应调整学习率
        scheduler.step(val_miou)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "learning_rate": optimizer.param_groups[-1]['lr'],
            "epoch": epoch + 1
        })
        
        if val_miou > best_miou:
            best_miou = val_miou
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/unet_{args.loss}_best.pth')
            
    print(f"Training completed. Best Validation mIoU: {best_miou:.4f}")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HW2 Task 3: U-Net Segmentation")
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset directory or iccv09Data.tar.gz')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'dice', 'combined'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    
    args = parser.parse_args()
    train(args)
