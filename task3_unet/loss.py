import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        手动实现的 Dice Loss
        :param logits: 网络输出的预测，形状为 (BatchSize, NumClasses, H, W)
        :param targets: 真实的标签掩码，形状为 (BatchSize, H, W)
        """
        num_classes = logits.size(1)
        
        # 1. 使用 softmax 将 logits 转换为概率分布
        probs = F.softmax(logits, dim=1)
        
        # 2. 将真实的类别索引目标(0, 1, ..., 7) 转换为 One-Hot 编码形式
        #    原 targets 形状: (B, H, W)
        #    One-hot 形状: (B, H, W, C) -> permute 后变成 (B, C, H, W) 与 probs 对齐
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 3. 展平张量，方便进行交集和并集的点乘计算
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)
        
        # 4. 计算交集 (Intersection) 和 并集 (Union)
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        # 5. 计算 Dice 系数 (加 smooth 防止分母为 0)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 6. Dice Loss = 1 - Dice Coefficient (因为我们要最小化 Loss，所以用 1 减去)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        # 组合损失：将交叉熵损失与 Dice 损失按权重相加
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
