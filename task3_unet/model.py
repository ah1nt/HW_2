import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 引入 Dropout (丢弃率0.2) 防止网络在小数据集上过拟合
            nn.Dropout2d(0.2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=8): # Stanford Background Dataset has 8 classes
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder (下采样)
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
        
        x4 = self.down4(p3)
        p4 = self.pool4(x4)
        
        # Bottleneck (桥接层)
        b = self.bottleneck(p4)
        
        # Decoder (上采样 + Skip Connection)
        u1 = self.up1(b)
        diffY = x4.size()[2] - u1.size()[2]
        diffX = x4.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Skip Connection: 将编码器对应层的特征 (x4) 与解码器上采样后的特征 (u1) 在通道维度(dim=1)进行拼接
        c1 = torch.cat([x4, u1], dim=1)
        u1 = self.up_conv1(c1)
        
        u2 = self.up2(u1)
        diffY = x3.size()[2] - u2.size()[2]
        diffX = x3.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Skip Connection
        c2 = torch.cat([x3, u2], dim=1)
        u2 = self.up_conv2(c2)
        
        u3 = self.up3(u2)
        diffY = x2.size()[2] - u3.size()[2]
        diffX = x2.size()[3] - u3.size()[3]
        u3 = F.pad(u3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Skip Connection
        c3 = torch.cat([x2, u3], dim=1)
        u3 = self.up_conv3(c3)
        
        u4 = self.up4(u3)
        diffY = x1.size()[2] - u4.size()[2]
        diffX = x1.size()[3] - u4.size()[3]
        u4 = F.pad(u4, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # Skip Connection
        c4 = torch.cat([x1, u4], dim=1)
        u4 = self.up_conv4(c4)
        
        out = self.out_conv(u4)
        return out
