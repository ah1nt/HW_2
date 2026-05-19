# 深度学习与空间智能 - HW2

本仓库包含《深度学习与空间智能》课程的第二次作业任务1、3的代码，涵盖了图像分类以及图像语义分割两个任务。

## 环境配置

推荐使用 Python 3.8+ 及 Anaconda 创建虚拟环境：

```bash
conda create -n hw2 python=3.9
conda activate hw2
pip install -r requirements.txt
```

如果需要使用 GPU 训练，请确保安装了与 CUDA 版本对应的 PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 数据集说明

首次运行前，请先按下面说明准备数据集。

### 任务1数据集：102 Category Flower Dataset

任务1使用 `torchvision.datasets.Flowers102` 加载数据，代码中已开启 `download=True`，因此在网络正常的情况下无需手动放置图片文件。


- 默认下载目录：`task1_flower/data/`
- 官方划分：
  - 训练集：`train`
  - 验证集：`val`

使用方式：

```bash
cd task1_flower
python run_experiments.py
```

如果自动下载失败，可以手动重新运行训练脚本，或检查网络后再次执行；数据成功下载后会缓存在 `task1_flower/data/` 下。

### 任务3数据集：Stanford Background Dataset

任务3使用的是 **Stanford Background Dataset**。当前代码默认读取 `task3_unet/data/` 下的压缩包 `iccv09Data.tar.gz`，并在首次运行时自动解压；如果你已经手动解压好数据集，也可以直接使用解压后的目录。

- 默认数据位置：`task3_unet/data/iccv09Data.tar.gz`
- 自动解压后的目录：`task3_unet/data/iccv09Data/`

代码实际识别的目录结构如下：

```text
task3_unet/
└── data/
    ├── iccv09Data.tar.gz
    └── iccv09Data/
        ├── images/
        │   ├── 0000001.jpg
        │   ├── 0000002.jpg
        │   └── ...
        └── labels/
            ├── 0000001.regions.txt
            ├── 0000001.layers.txt
            ├── 0000001.surfaces.txt
            ├── 0000002.regions.txt
            └── ...
```

说明：

- `images/` 中存放原始 RGB 图像，文件后缀为 `.jpg`
- `labels/` 中包含多种标注文件，训练代码只使用其中的 `*.regions.txt`
- 代码会根据图片名自动匹配同名标注文件，例如：
  - `0000001.jpg`
  - `0000001.regions.txt`
- 训练脚本内部会按排序后的文件列表自动进行 `80% / 20%` 的训练集与验证集划分
- 如果 `task3_unet/data/iccv09Data/` 不存在，但 `task3_unet/data/iccv09Data.tar.gz` 存在，代码会先自动解压再开始训练

准备好数据后，运行：

```bash
cd task3_unet
python run_experiments.py
```

如果你想把数据放在别的位置，也可以在运行训练脚本时手动指定目录，传入的路径既可以是压缩包所在目录，也可以直接是解压后的 `iccv09Data/` 目录：

```bash
python train.py --data_dir /path/to/data --loss ce --epochs 30 --batch_size 8 --lr 0.001
```

---

## 任务1：微调在 ImageNet 上预训练的卷积神经网络实现宠物识别 (102 Category Flower Dataset)

### 运行说明

进入 `task1_flower` 目录：
```bash
cd task1_flower
```

直接运行：
```bash
python run_experiments.py
```
这将会自动在 W&B 上记录 7 组实验的指标，包括：Baseline 微调、预训练消融、不同超参数对比，以及引入 SE-block / Vision Transformer 的性能对比。

---


## 任务3：从零搭建与损失函数工程：图像分割模型的像素级训练

### 运行说明

进入 `task3_unet` 目录：
```bash
cd task3_unet
```

直接运行：
```bash
python run_experiments.py
```
这将会自动运行三组实验：
1. 仅使用标准的 Cross-Entropy Loss
2. 仅使用手动实现的 Dice Loss
3. 组合损失 (CE Loss + Dice Loss)

训练脚本会计算验证集上的 mIoU 指标，并自动将数据记录至 W&B 面板。

---
