import subprocess
import sys

def run_experiment(name, cmd_args):
    print(f"\n{'='*60}")
    print(f"🚀 正在运行实验: {name}")
    print(f"💻 执行命令: python train.py {' '.join(cmd_args)}")
    print(f"{'='*60}\n")
    
    cmd = [sys.executable, "train.py"] + cmd_args
    subprocess.run(cmd)

if __name__ == '__main__':
    # ==========================================
    # 实验 (1): Baseline - 使用预训练模型微调
    # ==========================================
    run_experiment(
        "1. Baseline (ResNet18 预训练微调)",
        ["--model", "resnet18", "--pretrained", "--epochs", "20", "--lr", "0.001"]
    )

    # ==========================================
    # 实验 (3): 预训练消融实验 - 从零开始训练
    # ==========================================
    run_experiment(
        "3. 预训练消融 (ResNet18 随机初始化)",
        ["--model", "resnet18", "--epochs", "20", "--lr", "0.001"]
    )

    # ==========================================
    # 实验 (2): 超参数分析 - 测试不同的学习率和训练步数
    # ==========================================
    run_experiment(
        "2. 超参数分析 (较小学习率 lr=1e-4)",
        ["--model", "resnet18", "--pretrained", "--epochs", "20", "--lr", "0.0001"]
    )
    run_experiment(
        "2. 超参数分析 (较大学习率 lr=5e-3)",
        ["--model", "resnet18", "--pretrained", "--epochs", "20", "--lr", "0.005"]
    )
    run_experiment(
        "2. 超参数分析 (增加训练轮数 epochs=30)",
        ["--model", "resnet18", "--pretrained", "--epochs", "30", "--lr", "0.001"]
    )

    # ==========================================
    # 实验 (4): 引入注意力机制 - SE-block 与 Vision Transformer
    # ==========================================
    run_experiment(
        "4. 注意力机制 (ResNet18 + SE-block)",
        ["--model", "resnet18_se", "--pretrained", "--epochs", "20", "--lr", "0.001"]
    )
    run_experiment(
        "4. 注意力机制 (Swin-T Vision Transformer)",
        ["--model", "vit", "--pretrained", "--epochs", "20", "--lr", "0.001"]
    )

    print("\n🎉 所有实验运行完毕！请前往 W&B 面板查看实验对比曲线，并根据数据完成您的作业报告。")
