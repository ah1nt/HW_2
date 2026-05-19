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
    # 任务3 (3): 损失函数工程 - 三组对比实验
    # ==========================================
    
    # 实验 A: 仅使用标准交叉熵损失 (CE Loss)
    run_experiment(
        "1. 仅使用 Cross-Entropy Loss",
        ["--loss", "ce", "--epochs", "30", "--batch_size", "8", "--lr", "0.001"]
    )

    # 实验 B: 仅使用手动实现的 Dice Loss
    run_experiment(
        "2. 仅使用手动实现的 Dice Loss",
        ["--loss", "dice", "--epochs", "30", "--batch_size", "8", "--lr", "0.001"]
    )

    # 实验 C: 组合损失 (CE Loss + Dice Loss)
    run_experiment(
        "3. 组合损失 (Cross-Entropy Loss + Dice Loss)",
        ["--loss", "combined", "--epochs", "30", "--batch_size", "8", "--lr", "0.001"]
    )

    print("\n🎉 所有分割训练实验运行完毕！请前往 W&B 面板查看验证集 mIoU 的对比曲线。")
