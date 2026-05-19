import os
import cv2
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

def train_yolo(args):
    print("\n--- 1. 开始训练专属 YOLOv8 目标检测模型 ---")
    base_dir = Path(__file__).resolve().parent
    
    # 【避坑】加载预训练模型是关键，不要从头随机初始化！
    model = YOLO('yolov8n.pt')
    
    yaml_path = base_dir / "data" / "trafic_data" / "data_1.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"找不到数据集配置文件: {yaml_path}。请先将数据集放到 data/trafic_data/ 目录下。"
        )
    
    print(f"正在读取数据集配置: {yaml_path}")
    print(f"配置的 Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    
    # 开始微调训练，ultralytics 库在内部会接管损失函数、优化器和可视化
    device_to_use = 'cpu' if args.force_cpu else (0 if torch.cuda.is_available() else 'cpu')
    
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch_size,
        name=args.run_name,
        device=device_to_use,
        project=str(base_dir / 'runs' / 'detect' / 'hw2-yolo'),
        exist_ok=True
    )
    print("✅ YOLO 模型微调完成！")
    return model

def track_and_count(args):
    print("\n--- 2. 开始视频多目标跟踪与越线计数 ---")
    base_dir = Path(__file__).resolve().parent
    
    # 1. 优先读取手动下载的 best.pt，其次读取本地训练结果
    weights_path = base_dir / 'weights' / 'best.pt'
    trained_best = base_dir / 'runs' / 'detect' / 'hw2-yolo' / args.run_name / 'weights' / 'best.pt'
    if not weights_path.exists() and trained_best.exists():
        weights_path = trained_best

    if not weights_path.exists():
        print(f"⚠️ 未找到 best.pt ({weights_path})，退回使用官方 yolov8n.pt 进行演示！")
        model = YOLO('yolov8n.pt')
    else:
        print(f"✅ 加载专属微调权重: {weights_path}")
        model = YOLO(weights_path)
    # 强制使用 CPU 进行追踪，绕过 RTX 5060 兼容性问题
    model.to('cpu')

    # 2. 视频 I/O 设置
    input_video_path = Path(args.video_path)
    if not input_video_path.is_absolute():
        input_video_path = base_dir / input_video_path
    if not input_video_path.exists():
        raise FileNotFoundError(f"找不到测试视频: {input_video_path}。请先准备一段 10-30 秒的 mp4 视频并放到该路径下！")

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {input_video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    output_dir = base_dir / "runs" / "track" / "hw2-yolo" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "tracked_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频文件: {output_video_path}")

    # 3. 核心：越线计数逻辑的防坑状态机
    # 设定一条竖直越线：画面宽度按 line_ratio 比例
    line_ratio = max(0.05, min(0.95, args.line_ratio))
    line_x = int(width * line_ratio)
    
    # 历史轨迹字典 {track_id: [prev_x, current_x]}
    track_history = {}
    # 已经计算过越线的 ID 集合（防重复计数）
    counted_ids = set()
    # 总越线车辆数
    total_crossed = 0

    print(f"设定越线阈值 X = {line_x} (line_ratio={line_ratio:.2f})，开始逐帧推理...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # 使用 YOLO 内置的 ByteTrack 或 DeepSORT 跟踪器进行推理
        # tracker="bytetrack.yaml" 是一个非常稳定抗遮挡的官方追踪器配置
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # 在画面上画出越界线（竖线）
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 3)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                # 计算当前检测框的中心点
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # 更新轨迹状态机
                if track_id not in track_history:
                    track_history[track_id] = [cx, cx]
                else:
                    track_history[track_id][0] = track_history[track_id][1]
                    track_history[track_id][1] = cx
                
                prev_x = track_history[track_id][0]
                curr_x = track_history[track_id][1]
                
                # 双向越线判定：只要跨过线且未计数就加一
                if (prev_x - line_x) * (curr_x - line_x) < 0 and track_id not in counted_ids:
                    total_crossed += 1
                    counted_ids.add(track_id)
                    print(f"🚗 [Frame {frame_idx}] 目标 ID: {track_id} 越线！当前总数: {total_crossed}")
                
                # 绘制中心点和轨迹标识
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                label = f"ID:{track_id} Cls:{model.names[cls]}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 绘制实时统计信息看板
        cv2.rectangle(frame, (20, 20), (400, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Crossed: {total_crossed}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 视频推理完成！共检测到越线车辆数: {total_crossed}")
    print(f"✅ 结果视频已保存至: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "track"], required=True, help="选择执行训练还是视频追踪")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force_cpu", action="store_true", help="强制使用 CPU 训练以避开最新显卡兼容性问题")
    parser.add_argument("--run_name", type=str, default="yolo_road_vehicle")
    parser.add_argument("--video_path", type=str, default="test_video.mp4", help="Task 2(2) 要求的一段测试视频路径")
    parser.add_argument("--line_ratio", type=float, default=0.6, help="越线位置比例，范围建议 0.05~0.95")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_yolo(args)
    elif args.mode == "track":
        track_and_count(args)
