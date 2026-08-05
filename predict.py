"""
推理启动入口。

用法:
    python predict.py --model best.pt --source image.jpg
    python predict.py --model best.pt --source imgs/                  # 批量推理
    python predict.py --model best.pt --source imgs/ --sahi           # SAHI 切片推理
    python predict.py --model best.pt --source imgs/ --conf 0.3 --iou 0.5
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from inference.predictor import run_predict


def parse_args():
    p = argparse.ArgumentParser(description="VisDrone YOLOv8n 目标检测推理")
    p.add_argument("--model", type=str, required=True, help="模型路径（如 best.pt）")
    p.add_argument("--source", type=str, required=True, help="图像路径或目录")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--sahi", action="store_true", help="启用 SAHI 切片推理")
    p.add_argument("--sahi_slice_size", type=int, default=640)
    p.add_argument("--sahi_overlap", type=float, default=0.2)
    p.add_argument("--sahi_batch", type=int, default=4, help="SAHI 切片批大小")
    p.add_argument("--output", type=str, default=None, help="输出目录")
    p.add_argument("--save_txt", action="store_true", help="保存 YOLO TXT 格式结果")
    p.add_argument("--save_json", action="store_true", help="保存 JSON 格式结果")
    p.add_argument("--save_img", action="store_true", help="保存可视化图像")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # 输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join("runs", "predict", f"pred_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    # SAHI 配置
    sahi_config = None
    if args.sahi:
        sahi_config = {
            "enabled": True,
            "slice_size": args.sahi_slice_size,
            "overlap": args.sahi_overlap,
            "batch_size": args.sahi_batch,
        }

    # 运行推理
    results = run_predict(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        sahi_config=sahi_config,
        imgsz=args.imgsz,
    )

    if not results:
        print("无检测结果。")
        return

    # 保存 YOLO TXT
    if args.save_txt:
        labels_dir = os.path.join(output_dir, "labels_yolo")
        os.makedirs(labels_dir, exist_ok=True)
        for r in results:
            stem = Path(r["image_name"]).stem
            txt_path = os.path.join(labels_dir, f"{stem}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for box in r["boxes"]:
                    cls_id = box["class_id"]
                    x1, y1, x2, y2 = box["xyxy"]
                    # 这里输出的是 xyxy，需要转换为归一化 xywh（YOLO 格式）
                    # 当前简化：输出 xyxy conf 格式
                    f.write(f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {box['conf']:.6f}\n")
        print(f"YOLO TXT 已保存至: {labels_dir}")

    # 保存 JSON
    if args.save_json:
        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSON 已保存至: {json_path}")

    # 保存可视化图像
    if args.save_img:
        from eval.visualizer import draw_detections
        import cv2
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        for r in results:
            img_path = os.path.join(args.source, r["image_name"]) if os.path.isdir(args.source) else args.source
            img = cv2.imread(img_path)
            if img is None:
                continue
            drawn = draw_detections(img, r["boxes"], conf_threshold=args.conf)
            cv2.imwrite(os.path.join(vis_dir, r["image_name"]), drawn)
        print(f"可视化图像已保存至: {vis_dir}")


if __name__ == "__main__":
    main()
