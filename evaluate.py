"""
评估启动入口。

用法:
    python evaluate.py --model best.pt --data data.yaml
    python evaluate.py --model best.pt --data data.yaml --plots
    python evaluate.py --model best.pt --data data.yaml --batch_size 8
"""

import os

# Windows 页面文件较小导致 cuDNN DLL 加载失败时的缓解措施
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import argparse
import logging
from pathlib import Path

from eval.evaluator import run_eval
from eval.visualizer import plot_curves


def parse_args():
    p = argparse.ArgumentParser(description="VisDrone YOLOv8n 模型评估")
    p.add_argument("--model", type=str, required=True, help="模型 checkpoint 路径（如 best.pt）")
    p.add_argument("--data", type=str, default="datasets/visdrone/data.yaml", help="data.yaml 路径")
    p.add_argument("--batch_size", type=int, default=4, help="评估批大小")
    p.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    p.add_argument("--plots", action="store_true", help="生成 PR/F1/混淆矩阵曲线图")
    p.add_argument("--output", type=str, default=None, help="曲线输出目录（默认 runs/eval/<model_name>/curves）")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # 1) 运行评估
    result = run_eval(
        model_path=args.model,
        data_yaml=args.data,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
    )

    # 2) 输出摘要
    print()
    print(f"{'='*60}")
    print(f"  评估完成")
    print(f"{'='*60}")
    print(f"  模型:     {args.model}")
    print(f"  mAP@50:   {result.mAP50:.4f}")
    print(f"  mAP@50-95:{result.mAP50_95:.4f}")
    print(f"  Precision:{result.precision:.4f}")
    print(f"  Recall:   {result.recall:.4f}")
    print()
    print(f"  每类 mAP@50:")
    for cls_name in sorted(result.per_class_AP.keys()):
        ap = result.per_class_AP[cls_name]
        fnr = result.fn_rate_per_class.get(cls_name, 0.0)
        flag = " ⚠️" if fnr > 0.4 else ""
        print(f"    {cls_name:20s} {ap:.4f}{flag}")

    if result.skipped_classes:
        print(f"\n  跳过类别（无验证样本）: {result.skipped_classes}")
    if result.high_fn_classes:
        print(f"\n  高漏检率类别（FN>40%）: {', '.join(result.high_fn_classes)}")

    # 3) 生成曲线图
    if args.plots:
        output_dir = args.output or os.path.join(
            "runs/eval", Path(args.model).stem, "curves"
        )
        paths = plot_curves(result, output_dir=output_dir)
        print(f"\n  曲线图已保存 ({len(paths)} 张):")
        for p in paths:
            print(f"    {p}")

    print(f"\n  完成.")


if __name__ == "__main__":
    main()
