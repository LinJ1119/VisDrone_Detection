"""
模型部署导出入口。

用法:
    python export.py --model best.pt --format onnx
    python export.py --model best.pt --format engine
    python export.py --model best.pt --format onnx --benchmark
"""

import argparse
import logging
import os
import sys

# Windows 页面文件缓解
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

from pathlib import Path

from export.exporter import export_model
from export.inference_benchmark import run_full_benchmark


def parse_args():
    p = argparse.ArgumentParser(description="VisDrone YOLOv8n 模型部署导出")
    p.add_argument("--model", type=str, required=True, help="模型路径")
    p.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine"],
                   help="导出格式 (onnx | engine)")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset 版本")
    p.add_argument("--benchmark", action="store_true", help="导出后运行推理速度对比")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # 1) 导出
    exported_path = export_model(
        model_path=args.model,
        format=args.format,
        imgsz=args.imgsz,
        opset=args.opset,
    )

    print(f"\n  导出完成: {exported_path}")

    # 2) 基准测试
    if args.benchmark:
        onnx_path = exported_path if args.format == "onnx" else None
        engine_path = exported_path if args.format == "engine" else None
        report = run_full_benchmark(
            args.model, onnx_path=onnx_path, engine_path=engine_path,
            imgsz=args.imgsz,
        )
        # 保存对比报告
        report_path = Path(exported_path).with_suffix(".benchmark.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 推理速度对比\n\n")
            f.write(f"| 模式 | 耗时 (ms) |\n")
            f.write(f"|------|----------:|\n")
            f.write(f"| PyTorch FP32 | {report['pytorch_fp32_ms']:.3f} |\n")
            if report.get('onnx_fp32_ms'):
                f.write(f"| ONNX FP32 | {report['onnx_fp32_ms']:.3f} |\n")
            if report.get('tensorrt_fp32_ms'):
                f.write(f"| TensorRT FP32 | {report['tensorrt_fp32_ms']:.3f} |\n")
            if report.get('trt_speedup_vs_pytorch'):
                f.write(f"\nTensorRT 加速比: **{report['trt_speedup_vs_pytorch']:.2f}×** "
                        f"{report.get('trt_grade', '')}\n")
        print(f"  对比报告: {report_path}")


if __name__ == "__main__":
    main()
