"""
推理速度基准测试。

比较 PyTorch / ONNX / TensorRT 三种模式的推理耗时。
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def benchmark_pytorch(model_path, imgsz=640, warmup=10, runs=100):
    """PyTorch FP32 推理基准。

    Returns:
        (avg_ms, std_ms)
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    # 预热
    for _ in range(warmup):
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(dummy, imgsz=imgsz, verbose=False)

    times = []
    for _ in range(runs):
        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def benchmark_onnx(onnx_path, imgsz=640, warmup=10, runs=100):
    """ONNX Runtime 推理基准。

    Returns:
        (avg_ms, std_ms) or (None, None) if ONNX Runtime unavailable.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("ONNX Runtime 未安装，跳过 ONNX 基准")
        return None, None

    # 加载 session（直接 CPU，避免 CUDA context 冲突）
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        logger.warning("ONNX Runtime 加载失败: %s", e)
        return None, None

    # 预热
    dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    input_name = session.get_inputs()[0].name
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(runs):
        dummy = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def benchmark_tensorrt(engine_path, imgsz=640, warmup=10, runs=100):
    """TensorRT 推理基准。

    Returns:
        (avg_ms, std_ms) or (None, None) if TensorRT unavailable.
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
    except ImportError:
        logger.warning("TensorRT / PyCUDA 未安装，跳过 TensorRT 基准")
        return None, None

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Allocate buffers
    input_shape = (1, 3, imgsz, imgsz)
    dummy = np.random.randn(*input_shape).astype(np.float32)
    d_input = cuda.mem_alloc(dummy.nbytes)
    output_shape = (1, 10, 8400)  # approximate for YOLOv8n
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod(d_input, dummy)
        context.execute_v2(bindings=[int(d_input), int(d_output)])

    times = []
    for _ in range(runs):
        cuda.memcpy_htod(d_input, dummy)
        t0 = time.perf_counter()
        context.execute_v2(bindings=[int(d_input), int(d_output)])
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.std(times))


def run_full_benchmark(model_path, onnx_path=None, engine_path=None,
                       imgsz=640, warmup=10, runs=100):
    """运行完整推理速度对比，返回结构化报告。

    Returns:
        dict: {"pytorch_fp32_ms", "onnx_fp32_ms", "tensorrt_fp32_ms",
               "trt_speedup", "trt_grade"}
    """
    logger.info("运行推理速度对比...")

    # PyTorch 基准
    pt_ms, pt_std = benchmark_pytorch(model_path, imgsz, warmup, runs)
    report = {"pytorch_fp32_ms": round(pt_ms, 4)}

    # ONNX 基准
    onnx_ms, onnx_std = None, None
    if onnx_path and Path(onnx_path).is_file():
        onnx_ms, onnx_std = benchmark_onnx(onnx_path, imgsz, warmup, runs)
    report["onnx_fp32_ms"] = round(onnx_ms, 4) if onnx_ms else None

    # TensorRT 基准
    trt_ms, trt_std = None, None
    if engine_path and Path(engine_path).is_file():
        trt_ms, trt_std = benchmark_tensorrt(engine_path, imgsz, warmup, runs)
    report["tensorrt_fp32_ms"] = round(trt_ms, 4) if trt_ms else None

    # 分级评估
    if trt_ms:
        speedup = pt_ms / trt_ms
        report["trt_speedup_vs_pytorch"] = round(speedup, 3)
        if speedup >= 1.2:
            grade = "✅ 达标"
        elif speedup >= 1.1:
            grade = "⚠️ Pascal 架构层融合收益低于预期，TensorRT 功能正常但加速有限"
        elif speedup >= 1.0:
            grade = "⚠️ TensorRT 未产生显著加速效果，建议使用 PyTorch 原生推理"
        else:
            grade = "❌ TensorRT 导出异常，请检查引擎文件"
        report["trt_grade"] = grade

    _print_report(report)
    return report


def _print_report(report):
    logger.info("=" * 50)
    logger.info("推理速度对比")
    logger.info("  PyTorch FP32:  %.3f ms", report["pytorch_fp32_ms"])
    if report.get("onnx_fp32_ms"):
        logger.info("  ONNX FP32:     %.3f ms", report["onnx_fp32_ms"])
    if report.get("tensorrt_fp32_ms"):
        logger.info("  TensorRT FP32: %.3f ms", report["tensorrt_fp32_ms"])
    if report.get("trt_speedup_vs_pytorch"):
        logger.info("  TRT 加速比:   %.2f× %s",
                    report["trt_speedup_vs_pytorch"], report.get("trt_grade", ""))
    logger.info("=" * 50)


from pathlib import Path  # noqa: E402
