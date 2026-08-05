# -*- coding: utf-8 -*-
"""Standalone benchmark: PyTorch vs ONNX (CPU)."""
import os, sys, time
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import numpy as np

MODEL_PT = "runs/train/visdrone_baseline/weights/best.pt"
MODEL_ONNX = "runs/train/visdrone_baseline/weights/best.onnx"
IMG_SIZE = 640
WARMUP = 10
RUNS = 100

print("=" * 60)
print("Inference speed benchmark")
print("=" * 60)

# --- PyTorch ---
print("\n[1] PyTorch FP32")
from ultralytics import YOLO
model = YOLO(MODEL_PT)
for _ in range(WARMUP):
    dummy = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
times = []
for _ in range(RUNS):
    dummy = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    model.predict(dummy, imgsz=IMG_SIZE, verbose=False)
    times.append((time.perf_counter() - t0) * 1000)
pt_ms = np.mean(times)
pt_std = np.std(times)
print(f"  {pt_ms:.2f} ms +/- {pt_std:.2f}  (warmup={WARMUP}, runs={RUNS})")
print(f"  Throughput: {1000/pt_ms:.0f} FPS")

# --- ONNX CPU ---
print("\n[2] ONNX CPU")
try:
    import onnxruntime as ort
    session = ort.InferenceSession(MODEL_ONNX, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for _ in range(WARMUP):
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        session.run([output_name], {input_name: dummy})
    times = []
    for _ in range(RUNS):
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        t0 = time.perf_counter()
        session.run([output_name], {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    onnx_ms = np.mean(times)
    onnx_std = np.std(times)
    print(f"  {onnx_ms:.2f} ms +/- {onnx_std:.2f}  (cpu, warmup={WARMUP}, runs={RUNS})")
    speedup = pt_ms / onnx_ms
    print(f"  vs PyTorch: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
except Exception as e:
    print(f"  Skipped: {e}")

# --- ONNX CUDA ---
print("\n[3] ONNX CUDA")
try:
    import onnxruntime as ort
    session = ort.InferenceSession(MODEL_ONNX, providers=["CUDAExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for _ in range(WARMUP):
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        session.run([output_name], {input_name: dummy})
    times = []
    for _ in range(RUNS):
        dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        t0 = time.perf_counter()
        session.run([output_name], {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    onnxcu_ms = np.mean(times)
    onnxcu_std = np.std(times)
    print(f"  {onnxcu_ms:.2f} ms +/- {onnxcu_std:.2f}  (cuda, warmup={WARMUP}, runs={RUNS})")
    speedup2 = pt_ms / onnxcu_ms
    print(f"  vs PyTorch: {speedup2:.2f}x {'faster' if speedup2 > 1 else 'slower'}")
except Exception as e:
    print(f"  Skipped: {e}")

print(f"\nDone.")
print(f"Model: {os.path.getsize(MODEL_ONNX)/1e6:.1f} MB ONNX | {os.path.getsize(MODEL_PT)/1e6:.1f} MB PT")
