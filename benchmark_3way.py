import os
os.environ["PATH"] = r"E:\Development\TensorRT-8.5.3.1\lib;" + os.environ["PATH"]
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import time
import numpy as np
# Patch for TensorRT 8.5 + NumPy >= 1.24 (np.bool removed)
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "str"):
    np.str = str
import torch

MODEL_PT  = "runs/train/visdrone_baseline/weights/best.pt"
MODEL_ONNX = "runs/train/visdrone_baseline/weights/best.onnx"
MODEL_ENGINE = "runs/train/visdrone_baseline/weights/best.engine"
IMGSZ = 640
WARMUP = 30
RUNS = 200

print("=" * 60)
print("PyTorch vs ONNX vs TensorRT  (FP32, 640x640)")
print("=" * 60)

# ---- PyTorch ----
from ultralytics import YOLO
model = YOLO(MODEL_PT)
for _ in range(WARMUP):
    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=IMGSZ, verbose=False)
torch.cuda.synchronize()
times = []
for _ in range(RUNS):
    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    model.predict(dummy, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)
pt_ms = np.mean(times)
pt_std = np.std(times)
print(f"\n[1] PyTorch FP32")
print(f"    {pt_ms:.2f} ms +/- {pt_std:.3f}  |  {1000/pt_ms:.0f} FPS")

# ---- ONNX CUDA ----
import onnxruntime as ort
session = ort.InferenceSession(MODEL_ONNX, providers=["CUDAExecutionProvider"])
iname = session.get_inputs()[0].name
oname = session.get_outputs()[0].name
for _ in range(WARMUP):
    d = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    session.run([oname], {iname: d})
times = []
for _ in range(RUNS):
    d = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    t0 = time.perf_counter()
    session.run([oname], {iname: d})
    times.append((time.perf_counter() - t0) * 1000)
onnx_ms = np.mean(times)
onnx_std = np.std(times)
speedup = pt_ms / onnx_ms
print(f"[2] ONNX CUDA")
print(f"    {onnx_ms:.2f} ms +/- {onnx_std:.3f}  |  {1000/onnx_ms:.0f} FPS  |  {speedup:.2f}x vs PT")

# ---- TensorRT ----
# Use ultralytics internal predict to load engine properly
trt_model = YOLO(MODEL_ENGINE)
for _ in range(WARMUP):
    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    trt_model.predict(dummy, imgsz=IMGSZ, verbose=False)
torch.cuda.synchronize()
times = []
for _ in range(RUNS):
    dummy = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    trt_model.predict(dummy, imgsz=IMGSZ, verbose=False)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)
trt_ms = np.mean(times)
trt_std = np.std(times)
speedup2 = pt_ms / trt_ms
print(f"[3] TensorRT FP32")
print(f"    {trt_ms:.2f} ms +/- {trt_std:.3f}  |  {1000/trt_ms:.0f} FPS  |  {speedup2:.2f}x vs PT")

print(f"\n{'=' * 60}")
print(f"Summary:")
print(f"  PyTorch:  {pt_ms:.2f} ms  ({1000/pt_ms:.0f} FPS)")
print(f"  ONNX:     {onnx_ms:.2f} ms  ({1000/onnx_ms:.0f} FPS)  ({speedup:.2f}x)")
print(f"  TensorRT: {trt_ms:.2f} ms  ({1000/trt_ms:.0f} FPS)  ({speedup2:.2f}x)")
print(f"\n  Model sizes: PT={os.path.getsize(MODEL_PT)/1e6:.1f}MB  ONNX={os.path.getsize(MODEL_ONNX)/1e6:.1f}MB  TRT={os.path.getsize(MODEL_ENGINE)/1e6:.1f}MB")

grade = ""
if speedup2 >= 1.20: grade = "[PASS] >= 1.20x"
elif speedup2 >= 1.10: grade = "[WARN] 1.10-1.20x, limited by Pascal arch"
elif speedup2 >= 1.00: grade = "[WARN] < 1.10x, suggest PyTorch native"
else: grade = "[FAIL] < 1.00x, export anomaly"
print(f"  TRT grade: {grade}")
print(f"{'=' * 60}")
