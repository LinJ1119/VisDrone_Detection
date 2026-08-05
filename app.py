"""
VisDrone 目标检测推理服务 - FastAPI 封装.

启动:
    python app.py
    uvicorn app:app --host 0.0.0.0 --port 8000

接口:
    POST /predict  - 上传图像, 返回 JSON 检测结果
    GET  /health   - 健康检查
    GET  /         - 服务信息
"""

# NumPy 兼容性补丁 (TensorRT 8.5 + NumPy >=1.24)
import numpy as np
for _attr in ("bool", "int", "float", "complex", "object", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(__builtins__, _attr, None))

import io
import os
import time
from typing import Dict, List

from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile

# ── 模型路径 ─────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "runs", "train", "visdrone_baseline", "weights")
ENGINE_PATH = os.path.join(MODEL_DIR, "best.engine")
PT_PATH = os.path.join(MODEL_DIR, "best.pt")

# TensorRT lib PATH
TRT_LIB = r"E:\Development\TensorRT-8.5.3.1\lib"
if os.path.isdir(TRT_LIB):
    os.environ["PATH"] = TRT_LIB + ";" + os.environ.get("PATH", "")

# ── 常量 ─────────────────────────────────────────────
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMGSZ = 640


# ── 模型加载 (模块级) ───────────────────────────────

def _init_model():
    """优先 TensorRT engine, 回退 PyTorch."""
    if os.path.isfile(ENGINE_PATH):
        try:
            from ultralytics import YOLO
            model = YOLO(ENGINE_PATH, task="detect")
            print(f"[INIT] TensorRT engine: {ENGINE_PATH} ({os.path.getsize(ENGINE_PATH)/1e6:.1f} MB)")
            return model, "tensorrt"
        except Exception as e:
            print(f"[INIT] TensorRT failed ({e}), fallback to PyTorch")
    if os.path.isfile(PT_PATH):
        from ultralytics import YOLO
        model = YOLO(PT_PATH)
        print(f"[INIT] PyTorch model: {PT_PATH} ({os.path.getsize(PT_PATH)/1e6:.1f} MB)")
        return model, "pytorch"
    raise RuntimeError(f"No model found: {ENGINE_PATH} or {PT_PATH}")


_model, _model_type = _init_model()


# ── FastAPI 应用 ─────────────────────────────────────

app = FastAPI(
    title="VisDrone Detection API",
    description="YOLOv8n 10-class drone imagery object detection",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "service": "VisDrone Detection API",
        "version": "1.0.0",
        "model_type": _model_type,
        "classes": CLASS_NAMES,
        "endpoints": {
            "predict": "POST /predict  (upload image -> JSON)",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_type": _model_type}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload image, return detection results."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Only image files accepted, got: {file.content_type}")
    # content_type may be None from some clients — accept anyway, PIL will validate

    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))
        image.load()
    except Exception:
        raise HTTPException(400, "Image corrupt or unsupported format")

    t0 = time.perf_counter()
    try:
        detections = _infer(image)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    t_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "image_name": file.filename or "upload",
        "model_type": _model_type,
        "inference_time_ms": t_ms,
        "num_detections": len(detections),
        "detections": detections,
    }


# ── 推理核心 ─────────────────────────────────────────

def _infer(image: Image.Image) -> List[Dict]:
    results = _model.predict(image, imgsz=IMGSZ, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i])
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}",
                "confidence": round(float(r.boxes.conf[i]), 4),
                "xyxy": [round(float(v), 2) for v in r.boxes.xyxy[i].tolist()],
            })
    return detections


# ── 启动入口 ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
