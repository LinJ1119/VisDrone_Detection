"""Test the FastAPI detection service."""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

from app import _model, _model_type, _infer
from PIL import Image

# Test 1: model loaded
print(f"[1] Model: {_model_type}")
assert _model is not None, "Model failed to load"
print("    OK")

# Test 2: predict on a real image
test_img = "D:/Data/VisDrone/test-dev/images/0000004_00001_d_0000027.jpg"
if os.path.isfile(test_img):
    t0 = time.perf_counter()
    dets = _infer(Image.open(test_img))
    t_ms = (time.perf_counter() - t0) * 1000
    print(f"[2] Predict: {len(dets)} detections in {t_ms:.1f} ms")
    for d in dets[:5]:
        print(f"    {d['class_name']:20s} conf={d['confidence']:.3f}  xyxy={d['xyxy']}")
    print("    OK")
else:
    print(f"[2] SKIP: test image not found")

print("\nAll tests OK. Start server: uvicorn app:app --host 0.0.0.0 --port 8000")
