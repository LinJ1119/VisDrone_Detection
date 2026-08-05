import os
os.environ["PATH"] = r"D:\ProgramData\TensorRT-8.5.3.1\lib;" + os.environ["PATH"]

from export.exporter import export_model

model = "runs/train/visdrone_baseline/weights/best.pt"
engine_path = export_model(model, format="engine", imgsz=640)
print("Engine:", engine_path)
