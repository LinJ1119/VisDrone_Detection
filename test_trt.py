import os
os.environ["PATH"] = r"D:\ProgramData\TensorRT-8.5.3.1\lib;" + os.environ["PATH"]
import tensorrt as trt
print("TensorRT version:", trt.__version__)
