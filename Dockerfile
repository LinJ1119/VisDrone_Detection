# VisDrone 目标检测推理服务 Docker 镜像
# 构建: docker build -t visdrone-detection .
# 运行: docker run -p 8000:8000 --gpus all visdrone-detection

FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# apt 国内镜像加速
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.8 /usr/bin/python

# pip 升级 + 国内镜像
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

# 工作目录
WORKDIR /app

# 安装 PyTorch (本地 wheel, 无需下载)
COPY vendor/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl \
    && rm /tmp/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl

# 安装 torchvision (4.6MB, 从 PyTorch 官方源)
RUN pip install --no-cache-dir torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 安装其余 Python 依赖
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# 复制模型权重
RUN mkdir -p runs/train/visdrone_baseline/weights
COPY runs/train/visdrone_baseline/weights/best.pt runs/train/visdrone_baseline/weights/

# 复制应用代码
COPY app.py .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动
CMD ["python", "app.py"]
