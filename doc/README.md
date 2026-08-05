# VisDrone 目标检测系统

基于 **YOLOv8n** 的无人机航拍影像目标检测系统，可在消费级显卡 (GTX 1050 Ti 4GB) 上完成从数据准备到模型部署的全流程。

**核心指标**: mAP@50 **30.8%** | 推理 **12.0 ms** (TensorRT) | 显存 ≤ **2.8 GB**

## 功能特性

| 功能 | 说明 | 对应需求 |
|------|------|:---:|
| 🔄 数据格式转换 | VisDrone 原始 8 字段标注 → YOLO 格式，12 类 → 10 类自动映射 | FR-1 |
| 🎨 在线数据增强 | Mosaic / HSV 色彩抖动 / 水平翻转 / LetterBox 缩放 | FR-2 |
| 🧠 模型训练 | YOLOv8n + P2 检测头，AMP 混合精度，显存 ≤ 2.8 GB | FR-3/FR-4 |
| 📊 模型评估 | COCO 标准 mAP，按尺寸分层评估，每类错误分析 | FR-5 |
| 🔍 SAHI 切片推理 | 大图自动切片，独立推理后 NMS 合并，提升小目标召回 | FR-6 |
| 📈 可视化 | PR 曲线 / F1-置信度曲线 / 混淆矩阵，≥300 DPI | FR-7 |
| 📉 训练监控 | 显存峰值告警、吞吐量统计、连续超限自动降 batch_size | FR-8 |
| 🚀 模型部署 | ONNX / TensorRT FP32 导出，加速比 **1.38×** | FR-9 |
| 🌐 服务化部署 | FastAPI RESTful API + Docker 容器化打包 | — |

## 技术架构

| 层 | 技术 | 版本 |
|------|------|------|
| 框架 | PyTorch + Ultralytics YOLOv8 | 1.12.1+cu113 / 8.1.0 |
| GPU | CUDA + cuDNN | 11.3 / 8.3.2 |
| 推理 | ONNX Runtime / TensorRT | 1.13.1 / 8.5.3.1 |
| 增强 | OpenCV / PIL / NumPy | 4.6.0 / 9.5.0 / 1.24.6 |
| 测试 | pytest + flake8 + pylint | 7.2.2 + 9.75/10 |

## 环境要求

| 项目 | 最低要求 |
|------|------|
| GPU | NVIDIA GTX 1050 Ti 4GB (或同等显存) |
| CUDA | 11.3 |
| Python | 3.8.x |
| 操作系统 | Windows 10/11 64位 或 Ubuntu 20.04+ |
| 磁盘空间 | ≥ 20 GB (数据集 + 模型) |

## 快速开始

```bash
# 1. 创建环境
conda create -n visdrone python=3.8 -y
conda activate visdrone

# 2. 安装依赖
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# 3. 训练 (需先放置 VisDrone 数据集到 D:/Data/VisDrone/)
python train.py --name my_experiment --epochs 50
```

## 项目结构

```
VisDrone_Detection/
├── config/                 # 配置模块 (default_config.yaml + 加载器)
├── data/                   # 数据准备 (加载/转换/增强)
├── models/                 # 模型构建 (yolov8n / yolov8n-p2)
├── train/                  # 训练模块 (训练器 + 监控器)
├── eval/                   # 评估模块 (评估器 + 可视化)
├── inference/              # 推理模块 (预测器 + SAHI 适配)
├── export/                 # 部署模块 (ONNX/TensorRT 导出)
├── utils/                  # 工具模块 (坐标/NMS/格式转换/显存)
├── tests/                  # 测试 (62 个用例)
├── doc/                    # 文档 (设计 + 分析 + 规范)
├── train.py                # 训练入口
├── evaluate.py             # 评估入口
├── predict.py              # 推理入口
├── export.py               # 导出入口
├── app.py                  # FastAPI 推理服务
├── batch_predict.py        # 批量推理测试
├── Dockerfile              # Docker 镜像构建
├── requirements_api.txt    # 推理服务最小依赖
└── check_env.py            # 环境检测
```

## 使用方式

```bash
# 训练 (具体参数见 CONFIG_GUIDE.md)
python train.py --name exp01 --epochs 50 --batch_size 4

# 评估
python evaluate.py --model best.pt --data data.yaml --plots

# 推理
python predict.py --model best.pt --source img.jpg           # 直接缩放
python predict.py --model best.pt --source imgs/ --sahi      # SAHI 切片

# 导出
python export.py --model best.pt --format onnx
python export.py --model best.pt --format engine

# 服务化部署 (FastAPI)
python app.py                                                # 直接启动
curl http://localhost:8000/health                            # 健康检查
curl -X POST http://localhost:8000/predict -F "file=@img.jpg" # 推理

# Docker 部署
docker build -t visdrone-detection .
docker run -d -p 8000:8000 --gpus all --name visdrone visdrone-detection
docker save -o visdrone-detection.tar visdrone-detection      # 导出交付
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [INSTALL](INSTALL.md) | 环境搭建、依赖安装、数据集准备 |
| [USER_GUIDE](USER_GUIDE.md) | 训练/评估/推理/部署完整操作指南 |
| [CONFIG_GUIDE](CONFIG_GUIDE.md) | 所有配置项详解与调参建议 |
| [TROUBLESHOOTING](TROUBLESHOOTING.md) | 常见问题与解决方案 |
| [CHANGELOG](CHANGELOG.md) | 版本历史 |

## 许可证

本项目仅用于学习和研究目的。VisDrone 数据集使用请遵循其原始许可协议。
