# VisDrone 目标检测系统 —— 用户使用手册

> **依据**：GB/T 8567 第 9.3 节 "用户手册" + 第 9.5 节 "操作手册"
> **前置**：已完成安装部署，环境检测通过

---

## 目录

- [1. 程序整体工作流程](#1-程序整体工作流程)
- [2. 训练模式](#2-训练模式)
  - [2.1 基本训练](#21-基本训练)
  - [2.2 断点续训](#22-断点续训)
  - [2.3 CLI 参数覆盖配置](#23-cli-参数覆盖配置)
  - [2.4 训练监控](#24-训练监控)
  - [2.5 训练输出](#25-训练输出)
- [3. 评估模式](#3-评估模式)
  - [3.1 基本评估](#31-基本评估)
  - [3.2 评估 + 可视化](#32-评估--可视化)
  - [3.3 评估指标解读](#33-评估指标解读)
- [4. 推理模式](#4-推理模式)
  - [4.1 直接缩放推理](#41-直接缩放推理)
  - [4.2 SAHI 切片推理](#42-sahi-切片推理)
  - [4.3 输出格式](#43-输出格式)
- [5. 模型导出](#5-模型导出)
- [6. 环境检测](#6-环境检测)
- [7. 服务化部署](#7-服务化部署)
  - [7.1 FastAPI 直接启动](#71-fastapi-直接启动)
  - [7.2 Docker 容器部署](#72-docker-容器部署)
  - [7.3 两种部署方式对比](#73-两种部署方式对比)

---

## 1. 程序整体工作流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 环境检测  │ ─→ │ 数据准备  │ ─→ │ 模型训练  │ ─→ │ 模型评估  │
│check_env │    │(自动执行) │    │ train.py │    │evaluate.py│
└──────────┘    └──────────┘    └────┬─────┘    └──────────┘
                                     │
                    ┌────────────────┼────────────────────────┐
                    ▼                ▼                ▼       ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐  ┌──────────┐
              │ 推理预测  │   │ 可视化    │   │ 模型导出  │  │ 服务部署  │
              │predict.py│   │evaluate   │   │ export.py │  │  app.py   │
              └──────────┘   │  --plots  │   └──────────┘  │ Docker    │
                             └──────────┘                  └──────────┘
```

所有入口脚本均支持 `--help` 查看完整参数列表。

---

## 2. 训练模式

### 2.1 基本训练

```bash
python train.py --name visdrone_baseline
```

**执行流程**（自动）：
1. 显存预估 (子进程 Dry Run) → 确认 ≤3.1 GB
2. 数据准备 (VisDrone 原始 → YOLO 格式 → `datasets/visdrone/`)
3. 模型构建 (YOLOv8n + COCO 预训练权重)
4. 训练循环 (50 epoch, AMP, Monitor 实时监控)

**预期输出**（训练完成时）：
```
50 epochs completed in 5.306 hours.
训练完成！最优模型: ./runs/train/visdrone_baseline/weights/best.pt
```

### 2.2 断点续训

```bash
python train.py --resume runs/train/visdrone_baseline/weights/last.pt
```

从 `last.pt` 恢复训练（含 optimizer state + epoch 编号），训练从断点继续。

### 2.3 CLI 参数覆盖配置

以下参数可通过命令行覆盖 `default_config.yaml` 中的设置：

| 参数 | 类型 | 示例 |
|------|------|------|
| `--batch_size` | int | `--batch_size 2` |
| `--epochs` | int | `--epochs 100` |
| `--lr0` | float | `--lr0 0.001` |
| `--amp` | bool | `--amp false` |
| `--data_root` | str | `--data_root D:/Data/VisDrone` |
| `--config` | str | `--config my_config.yaml` |

```bash
# 使用自定义配置 + 覆盖参数
python train.py --config config/custom.yaml --name exp02 --epochs 100 --batch_size 2
```

### 2.4 训练监控

训练期间，每 batch 和每 epoch 自动输出监控信息：

**Progress Bar 解读**：
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50     0.908G      2.043       2.09      1.135        174        640
  │          │          │          │          │            │          │
  │          │          │          │          │            │          └─ 输入尺寸
  │          │          │          │          │            └─ 每 batch 平均目标数
  │          │          │          │          └─ 分布焦点损失
  │          │          │          └─ 分类损失
  │          │          └─ 边界框回归损失
  │          └─ GPU 显存使用 (应 ≤ 3.1 GB)
  └─ 当前 epoch / 总 epoch
```

**TensorBoard 监控**（另开终端）：
```bash
tensorboard --logdir runs/train/
# 浏览器打开 http://localhost:6006
```

**Monitor 日志示例**：
```
[INFO] Epoch 吞吐量: 18.1 张/s
[WARNING] 显存超限！global_step 3237: 3.21 GB > 3.0 GB (连续 1 次)
```

### 2.5 训练输出

```
runs/train/visdrone_baseline/
├── weights/
│   ├── best.pt          ← 验证集 mAP@50 最高的模型 (6.2 MB)
│   └── last.pt          ← 最后一个 epoch 的模型 (6.2 MB)
├── config_current.yaml  ← 本次训练使用的完整配置快照
├── labels.jpg           ← 标注框分布统计图
├── results.csv          ← 每个 epoch 的详细指标
└── events.out.tfevents  ← TensorBoard 事件文件
```

---

## 3. 评估模式

### 3.1 基本评估

```bash
python evaluate.py --model runs/train/visdrone_baseline/weights/best.pt --data datasets/visdrone/data.yaml
```

**输出示例**：
```
============================================
  评估完成
============================================
  mAP@50:   0.3084
  mAP@50-95:0.1798
  Precision:0.4148
  Recall:   0.3094

  每类 mAP@50:
    car                  0.7386
    bus                  0.4305
    van                  0.3586
    motor                0.3335
    pedestrian           0.3184
    truck                0.2714
    people               0.2570
    tricycle             0.1959
    awning-tricycle      0.1182
    bicycle              0.0617

  高漏检率类别（FN>40%）: pedestrian, people, bicycle, van, ...
```

### 3.2 评估 + 可视化

```bash
python evaluate.py --model best.pt --data data.yaml --plots
```

生成 3 张曲线图 (≥300 DPI)：
- `pr_curve.png` — 全类 + 每类 PR 曲线
- `f1_curve.png` — F1-置信度曲线
- `confusion_matrix.png` — 混淆矩阵

### 3.3 评估指标解读

| 指标 | 含义 | 本项目 | 标准 |
|------|------|:---:|:---:|
| **mAP@50** | IoU≥0.5 时的平均精度 | 0.308 | 达标 |
| **mAP@50-95** | IoU 0.5→0.95 每 0.05 步长取均值 | 0.180 | 严格 |
| **Precision** | 预测为目标的框中真正目标的比例 | 0.415 | — |
| **Recall** | 真实目标中被检测出的比例 | 0.309 | — |

> 注意：mAP@50 是工业和竞赛常用指标，mAP@50-95 是 COCO 竞赛标准。

---

## 4. 推理模式

### 4.1 直接缩放推理

```bash
# 单张
python predict.py --model best.pt --source D:/Data/VisDrone/test-dev/images/0000001_00001.jpg --conf 0.25

# 批量
python predict.py --model best.pt --source D:/Data/VisDrone/test-dev/images/ --save_img --save_json
```

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--conf` | 0.25 | 置信度阈值，低于此值的框不输出 |
| `--iou` | 0.45 | NMS IoU 阈值 |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--save_img` | — | 保存带框的可视化图像到 `runs/predict/*/vis/` |
| `--save_txt` | — | 保存 YOLO TXT 格式结果 |
| `--save_json` | — | 保存 JSON 格式结果 |
| `--output` | — | 指定输出目录 |

**推理耗时**（640×640 输入, GTX 1050 Ti）：
- PyTorch 直接推理：约 16.5 ms
- ONNX CUDA：约 14.2 ms
- TensorRT：约 12.0 ms

### 4.2 SAHI 切片推理

适用于**大尺寸图像**中检测小目标：

```bash
python predict.py --model best.pt --source D:/Data/VisDrone/test-dev/images/ --sahi --save_img
```

| SAHI 参数 | 默认值 | 说明 |
|------|------|------|
| `--sahi` | — | 启用 SAHI 切片推理 |
| `--sahi_slice_size` | 640 | 切片尺寸 (px) |
| `--sahi_overlap` | 0.2 | 切片重叠率 (0~1) |
| `--sahi_batch` | 4 | 切片批处理大小 |

**原理**：将大图切分为多个 640×640 窗口 → 每窗口独立推理 → NMS 合并 → 全图结果。

**对比**：SAHI 模式小目标召回率高于直接缩放，但推理时间更长 (单张 3-5 秒)。

### 4.3 输出格式

**JSON 格式** (`--save_json` 时生成)：
```json
[{
  "image_name": "0000001_00001.jpg",
  "boxes": [
    {"class_id": 3, "class_name": "car", "conf": 0.87, "xyxy": [340, 560, 520, 720]},
    {"class_id": 0, "class_name": "pedestrian", "conf": 0.52, "xyxy": [120, 80, 145, 115]}
  ],
  "inference_time_ms": 16.5,
  "mode": "direct"
}]
```

---

## 5. 模型导出

```bash
# ONNX 导出 (跨平台通用)
python export.py --model best.pt --format onnx

# TensorRT 导出 (NVIDIA GPU 专用, 最快)
python export.py --model best.pt --format engine

# 导出 + 推理速度对比
python export.py --model best.pt --format engine --benchmark
```

**导出产物**：

| 格式 | 文件 | 大小 | 推理耗时 | 加速比 |
|------|------|------|------|:---:|
| PyTorch | `best.pt` | 6.2 MB | 16.5 ms | 1.00× |
| ONNX | `best.onnx` | 12.2 MB | 14.2 ms | 1.17× |
| TensorRT | `best.engine` | 18.0 MB | 12.0 ms | **1.38×** |

---

## 6. 环境检测

```bash
python check_env.py
```

检测项：Python 版本、PyTorch + CUDA、GPU 显存、磁盘空间、依赖包版本、shapely 可用性、YOLOv8n 前向推理。

---

## 7. 服务化部署

### 7.1 FastAPI 直接启动

```bash
# 启动服务
python app.py
# 看到 [INIT] pytorch model loaded 即成功
```

**API 端点**

| 方法 | 路径 | 功能 | 示例 |
|------|------|------|------|
| `GET` | `/` | 服务信息 | `curl http://localhost:8000/` |
| `GET` | `/health` | 健康检查 | `curl http://localhost:8000/health` |
| `POST` | `/predict` | 上传图像推理 | `curl -X POST http://localhost:8000/predict -F "file=@img.jpg"` |

**推理响应格式**
```json
{
  "image_name": "test.jpg",
  "model_type": "tensorrt",
  "inference_time_ms": 27.3,
  "num_detections": 40,
  "detections": [
    {"class_id": 3, "class_name": "car", "confidence": 0.84, "xyxy": [340, 560, 520, 720]}
  ]
}
```

### 7.2 Docker 容器部署

```bash
# 构建镜像
docker build -t visdrone-detection .

# 启动容器
docker run -d -p 8000:8000 --gpus all --name visdrone visdrone-detection

# 查看日志
docker logs visdrone

# 批量测试
python batch_predict.py

# 导出交付
docker save -o visdrone-detection.tar visdrone-detection
```

**常用管理命令**

| 操作 | 命令 |
|------|------|
| 启动容器 | `docker start visdrone` |
| 停止容器 | `docker stop visdrone` |
| 查看状态 | `docker ps` |
| 进入调试 | `docker exec -it visdrone bash` |

### 7.3 两种部署方式对比

| | FastAPI 直接 | Docker |
|------|:---:|:---:|
| 启动命令 | `python app.py` | `docker start visdrone` |
| 环境要求 | conda + 依赖 | 仅 Docker Desktop |
| 跨机器迁移 | 重装依赖 | 复制 tar 文件 |
| 推理延迟 | ~27ms | ~25ms |
| 适合场景 | 开发调试 | **客户交付** |

---

**相关文档**：
- 所有参数详解 → [配置指南](CONFIG_GUIDE.md)
- 训练/推理异常 → [故障排除指南](TROUBLESHOOTING.md)
