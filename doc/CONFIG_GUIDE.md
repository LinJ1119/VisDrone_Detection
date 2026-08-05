# VisDrone 目标检测系统 —— 配置指南

> 所有配置项集中管理于 `config/default_config.yaml`，通过 `config/config_loader.py` 加载和校验。

---

## 目录

- [1. 配置文件结构总览](#1-配置文件结构总览)
- [2. data — 数据配置](#2-data--数据配置)
- [3. model — 模型配置](#3-model--模型配置)
- [4. aug — 增强配置](#4-aug--增强配置)
- [5. train — 训练配置](#5-train--训练配置)
- [6. system — 系统配置](#6-system--系统配置)
- [7. inference — 推理配置](#7-inference--推理配置)
- [8. export — 部署配置](#8-export--部署配置)
- [9. 参数速查表](#9-参数速查表)
- [10. 典型配置示例](#10-典型配置示例)
- [11. 调参建议](#11-调参建议)

---

## 1. 配置文件结构总览

```
default_config.yaml
├── data/          # 数据集路径和类别
├── model/         # 模型选择和预训练权重
├── aug/           # 数据增强参数
├── train/         # 训练超参数
├── system/        # 系统参数 (种子、workers、显存)
├── inference/     # 推理阈值
└── export/        # 模型导出参数
```

**配置合并规则**：
1. `default_config.yaml` (内置默认) → 2. 用户自定义 `config.yaml` 覆盖 → 3. CLI 参数覆盖

---

## 2. data — 数据配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `train_root` | str | `D:/Data/VisDrone/train` | 训练集原始目录 (含 images/ 和 annotations/) |
| `val_root` | str | `D:/Data/VisDrone/val` | 验证集原始目录 |
| `test_root` | str | `D:/Data/VisDrone/test-dev` | 测试集原始目录 |
| `image_dir` | str | `images` | 图像子目录名 |
| `annotation_dir` | str | `annotations` | 标注子目录名 |
| `output_base` | str | `./datasets/visdrone` | 转换后 YOLO 格式数据集输出路径 |
| `nc` | int | 10 | 类别数 (VisDrone: 10) |
| `names` | list | [pedestrian, ..., motor] | 类别名列表 |
| `val_split_ratio` | float | 0.0 | 从训练集随机切分验证集比例 (0=使用官方划分) |

**路径格式**：所有路径使用正斜杠 `/`，确保 Windows/Linux 兼容。

---

## 3. model — 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `name` | str | `yolov8n` | 模型名称，合法值 `yolov8n` / `yolov8n-p2` |
| `pretrained` | str | `yolov8n.pt` | 预训练权重路径 (None=随机初始化) |
| `imgsz` | int | 640 | 输入图像尺寸 (正方形) |
| `nc` | int | 10 | 类别数 |
| `p2_head` | bool | false | 是否启用 P2 检测头 (小目标优化) |

**P2 检测头说明**：

| 指标 | yolov8n | yolov8n-p2 | 变化 |
|------|:---:|:---:|:---:|
| 参数量 | 3.01M | 2.93M | -2.7% |
| FLOPs | 8.2G | 12.4G | +51% |
| 检测层 | 3 层 | **4 层** | 增加 160² 检测层 |
| 显存占用 | 约 +0.5 GB | — | batch_size 需从 4 → 2 |
| 小目标 mAP | — | 预期 +1-3 pp | — |

---

## 4. aug — 增强配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `mosaic` | bool | true | Mosaic 增强 (4 张图拼接为 1 张) |
| `flip_prob` | float | 0.5 | 水平翻转概率 |
| `hsv_h` | float | 0.015 | HSV 色调抖动幅度 |
| `hsv_s` | float | 0.7 | HSV 饱和度抖动幅度 |
| `hsv_v` | float | 0.4 | HSV 明度抖动幅度 |
| `close_mosaic` | int | 10 | 最后 N 个 epoch 关闭 Mosaic |
| `multiscale` | bool | false | 多尺度训练开关 |

**增强时间线**：
```
Epoch 1-40:  Mosaic ON  (小目标增强, 泛化)
Epoch 41-50: Mosaic OFF (适应真实图像分布, 微调)
全程:        HSV 抖动 + 水平翻转 + LetterBox
```

---

## 5. train — 训练配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `batch_size` | int | 4 | 每批图像数 (GTX 1050 Ti 4GB 上限) |
| `accumulation_steps` | int | 2 | 梯度累积步数 (有效 batch = bs × acum) |
| `epochs` | int | 50 | 训练总轮数 |
| `amp` | bool | true | 混合精度训练 (省显存 30-40%) |
| `lr0` | float | 0.01 | 初始学习率 |
| `lrf` | float | 0.01 | 最终学习率因子 (lr_final = lr0 × lrf) |
| `momentum` | float | 0.937 | 动量 |
| `weight_decay` | float | 0.0005 | 权重衰减 (L2 正则化) |
| `optimizer` | str | `AdamW` | 优化器类型 |
| `dropout` | float | 0.0 | Dropout 概率 (0=关闭) |
| `early_stop_patience` | int | 15 | 早停耐心值 (0=不启用) |
| `multiscale` | bool | false | 多尺度训练开关 |
| `output_dir` | str | `./runs/train` | 训练输出根目录 |
| `name` | str | `""` | 实验名称 (空=自动时间戳) |

---

## 6. system — 系统配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `gpu_memory_fraction` | float | 0.75 | 进程可用显存比例上限 (0~1) |
| `num_workers` | int | 1 | DataLoader 子进程数 |
| `pin_memory` | bool | false | 是否使用锁页内存 |
| `seed` | int | 42 | 全局随机种子 (确保实验可复现) |

**num_workers 选择指南**：

| 值 | 吞吐量 (实测) | 适用场景 |
|:---:|:---:|------|
| 0 | ~9 张/s | 调试、单步运行、内存紧张 |
| **1** | ~14 张/s | Windows 页面文件受限时的平衡选择 |
| 2 | ~17 张/s | 页面文件 ≥8GB 时的最优选择 |

---

## 7. inference — 推理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `conf_threshold` | float | 0.25 | 全局置信度阈值 |
| `iou_threshold` | float | 0.45 | NMS IoU 阈值 |
| `sahi_enabled` | bool | false | SAHI 切片推理开关 |
| `sahi_slice_size` | int | 640 | SAHI 切片尺寸 |
| `sahi_overlap` | float | 0.2 | SAHI 切片重叠率 |
| `sahi_batch_size` | int | 4 | SAHI 切片批处理大小 |

---

## 8. export — 部署配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `format` | str | `onnx` | 导出格式 `onnx` / `engine` |
| `quantize` | str | `fp32` | 量化精度 (Pascal 仅支持 fp32) |
| `dynamic_batch` | bool | true | 是否支持动态批大小 |
| `calib_dataset` | str | `""` | 量化校准数据集路径 (fp32 不需要) |
| `opset` | int | 12 | ONNX opset 版本号 |

---

## 9. 参数速查表

| 参数路径 | 类型 | 默认值 | 范围/选项 |
|------|------|------|------|
| `data.train_root` | str | `D:/Data/VisDrone/train` | 任意有效路径 |
| `data.nc` | int | 10 | 1~80 |
| `model.name` | str | `yolov8n` | `yolov8n` / `yolov8n-p2` |
| `model.imgsz` | int | 640 | 320, 416, 512, **640**, 960 |
| `aug.mosaic` | bool | true | true / false |
| `aug.flip_prob` | float | 0.5 | 0~1 |
| `aug.hsv_h` | float | 0.015 | 0~0.1 |
| `aug.close_mosaic` | int | 10 | 0~epochs |
| `train.batch_size` | int | 4 | 1~16 (受显存限制) |
| `train.epochs` | int | 50 | 1~500 |
| `train.amp` | bool | true | true / false |
| `train.lr0` | float | 0.01 | 1e-5~1e-1 |
| `train.lrf` | float | 0.01 | 1e-4~1.0 |
| `train.optimizer` | str | `AdamW` | `SGD` / `Adam` / `AdamW` |
| `train.dropout` | float | 0.0 | 0~0.5 |
| `train.early_stop_patience` | int | 15 | 0~200 |
| `system.num_workers` | int | 1 | 0~8 |
| `system.seed` | int | 42 | 任意整数 |
| `system.gpu_memory_fraction` | float | 0.75 | 0.1~1.0 |
| `inference.conf_threshold` | float | 0.25 | 0.01~1.0 |
| `inference.iou_threshold` | float | 0.45 | 0.01~1.0 |
| `export.opset` | int | 12 | 9~17 |

---

## 10. 典型配置示例

### 快速测试 (2 epoch, 验证流程可跑通)

```yaml
train:
  batch_size: 2
  epochs: 2
system:
  num_workers: 0
```

### 正式训练 (本项目最优配置)

```yaml
train:
  batch_size: 4
  accumulation_steps: 2
  epochs: 50
  amp: true
system:
  num_workers: 1
  seed: 42
```

### 小目标优化 (P2 头 + 多尺度)

```yaml
model:
  name: "yolov8n-p2"
train:
  batch_size: 2        # 必须降低，P2头+0.5GB显存
  epochs: 100
  multiscale: true
aug:
  multiscale: true
```

---

## 11. 调参建议

| 场景 | 调整参数 | 调整方向 |
|------|------|------|
| **显存不足 (OOM)** | `batch_size` | 降低至 2 或 1 |
| | `num_workers` | 降低至 0 |
| | `amp` | 确保为 true |
| | `model.name` | 使用 `yolov8n` (非 P2) |
| **训练太慢** | `num_workers` | 增加至 2 |
| | `batch_size` | 增加至 4 |
| **过拟合** (train↓ val↑) | `dropout` | 增加至 0.1~0.2 |
| | `weight_decay` | 增加至 0.001 |
| | `aug.multiscale` | true |
| **欠拟合** (loss 仍在下滑) | `epochs` | 增加至 100~150 |
| **小目标检测差** | `model.name` | `yolov8n-p2` |
| | `aug.multiscale` | true |
| | `aug.close_mosaic` | 增加至 15 |
| **推理误检太多** | `inference.conf_threshold` | 提高至 0.35~0.5 |
| **推理漏检太多** | `inference.conf_threshold` | 降低至 0.15~0.2 |
| | 推理模式 | 启用 SAHI |

---

**相关文档**：
- 环境搭建 → [安装部署指南](INSTALL.md)
- 操作指南 → [用户使用手册](USER_GUIDE.md)
- 异常排查 → [故障排除指南](TROUBLESHOOTING.md)
