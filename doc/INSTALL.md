# VisDrone 目标检测系统 —— 安装部署指南

> **目标**：从零开始，30 分钟内完成环境搭建和数据集准备。
> **依据**：GB/T 8567 第 9.4 节 "安装手册"

---

## 目录

- [1. 系统要求](#1-系统要求)
- [2. Conda 环境创建](#2-conda-环境创建)
- [3. PyTorch 安装](#3-pytorch-安装)
- [4. 其余依赖安装](#4-其余依赖安装)
- [5. 可选依赖安装](#5-可选依赖安装)
- [6. 安装验证](#6-安装验证)
- [7. 数据集准备](#7-数据集准备)
- [8. 配置文件修改](#8-配置文件修改)
- [9. 常见安装问题](#9-常见安装问题)

---

## 1. 系统要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 | 本项目实测 |
|------|---------|---------|:---:|
| GPU | NVIDIA GPU, 4GB VRAM | 6GB+ VRAM | GTX 1050 Ti 4GB |
| 内存 | 8 GB | 16 GB+ | 16 GB |
| 磁盘 | 20 GB 空闲 | 50 GB+ SSD | — |

**GPU 兼容性要求**：
- 计算能力 (Compute Capability) ≥ 6.1 (Pascal+)
- 本项目 GTX 1050 Ti 为 SM 6.1，支持 FP32 训练/推理
- **不支持** 硬件 FP16 加速 (需 Volta SM 7.0+)

### 软件要求

| 软件 | 版本 | 锁定原因 |
|------|------|------|
| Windows | 10/11 Pro 64位 | spawn 多进程模式 |
| 或 Ubuntu | 20.04+ 64位 | fork 多进程模式 |
| CUDA | **11.3** | PyTorch 1.12.1+cu113 依赖 |
| Python | **3.8.x** (3.8.0~3.8.20) | 兼容性最佳 |
| NVIDIA 驱动 | ≥ 472.12 (Windows) / ≥ 470 (Linux) | CUDA 11.3 最低要求 |

---

## 2. Conda 环境创建

```bash
# 创建 Python 3.8 虚拟环境
conda create -n visdrone python=3.8 -y

# 激活环境
conda activate visdrone
```

> **幂等性**：`conda create -n visdrone` 在环境已存在时会报错，如需重建先执行 `conda remove -n visdrone --all`。

> **说明**：不需要指定精确的 micro 版本 (如 3.8.10)，`python=3.8` 会自动选择 3.8 系最新的可用版本。深度学习依赖库只关心 3.8 大版本 API 兼容性。

---

## 3. PyTorch 安装

**必须使用以下命令**，不能走 PyPI 默认源：

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

**验证 PyTorch 能识别 GPU**（预期输出最后一行）：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

预期输出：
```
1.12.1+cu113
True
NVIDIA GeForce GTX 1050 Ti
```

**判断标准**：
- ✅ `True` + 正确 GPU 名称 → 成功
- ❌ `False` → 检查 CUDA 版本、驱动版本、PyTorch 是否为 cu113 版
- ❌ 报错 → 检查 CUDA 11.3 是否已安装，驱动是否 ≥ 472.12

---

## 4. 其余依赖安装

```bash
cd D:\myproject\VisDrone_Detection
pip install -r requirements.txt
```

**依赖清单** (requirements.txt)：

| 包 | 版本 | 用途 |
|------|------|------|
| ultralytics | **8.1.0** | YOLOv8 训练/评估/导出引擎 |
| numpy | 1.24.6 | 数值计算 |
| opencv-python-headless | 4.6.0 | 图像读写/绘制 |
| Pillow | 9.5.0 | 图像格式校验 |
| matplotlib | 3.5.3 | PR/F1/混淆矩阵绘图 |
| PyYAML | 6.0 | YAML 配置解析 |
| pytest | 7.2.2 | 自动化测试 |
| pytest-cov | 4.0.0 | 测试覆盖率 |
| tensorboard | 2.10.1 | 训练指标可视化 |
| sahi | 0.11.15 | SAHI 切片推理 |

> **注意**：`tensorrt` 不在此文件中（需从 NVIDIA ZIP 包手动安装，详见 5.3 节）。

---

## 5. 可选依赖安装

### 5.1 SAHI 依赖 (切片推理)

```bash
conda install -c conda-forge shapely -y
```

验证：
```bash
python -c "import shapely; print(shapely.__version__)"
```

> 如果 conda 失败可尝试 `pip install shapely`。两者都失败不影响 Must 核心流程 (训练/评估/直接推理)，仅 SAHI 推理不可用。

### 5.2 代码审查工具

```bash
pip install flake8 pylint
```

### 5.3 TensorRT (模型加速部署)

由于 CUDA 11.3 的限制，TensorRT 需从 NVIDIA 官网下载 ZIP 包手动安装：

1. 前往 [NVIDIA TensorRT Archive](https://developer.nvidia.com/tensorrt/download) 下载：
   `TensorRT-8.5.3.1.Windows10.x86_64.cuda-11.8.cudnn8.6.zip`

2. 解压到 `D:\ProgramData\TensorRT-8.5.3.1`

3. 安装 Python wheel：
```bash
pip install "D:\ProgramData\TensorRT-8.5.3.1\python\tensorrt-8.5.3.1-cp38-none-win_amd64.whl"
```

4. 将 TensorRT lib/ 加入 PATH（或在导出脚本中设置）：
```python
import os
os.environ["PATH"] = r"D:\ProgramData\TensorRT-8.5.3.1\lib;" + os.environ["PATH"]
```

验证：
```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```
预期输出：`8.5.3.1`

---

## 6. 安装验证

运行环境检测脚本：

```bash
python check_env.py
```

**预期输出（全部通过）**：
```
=== 环境检测报告 ===
[PASS] Python 版本: 3.8.20
[PASS] PyTorch 版本: 1.12.1+cu113
[PASS] CUDA 可用: True (GeForce GTX 1050 Ti, 4096 MiB)
[PASS] 磁盘空间: 50.2 GB 可用
[PASS] 依赖包: 15/15 已安装
[WARN] shapely: 未安装 (SAHI 推理不可用，不影响训练)
[PASS] YOLOv8n 前向推理: 显存增量 0.12 GB
```

**判断标准**：
- 所有项 `[PASS]` → 环境就绪 ✅
- 仅 SAHI/shapely 项 `[WARN]` → 不影响核心流程 ✅
- 任何 `[FAIL]` → 修复后重试

---

## 7. 数据集准备

### 7.1 数据集下载

从 VisDrone 官方下载 VisDrone2019-DET 数据集：http://aiskyeye.com/

### 7.2 目录结构

将数据按以下结构放置：

```
D:\Data\VisDrone\
├─ train\
│  ├─ images\          (6471 .jpg)
│  └─ annotations\     (6471 .txt)
├─ val\
│  ├─ images\          (548 .jpg)
│  └─ annotations\     (548 .txt)
└─ test-dev\
   ├─ images\          (1610 .jpg)
   └─ annotations\     (1610 .txt)
```

> **关键检查**：每个 `.jpg` 文件在 `annotations/` 目录下必须有**同名** `.txt` 文件。用 `load_dataset()` 配对验证，目标配对率 100%。

### 7.3 数据转换

首次运行时，`train.py` 自动执行数据转换：
- 输入：VisDrone 原始 8 字段逗号分隔标注
- 输出：`datasets/visdrone/` 下的 YOLO 格式数据集 + `data.yaml`

---

## 8. 配置文件修改

编辑 `config/default_config.yaml` 中的路径：

```yaml
data:
  train_root: "D:/Data/VisDrone/train"
  val_root: "D:/Data/VisDrone/val"
  test_root: "D:/Data/VisDrone/test-dev"
```

其他配置项说明见 [配置指南](CONFIG_GUIDE.md)。

---

## 9. 常见安装问题

### Q1: `torch.cuda.is_available() = False`

**原因**：PyTorch 版本不是 `cu113` 版，或 CUDA 驱动不可用。

**解决**：
```bash
# 检查已安装的 PyTorch 版本
pip show torch | grep Version

# 如果不是 +cu113 版，卸载后重装
pip uninstall torch torchvision -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### Q2: `ImportError: No module named 'ultralytics'`

**原因**：依赖未安装。

**解决**：`pip install ultralytics==8.1.0`

### Q3: `OSError: [WinError 1455] 页面文件太小`

**原因**：Windows 虚拟内存 (页面文件) 不足，无法同时加载多份 cuDNN DLL。

**解决**：
1. 控制面板 → 系统 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存 → 更改
2. 取消 "自动管理"
3. 选择 D 盘 → 自定义大小：初始 8192 MB，最大 16384 MB → 设置 → 确定
4. 重启计算机

**临时缓解**：在 `default_config.yaml` 中设置 `system.num_workers: 0`

### Q4: NVIDIA 驱动版本过旧

**原因**：驱动 < 472.12 不支持 CUDA 11.3。

**解决**：从 [NVIDIA 官网](https://www.nvidia.com/download/index.aspx) 下载 GTX 1050 Ti 最新驱动，选择 "自定义安装" → 勾选 "执行清洁安装"。

### Q5: 磁盘空间不足

**现象**：数据转换时报 `OSError: No space left on device`

**解决**：
- 数据集 (源) 约 2.5 GB，转换后 (副本) 约 2.5 GB
- 确保项目所在磁盘有 ≥ 10 GB 空闲空间

---

**安装完成后，继续阅读 [用户使用手册](USER_GUIDE.md) 了解如何训练和评估模型。**
