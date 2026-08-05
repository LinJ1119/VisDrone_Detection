# VisDrone 目标检测系统 —— 故障排除指南

> 每条故障按 "现象 → 原因 → 解决方案 → 预防措施" 标准格式编写。

---

## 目录

- [1. 环境问题](#1-环境问题)
- [2. 数据问题](#2-数据问题)
- [3. 训练问题](#3-训练问题)
- [4. 推理问题](#4-推理问题)
- [5. 部署问题](#5-部署问题)

---

## 1. 环境问题

### 1.1 `torch.cuda.is_available() = False`

**现象**：
```
>>> import torch; print(torch.cuda.is_available())
False
```

**原因**：PyTorch 不是 `cu113` 版，或 NVIDIA 驱动不支持 CUDA 11.3。

**解决方案**：
```bash
# 确认已安装版本
pip show torch | findstr Version
# 如果不是 +cu113，重装
pip uninstall torch torchvision -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
```
如果问题依旧，检查驱动版本：`nvidia-smi`，确认驱动 ≥ 472.12。

**预防措施**：创建环境后立刻验证 CUDA 可用性，不等到训练时才检查。

---

### 1.2 `OSError: [WinError 1455] 页面文件太小`

**现象**：
```
OSError: [WinError 1455] 页面文件太小，无法完成操作。
Error loading "...\cudnn_cnn_infer64_8.dll" or one of its dependencies.
```

**原因**：Windows 页面文件 (虚拟内存) 不足。每个 worker 子进程需独立加载 cuDNN DLL (~500MB/进程)，`num_workers=2` 时需约 1.5GB 额外虚拟地址空间。

**解决方案 (永久)**：
1. 控制面板 → 系统 → 高级系统设置 → 性能设置 → 高级 → 虚拟内存 → 更改
2. 取消 "自动管理"
3. 选 D 盘 → 自定义大小：初始 8192，最大 16384 → 设置 → 确定
4. 重启

**解决方案 (临时)**：
在 `config/default_config.yaml` 中设置 `system.num_workers: 0`

**预防措施**：环境检测脚本应检查页面文件大小并在不足时提示。

---

### 1.3 `ultralytics 版本不兼容`

**现象**：训练时报各种参数错误或 `AttributeError`。

**原因**：ultralytics ≥ 8.2 需要 PyTorch ≥ 2.0，与 CUDA 11.3 不兼容。

**解决方案**：
```bash
pip install ultralytics==8.1.0
```

**预防措施**：`requirements.txt` 中精确锁定 `ultralytics==8.1.0`。

---

### 1.4 Python 版本不存在

**现象**：
```
conda create -n visdrone python=3.8.10 -y
PackagesNotFoundError: python=3.8.10
```

**原因**：conda channels 中没有精确的 micro 版本 3.8.10。

**解决方案**：
```bash
conda create -n visdrone python=3.8 -y   # 自动选 3.8 系最新版
```

**预防措施**：不指定 micro 版本号，3.8.x 之间 API 完全兼容。

---

## 2. 数据问题

### 2.1 配对失败 (skipped_no_annotation > 0)

**现象**：训练日志显示 `skipped_no_annotation: 3`。

**原因**：图像文件在 `images/` 中但 `annotations/` 中没有同名的 `.txt` 标注文件。

**解决方案**：
1. 检查是否有孤立的图像文件 (有图片无标注)
2. 检查文件名是否完全一致 (除扩展名外)：`0000001_00001.jpg` ↔ `0000001_00001.txt`
3. 检查文件扩展名：标注必须是 `.txt`

**预防措施**：运行 `load_dataset()` 验证配对率，目标为 100%。

---

### 2.2 转换日志出现 "字段数异常"

**现象**：
```
WARNING: 标注行 12 字段数异常（期望 8，实际 9），跳过：440,541,271,152,1,6,0,0,
```

**原因**：VisDrone 部分标注文件行末有多余逗号，`split(",")` 后多一空字段。

**解决方案**：已在 `format_converter.py` 中处理 (`line.rstrip(",")`)。如果仍出现，检查原始标注文件是否有更复杂的格式错误。

**预防措施**：数据准备阶段检查 pairing stats，确认 `filtered_invalid_fields` 为 0。

---

### 2.3 图像损坏

**现象**：日志显示 `图像损坏，跳过`。

**原因**：JPEG/PNG 文件不完整或被截断。

**解决方案**：
1. 重新下载或复制损坏的图像文件
2. 用 `PIL.Image.verify()` 批量筛查

---

## 3. 训练问题

### 3.1 CUDA Out of Memory (OOM)

**现象**：
```
RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB
```

**原因**：`batch_size` 或 `imgsz` 过大，超出 GPU 显存。

**本项目已实现自动恢复**：
1. 捕获 OOM → `torch.cuda.empty_cache()`
2. `batch_size` 减半 (`max(1, bs // 2)`)
3. 重建 DataLoader
4. 从 `last.pt` 恢复当前 epoch
5. `batch_size=1` 仍 OOM 3 次 → 终止

**若自动恢复失败**，手动调整：
```yaml
train:
  batch_size: 2          # 或 1
  amp: true              # 确保开启
system:
  gpu_memory_fraction: 0.75
```

**预防措施**：训练前运行 Dry Run 显存预估。

---

### 3.2 Loss 出现 NaN

**现象**：训练中 loss 突然变为 `nan`。

**原因**：
1. 学习率过高
2. AMP 混合精度溢出
3. 标注数据有异常值

**解决方案**：
1. 降低 `lr0` (如 0.01 → 0.001)
2. 临时关闭 AMP (`amp: false`) 跑 1 个 epoch 对比
3. 检查最近的标注文件是否有非法坐标

---

### 3.3 训练吞吐量过低

**现象**：吞吐量 < 10 张/s。

**原因**：`num_workers` 配置不合理。

**解决方案**：
```yaml
system:
  num_workers: 1    # Windows 页面文件受限时
  num_workers: 2    # 页面文件 OK 时
```

---

## 4. 推理问题

### 4.1 检测结果为空 (0 个框)

**现象**：推理后输出的 `boxes` 为空列表。

**原因**：`conf_threshold` 设置过高，或有检测结果但被全部过滤。

**解决方案**：
```bash
python predict.py --model best.pt --source img.jpg --conf 0.15   # 降低阈值
```

---

### 4.2 SAHI 推理报 `ImportError: shapely`

**现象**：
```
ImportError: SAHI 推理需要 shapely 库，请运行：
  conda install -c conda-forge shapely
```

**解决方案**：
```bash
conda install -c conda-forge shapely -y
```

---

### 4.3 大图像推理显存不足

**现象**：推理 4000×3000 图像时报 OOM。

**原因**：SAHI 切片推理时，多个切片同时加载。

**解决方案**：
```bash
python predict.py --model best.pt --source large_img.jpg --sahi --sahi_batch 1
```

或将大图先降采样：程序自动检测最长边 >2500px 时执行降采样。

---

## 5. 部署问题

### 5.1 TensorRT 导出报 `nvinfer.dll not found`

**现象**：
```
FileNotFoundError: Could not find: nvinfer.dll. Is it on your PATH?
```

**原因**：TensorRT wheel 已安装但运行时 DLL 不在 PATH。

**解决方案**：
1. 确认 TensorRT ZIP 已解压
2. 将 `lib/` 目录加入 PATH：
```python
import os
os.environ["PATH"] = r"D:\ProgramData\TensorRT-8.5.3.1\lib;" + os.environ["PATH"]
```

**预防措施**：永久添加到系统环境变量 PATH。

---

### 5.2 TensorRT 报 `np.bool` 不存在

**现象**：
```
AttributeError: module 'numpy' has no attribute 'bool'
```

**原因**：NumPy ≥ 1.24 移除了 `np.bool`，但 TensorRT 8.5 仍使用它。

**解决方案**：在导入 tensorrt 前添加补丁：
```python
import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
```

---

### 5.3 ONNX Runtime CUDA 不可用

**现象**：推理时报错 `CUDA failure 100: no CUDA-capable device is detected`

**原因**：ONNX Runtime 的 CUDA provider 与当前 CUDA 版本不匹配，或 CUDA context 被其他进程占用。

**解决方案**：
- 回退到 CPU provider：`providers=["CPUExecutionProvider"]`
- 或改用 PyTorch 原生推理

---

## 6. 服务化部署问题

### 6.1 Docker Hub 无法连接

**现象**：`docker build` 报错 `dial tcp 108.160.163.106:443: connectex: A connection attempt failed`

**原因**：Docker Hub 在国内被 GFW 封锁。

**解决方案**：
```powershell
# 通过代理站拉取基础镜像
docker pull docker.1ms.run/nvidia/cuda:11.3.1-runtime-ubuntu20.04
docker tag docker.1ms.run/nvidia/cuda:11.3.1-runtime-ubuntu20.04 nvidia/cuda:11.3.1-runtime-ubuntu20.04
docker build -t visdrone-detection .
```

### 6.2 PyTorch 在 Docker 构建时下载超慢

**现象**：`torch-*.whl` 下载 900MB 停滞数小时。

**原因**：`download.pytorch.org` 在海外，Docker 下载不支持断点续传。

**解决方案**：浏览器下载 Linux 版 wheel 到 `vendor/` 目录，Dockerfile 改为本地安装（详见操作手册）。

### 6.3 容器状态 `(unhealthy)`

**现象**：`docker ps` 显示 `STATUS: Up ... (unhealthy)`

**原因**：Dockerfile 健康检查用 `curl`，但基础镜像不含 curl。

**解决方案**：健康检查改用 Python 内置 `urllib.request`（详见 Dockerfile 第 49-50 行）。

### 6.4 容器命名冲突

**现象**：`docker run --name visdrone` 报错 `Conflict. The container name "/visdrone" is already in use`

**解决方案**：
```powershell
docker start visdrone   # 启动已有容器，不要重新创建
```

### 6.5 客户端调用 400 错误

**现象**：`{"detail":"Only image files accepted, got: None"}`

**原因**：Python `requests.post(files=...)` 不自动设 Content-Type。

**解决方案**：服务端已兼容 `content_type=None`，不需客户端做任何修改。

---

**相关文档**：
- 环境搭建 → [安装部署指南](INSTALL.md)
- 参数详解 → [配置指南](CONFIG_GUIDE.md)
- 操作指南 → [用户使用手册](USER_GUIDE.md)
