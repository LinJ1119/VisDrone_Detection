"""
GPU 显存预估与实时查询。

接口定义参见概要设计 I-05。
Dry Run 在独立子进程中执行（Windows spawn 模式），规避 CUDA tensor 不可 pickle 的问题。
"""

import logging
import multiprocessing as mp
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# 子进程入口（独立函数，模块顶层，Windows spawn 模式要求）
# ══════════════════════════════════════════════════════════════════


def _dry_run_worker(send_queue, model_name, pretrained_path, nc,
                    input_size, batch_size, amp, project_root):
    """Dry Run 子进程入口：重建模型 → 随机 batch → 前向+反向 → 返回峰值显存。

    在主进程之外独立执行，结束后 GPU 显存自动释放。
    所有参数为纯 Python 类型（str/int/bool/tuple），可安全 pickle。
    """
    try:
        # 确保子进程能找到项目模块及所有依赖
        sys.path.insert(0, project_root)

        import numpy as np
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import IterableSimpleNamespace

        # 1) 预热 CUDA context
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.zeros(1, device="cuda")  # warmup

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2) 子进程独立重建模型
        yaml_file = {"yolov8n": "yolov8n.yaml", "yolov8n-p2": "yolov8n-p2.yaml"}[model_name]
        yaml_path = Path(__file__).resolve().parent.parent / "models" / "model_configs" / yaml_file
        if not yaml_path.is_file():
            send_queue.put((None, [f"模型 yaml 不存在: {yaml_path}"]))
            return

        model = YOLO(str(yaml_path))
        model.load(pretrained_path)
        model.model = model.model.to(device)
        model.model.train()

        # 3) 确保 model.args 支持属性访问（v8DetectionLoss 会访问 .box/.cls/.dfl）
        #    正常训练时 model.train() 内部会处理，Dry Run 需手动转换
        if isinstance(model.model.args, dict):
            model.model.args = IterableSimpleNamespace(**model.model.args)

        # 4) 构建随机 batch（模拟 VisDrone 标注密度：~50 框/图）
        img_h, img_w = input_size
        img = torch.randn(batch_size, 3, img_h, img_w, device=device)

        total_boxes = 0
        cls_list, bboxes_list, batch_idx_list = [], [], []
        for b in range(batch_size):
            n_boxes = int(np.random.randint(40, 61))  # 40~60 框/图
            total_boxes += n_boxes
            if n_boxes == 0:
                continue
            # 随机归一化中心点 + 宽高（模拟小目标为主的分布）
            cxy = np.random.uniform(0.05, 0.95, (n_boxes, 2)).astype(np.float32)
            wh = np.random.uniform(0.005, 0.3, (n_boxes, 2)).astype(np.float32)
            cls_list.append(np.random.randint(0, nc, n_boxes, dtype=np.int64))
            bboxes_list.append(np.concatenate([cxy, wh], axis=1))
            batch_idx_list.append(np.full(n_boxes, b, dtype=np.int64))

        if total_boxes == 0:
            # 至少一个框，避免空图
            cxy = np.array([[0.5, 0.5]], dtype=np.float32)
            wh = np.array([[0.1, 0.1]], dtype=np.float32)
            cls_list = [np.array([0], dtype=np.int64)]
            bboxes_list = [np.concatenate([cxy, wh], axis=1)]
            batch_idx_list = [np.array([0], dtype=np.int64)]
            total_boxes = 1

        batch = {
            "img": img,
            "cls": torch.from_numpy(np.concatenate(cls_list)).to(device),
            "bboxes": torch.from_numpy(np.concatenate(bboxes_list)).to(device),
            "batch_idx": torch.from_numpy(np.concatenate(batch_idx_list)).to(device),
        }

        # 5) 前向 + 反向（模拟一次完整训练 step）
        # model.model.loss(batch) 内部自动执行：
        #   preds = self.forward(batch["img"])
        #   loss = self.criterion(preds, batch)
        with torch.cuda.amp.autocast(enabled=amp):
            loss_sum, _loss_items = model.model.loss(batch)

        loss_sum.backward()

        # max_memory_reserved 比 allocated 更能反映 GPU 实际占用
        # （包含 PyTorch 缓存分配器的碎片和预分配池）
        peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        recommendations = _build_recommendations(peak_gb, batch_size, model_name)

        send_queue.put((peak_gb, recommendations))

    except Exception as e:
        send_queue.put((None, [f"Dry Run 异常: {type(e).__name__}: {e}"]))


def _build_recommendations(peak_gb, batch_size, model_name):
    """根据预估值生成静态通用建议。"""
    recs = []
    if peak_gb > 3.1:
        recs.append(f"预估值 {peak_gb:.2f} GB 超过 3.1 GB 硬限制，建议:")
        if batch_size > 1:
            recs.append(f"  - 降低 batch_size（当前 {batch_size} → {max(1, batch_size // 2)}）")
        if "p2" in model_name:
            recs.append("  - 关闭 P2 检测头（model_name='yolov8n'）")
        recs.append("  - 关闭多尺度训练（multiscale=False）")
        recs.append("  - 确认 AMP 已启用（amp=True）")
    return recs


# ══════════════════════════════════════════════════════════════════
# 公开 API
# ══════════════════════════════════════════════════════════════════


def estimate_vram(model_name="yolov8n", pretrained_path="yolov8n.pt",
                  nc=10, input_size=(640, 640), batch_size=2, amp=True):
    """通过独立子进程 Dry Run 实测训练所需显存。

    接口 I-05。
    子进程自行重建模型（传递配置参数，避免 Windows spawn 模式下
    CUDA tensor 不可 pickle 的问题）。

    Args:
        model_name: 模型名称 "yolov8n" | "yolov8n-p2"
        pretrained_path: 预训练权重路径
        nc: 类别数
        input_size: 输入尺寸 (h, w)，默认 (640, 640)
        batch_size: 批大小
        amp: 是否开启混合精度

    Returns:
        (peak_vram_gb, recommendations)
        - peak_vram_gb: 峰值显存（GB），误差 ±15%
        - recommendations: 若预估值 > 3.1 GB 则返回静态建议列表
    """
    if not _cuda_available():
        logger.error("CUDA 不可用，无法执行显存预估")
        return 0.0, ["CUDA 不可用"]

    # Windows spawn 模式：必须用 ctx 显式指定，并锁定当前 Python 解释器
    ctx = mp.get_context("spawn")
    ctx.set_executable(sys.executable)  # 确保子进程与主进程使用同一 Python
    recv_queue = ctx.Queue()

    project_root = str(Path(__file__).resolve().parent.parent)

    kwargs = {
        "send_queue": recv_queue,
        "model_name": model_name,
        "pretrained_path": pretrained_path,
        "nc": nc,
        "input_size": input_size,
        "batch_size": batch_size,
        "amp": amp,
        "project_root": project_root,
    }

    proc = ctx.Process(target=_dry_run_worker, kwargs=kwargs)
    proc.start()
    proc.join(timeout=300)  # 最多等 5 分钟

    if proc.is_alive():
        logger.error("Dry Run 超时（5 分钟），强制终止子进程")
        proc.terminate()
        proc.join(timeout=10)
        return 0.0, ["Dry Run 超时，请检查 GPU 状态后重试"]

    if recv_queue.empty():
        return 0.0, ["子进程未返回结果，请检查日志"]

    peak_gb, recs = recv_queue.get()
    if peak_gb is None:
        logger.error("Dry Run 失败: %s", recs)
        return 0.0, recs

    logger.info("显存预估完成: %.2f GB (batch=%d, amp=%s, model=%s)",
                peak_gb, batch_size, amp, model_name)

    if peak_gb > 3.1:
        logger.warning("预估值 %.2f GB 超过 3.1 GB 限制", peak_gb)
        for r in recs:
            logger.warning(r)

    return peak_gb, recs


def get_vram_info():
    """获取当前 GPU 显存使用信息。

    供 Monitor 回调使用（步骤 13）。

    Returns:
        dict: {"allocated_gb": float, "peak_gb": float, "total_gb": float}
        失败时返回 None
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return {"allocated_gb": allocated, "reserved_gb": reserved, "peak_gb": peak, "total_gb": total}
    except Exception:
        return None


def reset_peak_memory_stats():
    """重置 CUDA 峰值显存统计。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
