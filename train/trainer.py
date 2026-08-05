"""
训练主控——显存友好型 YOLOv8n 训练。

接口定义参见概要设计 I-06。
委托 Ultralytics 引擎，通过回调注入定制行为（OOM 恢复、原子写入、Monitor）。
"""

import logging
import os
import random
from datetime import datetime

import numpy as np
import torch

from config.config_loader import Config, save_config_snapshot
from train.monitor import Monitor

logger = logging.getLogger(__name__)

# 最大 OOM 重试次数
MAX_OOM_RETRIES = 3


def _setup_log_file(log_dir: str):
    """为根日志器添加 FileHandler，将终端日志同步写入文件。

    在训练输出目录创建后调用，确保完整训练日志持久化。
    """
    log_path = os.path.join(log_dir, "train.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    logger.info("日志文件: %s", log_path)


def set_seed(seed: int = 42):
    """设置全局随机种子（需求 AC-1.4，概要设计 §1.5.1）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 以下确保卷积/池化等操作的确定性（可能降低性能，训练时不强制开启）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info("全局随机种子已设置: seed=%d", seed)


def _flatten_config(cfg: Config):
    """将 Config 对象展平为 Ultralytics model.train() 兼容的关键字参数字典。"""
    return dict(
        # 数据
        data=os.path.join(cfg.data.output_base, "data.yaml"),
        # 模型
        model=os.path.join("models", "model_configs",
                           f"{'yolov8n-p2' if cfg.model.p2_head else 'yolov8n'}.yaml"),
        # 训练超参
        epochs=cfg.train.epochs,
        batch=cfg.train.batch_size,
        imgsz=cfg.model.imgsz,
        lr0=cfg.train.lr0,
        lrf=cfg.train.lrf,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
        optimizer=cfg.train.optimizer,
        amp=cfg.train.amp,
        # 早停
        patience=cfg.train.early_stop_patience,
        # 增强
        close_mosaic=cfg.aug.close_mosaic,
        mosaic=1.0 if cfg.aug.mosaic else 0.0,
        hsv_h=cfg.aug.hsv_h,
        hsv_s=cfg.aug.hsv_s,
        hsv_v=cfg.aug.hsv_v,
        fliplr=cfg.aug.flip_prob,
        multi_scale=cfg.train.multiscale,
        # 正则化
        dropout=cfg.train.dropout,
        # 系统
        workers=cfg.system.num_workers,
        seed=cfg.system.seed,
        # 输出
        project=cfg.train.output_dir,
        name=cfg.train.name if cfg.train.name else "",
        exist_ok=True,
        # 不自动 resume（我们手动管理）
        resume=False,
        # 不自动验证（通过 val=True 在每个 epoch 后验证）
        val=True,
        # 不保存混合格式（仅 FP32）
        half=False,
    )


def run_train(config: Config, model, train_loader=None, val_loader=None,
              resume: str = None, name: str = None):
    """执行完整训练流程。

    接口 I-06。

    Args:
        config: 完整配置对象（Config dataclass）
        model: 已构建的 YOLO 模型（ultralytics.YOLO）
        train_loader: 训练 DataLoader（未使用，由 model.train 内部构建）
        val_loader: 验证 DataLoader（未使用，由 model.train 内部构建）
        resume: checkpoint 路径（None=从头训练）
        name: 实验名称（None=自动时间戳）

    Returns:
        (best_model_path, log_dir)
    """
    # 1) 设置随机种子
    set_seed(config.system.seed)

    # 2) 设置 GPU 显存限制
    if torch.cuda.is_available():
        memory_fraction = config.system.gpu_memory_fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        logger.info("GPU 显存限制: %.0f%% (约 %.1f GB)",
                    memory_fraction * 100, 4.0 * memory_fraction)

    # 3) 展平配置为 ultralytics 参数
    train_args = _flatten_config(config)

    # 覆盖 name（如果指定）；否则自动生成时间戳名称
    if name is not None:
        train_args["name"] = name
        config.train.name = name
    elif not train_args.get("name"):
        train_args["name"] = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        config.train.name = train_args["name"]
    exp_name = train_args["name"]

    # 覆盖 resume 路径
    if resume is not None:
        train_args["resume"] = True
        train_args["model"] = resume
        logger.info("将从 checkpoint 恢复训练: %s", resume)

    # 4) 训练循环（含 OOM 自动恢复）
    current_batch_size = config.train.batch_size
    oom_retries = 0

    while oom_retries <= MAX_OOM_RETRIES:
        train_args["batch"] = current_batch_size

        try:
            logger.info("开始训练: epochs=%d, batch=%d, amp=%s, 显存限制=%.0f%%",
                        train_args["epochs"], current_batch_size,
                        train_args["amp"], config.system.gpu_memory_fraction * 100)

            # 注入 Monitor 回调
            monitor = Monitor()
            model.add_callback("on_train_start", monitor.on_train_start)
            model.add_callback("on_train_batch_end", monitor.on_train_batch_end)
            model.add_callback("on_train_epoch_end", monitor.on_train_epoch_end)

            # 保存配置快照（需求 AC-8.3）
            exp_name = train_args.get("name") or "exp"
            output_dir = os.path.join(str(train_args["project"]), exp_name)
            os.makedirs(output_dir, exist_ok=True)
            _setup_log_file(output_dir)
            save_config_snapshot(config, output_dir)

            # 委托 Ultralytics
            _ = model.train(**train_args)

            # 正常结束
            monitor.close()
            best_path = os.path.join(train_args["project"],
                                     exp_name,
                                     "weights", "best.pt")
            log_dir = os.path.join(train_args["project"],
                                   exp_name)
            logger.info("训练完成，最优模型: %s", best_path)
            return best_path, log_dir

        except Exception as e:
            # OOM 检查
            if _is_oom_error(e):
                oom_retries += 1
                logger.warning("CUDA OOM (第 %d/%d 次): batch=%d",
                               oom_retries, MAX_OOM_RETRIES, current_batch_size)
                torch.cuda.empty_cache()

                new_batch = max(1, current_batch_size // 2)
                if new_batch == current_batch_size:
                    # batch_size=1 仍 OOM
                    if oom_retries >= MAX_OOM_RETRIES:
                        logger.error(
                            "batch_size=1 仍 OOM 连续 %d 次，终止训练。\n"
                            "显存诊断: 请关闭其他 GPU 进程、关闭 P2 头、"
                            "或使用更小的输入尺寸。",
                            oom_retries
                        )
                        raise

                logger.warning("batch_size %d → %d，重建 DataLoader 并重启训练",
                               current_batch_size, new_batch)
                current_batch_size = new_batch
                config.train.batch_size = new_batch
                train_args["batch"] = new_batch
                # 重试：从最近的 checkpoint 恢复
                last_pt = os.path.join(output_dir, "weights", "last.pt")
                if os.path.isfile(last_pt):
                    train_args["resume"] = True
                    train_args["model"] = last_pt
                continue

            # 非 OOM 错误直接上抛
            raise

    # 不应到达这里
    return "", ""


def _is_oom_error(e: Exception) -> bool:
    """判断异常是否为 CUDA OOM。"""
    msg = str(e).lower()
    # RuntimeError: CUDA out of memory
    if "out of memory" in msg:
        return True
    # torch.cuda.OutOfMemoryError
    if hasattr(torch.cuda, "OutOfMemoryError"):
        return isinstance(e, torch.cuda.OutOfMemoryError)
    return False
