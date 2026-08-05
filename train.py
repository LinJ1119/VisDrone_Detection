"""
训练启动入口。

用法:
    python train.py                          # 使用默认配置训练
    python train.py --config my_config.yaml  # 使用自定义配置
    python train.py --resume last.pt         # 断点续训
    python train.py --name visdrone_baseline # 指定实验名称
    python train.py --batch_size 4           # 命令行覆盖配置
"""

import argparse
import logging
import sys
from pathlib import Path

from config.config_loader import load_config, Config
from data.data_loader import prepare_dataset, VISDRONE_CLASS_NAMES
from models.model_builder import build_model
from train.trainer import run_train
from utils.gpu_memory import estimate_vram

logger = logging.getLogger("train")


def parse_args():
    """解析命令行参数。CLI 参数优先级高于 config.yaml 同名项。"""
    p = argparse.ArgumentParser(
        description="VisDrone YOLOv8n 目标检测训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py
  python train.py --config config/custom.yaml --name exp01
  python train.py --resume runs/train/exp/weights/last.pt
  python train.py --batch_size 4 --epochs 100
        """,
    )
    p.add_argument("--config", type=str, default=None, help="配置文件路径")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    p.add_argument("--name", type=str, default=None, help="实验名称")
    # 以下参数可覆盖配置文件中的同名项
    p.add_argument("--batch_size", type=int, default=None, help="批大小")
    p.add_argument("--epochs", type=int, default=None, help="训练轮数")
    p.add_argument("--lr0", type=float, default=None, help="初始学习率")
    p.add_argument("--amp", type=bool, default=None, help="混合精度训练")
    p.add_argument("--data_root", type=str, default=None, help="数据集根目录（覆盖配置中所有路径）")
    return p.parse_args()


def _apply_cli_overrides(config: Config, args):
    """将 CLI 参数覆盖到 Config 对象上。"""
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
        logger.info("CLI 覆盖: batch_size=%d", args.batch_size)
    if args.epochs is not None:
        config.train.epochs = args.epochs
        logger.info("CLI 覆盖: epochs=%d", args.epochs)
    if args.lr0 is not None:
        config.train.lr0 = args.lr0
        logger.info("CLI 覆盖: lr0=%.4f", args.lr0)
    if args.amp is not None:
        config.train.amp = args.amp
        logger.info("CLI 覆盖: amp=%s", args.amp)
    if args.data_root is not None:
        root = args.data_root.rstrip("/\\")
        config.data.train_root = f"{root}/train"
        config.data.val_root = f"{root}/val"
        config.data.test_root = f"{root}/test-dev"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # 1) 加载配置
    config = load_config(args.config)
    _apply_cli_overrides(config, args)

    # 确定模型名称（P2 头 vs 标准）
    model_name = "yolov8n-p2" if config.model.p2_head else config.model.name

    # 2) 启动前显存预估（概要设计 §1.5.2 三层防护步骤 1-2）
    logger.info("=" * 50)
    logger.info("步骤 1: 显存预估 (Dry Run)...")
    peak_gb, recs = estimate_vram(
        model_name=model_name,
        pretrained_path=config.model.pretrained,
        nc=config.model.nc,
        batch_size=config.train.batch_size,
        amp=config.train.amp,
    )
    if peak_gb > 3.1:
        logger.error("显存预估 %.2f GB > 3.1 GB，拒绝运行", peak_gb)
        for r in recs:
            logger.error(r)
        sys.exit(1)
    logger.info("显存预估: %.2f GB (满足 ≤3.1 GB 要求)", peak_gb)

    # 3) 数据准备（概要设计 §3.1 步骤 2）
    logger.info("=" * 50)
    logger.info("步骤 2: 数据准备...")
    data_yaml = prepare_dataset(
        train_root=config.data.train_root,
        val_root=config.data.val_root,
        test_root=config.data.test_root,
        output_base=config.data.output_base,
        val_split_ratio=config.data.val_split_ratio,
        seed=config.system.seed,
        nc=config.data.nc,
        names=VISDRONE_CLASS_NAMES,
    )
    logger.info("data.yaml: %s", data_yaml)
    # 确保 data_yaml 路径作为 data 参数传给训练
    # _flatten_config(trainer.py) 已使用 config.data.output_base 推导

    # 4) 模型构建
    logger.info("=" * 50)
    logger.info("步骤 3: 模型构建 (%s)...", model_name)
    model = build_model(
        model_name=model_name,
        pretrained_path=config.model.pretrained,
        nc=config.model.nc,
    )
    logger.info("模型就绪")

    # 5) 启动训练
    logger.info("=" * 50)
    logger.info("步骤 4: 开始训练")
    best_path, log_dir = run_train(
        config=config,
        model=model,
        resume=args.resume,
        name=args.name,
    )

    logger.info("=" * 50)
    logger.info("训练完成！最优模型: %s", best_path)
    logger.info("日志目录: %s", log_dir)


if __name__ == "__main__":
    main()
