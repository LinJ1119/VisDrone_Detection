"""
训练过程实时监控。

接口定义参见概要设计 I-07。
通过 Ultralytics 回调钩子注入，监控显存、吞吐量，写入 TensorBoard。
"""

import logging
import time

import torch

logger = logging.getLogger(__name__)


class Monitor:
    """训练监控器——通过 Ultralytics 回调钩子注入。

    监控项（概要设计 I-07）：
    1. 显存峰值告警（> 3.0 GB → WARNING）
    2. 连续超限自动降 batch_size（下个 epoch 生效）
    3. 吞吐量统计（< 15 张/s 连续 3 epoch → WARNING）
    4. TensorBoard 写入

    使用方式：
        monitor = Monitor()
        trainer.add_callback("on_train_batch_end", monitor.on_train_batch_end)
        trainer.add_callback("on_train_epoch_end", monitor.on_train_epoch_end)
    """

    def __init__(self):
        # 显存超限计数器（仅连续超限时累加，不重置）
        self._vram_over_count = 0
        # 吞吐量统计
        self._batch_times = []
        self._epoch_start_time = None
        self._throughput_low_count = 0
        # TensorBoard writer（由 on_train_start 初始化）
        self._writer = None
        # 当前 batch_size（用于超限降级）
        self._batch_size = None
        # batch 自增计数器（trainer 无 batch_idx 属性）
        self._batch_idx = 0

    # ── 回调入口 ──────────────────────────────────────────────

    def on_train_start(self, trainer):
        """训练开始：初始化 TensorBoard writer。"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = str(trainer.save_dir)
            self._writer = SummaryWriter(log_dir=log_dir)
            self._batch_size = trainer.batch_size
            logger.info("Monitor 已启动，TensorBoard 日志目录: %s", log_dir)
        except ImportError:
            logger.warning("TensorBoard 不可用，跳过写入")
            self._writer = None

    def on_train_batch_end(self, trainer):
        """每个 batch 结束后触发（I-07 主回调）。"""
        self._batch_idx += 1

        # 1) 显存监控
        self._check_vram(trainer)

        # 2) 吞吐量计时
        self._record_batch_time()

        # 3) TensorBoard 写入
        self._write_tensorboard(trainer)

    def on_train_epoch_end(self, trainer):
        """每个 epoch 结束后触发。"""
        # 1) 吞吐量统计与告警
        self._check_throughput()

        # 2) epoch 结束后重置
        self._vram_over_count = 0
        self._batch_times = []
        self._epoch_start_time = None
        # batch_idx 不重置（全局递增）

    # ── 显存监控 ──────────────────────────────────────────────

    def _check_vram(self, trainer):
        """读取当前 batch 显存峰值，判断是否超限。"""
        if not torch.cuda.is_available():
            return

        try:
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

        if peak_gb > 3.0:
            self._vram_over_count += 1
            logger.warning(
                "显存超限！global_step %d: %.2f GB > 3.0 GB（连续 %d 次）",
                self._batch_idx,
                peak_gb, self._vram_over_count
            )

            if self._vram_over_count >= 5:
                new_bs = max(1, trainer.batch_size // 2)
                logger.warning(
                    "连续 %d 次超限，batch_size %d → %d（下个 epoch 生效）",
                    self._vram_over_count, trainer.batch_size, new_bs
                )
                trainer.batch_size = new_bs
                self._vram_over_count = 0
                self._batch_size = new_bs
        else:
            # 非连续超限不重置计数器，保持累积
            pass

    # ── 吞吐量统计 ────────────────────────────────────────────

    def _record_batch_time(self):
        """记录 batch 处理时间。"""
        now = time.perf_counter()
        if self._epoch_start_time is None:
            self._epoch_start_time = now
        self._batch_times.append(now)

    def _check_throughput(self):
        """epoch 结束时计算吞吐量并告警。"""
        if len(self._batch_times) < 2:
            return

        # 用 batch 间隔时间估算吞吐量
        interval = self._batch_times[-1] - self._epoch_start_time
        n_batches = len(self._batch_times) - 1  # 第一个是开始时间，不算
        batch_size = self._batch_size or 1
        throughput = (n_batches * batch_size) / max(interval, 1e-6)

        if throughput < 15.0:
            self._throughput_low_count += 1
            if self._throughput_low_count >= 3:
                logger.warning(
                    "吞吐量连续 %d 个 epoch < 15 张/s（当前 %.1f 张/s），"
                    "建议减少 num_workers 或检查 GPU 负载",
                    self._throughput_low_count, throughput
                )
        else:
            self._throughput_low_count = 0

        logger.info("Epoch 吞吐量: %.1f 张/s", throughput)

    # ── TensorBoard ───────────────────────────────────────────

    def _write_tensorboard(self, trainer):
        """将 batch 级指标写入 TensorBoard。"""
        if self._writer is None:
            return

        global_step = self._batch_idx

        try:
            # 训练 loss
            loss_items = trainer.loss_items
            if hasattr(loss_items, 'tolist'):
                loss_items = loss_items.tolist()
            if loss_items is not None:
                self._writer.add_scalar("train/box_loss", float(loss_items[0]), global_step)
                self._writer.add_scalar("train/cls_loss", float(loss_items[1]), global_step)
                self._writer.add_scalar("train/dfl_loss", float(loss_items[2]), global_step)

            # 学习率
            scheduler = trainer.scheduler
            if scheduler is not None:
                self._writer.add_scalar("train/lr", float(scheduler.get_last_lr()[0]), global_step)
        except Exception:
            pass

    # ── 清理 ──────────────────────────────────────────────────

    def close(self):
        """关闭 TensorBoard writer。"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
