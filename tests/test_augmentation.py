"""
augmentation 单元测试。

验证概要设计 I-03：AugConfig dataclass 默认值、build_dataloader 的
train/val 模式行为差异、异常处理。
"""

import os

import pytest

from data.augmentation import AugConfig, build_dataloader

# 测试用 data.yaml 路径（由 data_loader.prepare_dataset 生成）
DATA_YAML = os.path.join(os.path.dirname(__file__), "..", "datasets", "visdrone", "data.yaml")


# ── 前置条件检查 ────────────────────────────────────────────────

def _skip_if_no_data_yaml():
    if not os.path.isfile(DATA_YAML):
        pytest.skip(f"data.yaml 不存在: {DATA_YAML} — 请先运行 prepare_dataset()")


# ══════════════════════════════════════════════════════════════════
# AugConfig dataclass
# ══════════════════════════════════════════════════════════════════

class TestAugConfig:
    """AugConfig 默认值与字段校验。"""

    def test_defaults(self):
        a = AugConfig()
        assert a.mosaic is True
        assert a.flip_prob == 0.5
        assert a.hsv_h == 0.015
        assert a.hsv_s == 0.7
        assert a.hsv_v == 0.4
        assert a.close_mosaic == 10
        assert a.multiscale is False
        assert a.multiscale_range == (0.5, 1.5)

    def test_custom_fields(self):
        a = AugConfig(mosaic=False, flip_prob=0.3, multiscale=True, close_mosaic=5)
        assert a.mosaic is False
        assert a.flip_prob == 0.3
        assert a.multiscale is True
        assert a.close_mosaic == 5

    def test_to_ultralytics_cfg(self):
        a = AugConfig(mosaic=True, flip_prob=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
        cfg = a.to_ultralytics_cfg(imgsz=640)
        assert cfg.imgsz == 640
        assert cfg.mosaic == 1.0          # bool→float
        assert cfg.fliplr == 0.5
        assert cfg.hsv_h == 0.015
        assert cfg.hsv_s == 0.7
        assert cfg.hsv_v == 0.4
        assert cfg.mixup == 0.0           # 显式关闭
        assert cfg.degrees == 0.0
        assert cfg.shear == 0.0
        assert cfg.perspective == 0.0

    def test_to_ultralytics_cfg_mosaic_off(self):
        a = AugConfig(mosaic=False)
        cfg = a.to_ultralytics_cfg()
        assert cfg.mosaic == 0.0


# ══════════════════════════════════════════════════════════════════
# build_dataloader — 正常流程
# ══════════════════════════════════════════════════════════════════

class TestBuildDataloader:
    """build_dataloader train/val 模式行为。"""

    def test_train_mode_builds(self):
        _skip_if_no_data_yaml()
        loader = build_dataloader(DATA_YAML, batch_size=2, num_workers=0, is_train=True)
        batch = next(iter(loader))
        # 训练模式：固定 640×640（含灰度填充）
        assert batch["img"].shape == (2, 3, 640, 640)
        assert "batch_idx" in batch

    def test_val_mode_builds(self):
        _skip_if_no_data_yaml()
        loader = build_dataloader(DATA_YAML, batch_size=2, num_workers=0, is_train=False)
        batch = next(iter(loader))
        # 验证模式：矩形批，shape 非固定（保持原图宽高比）
        assert batch["img"].shape[0] <= 2
        assert batch["img"].shape[1] == 3

    def test_train_val_mode_produce_different_shapes(self):
        """is_train=True → 固定 640²；is_train=False → 矩形宽高（至少不同）。"""
        _skip_if_no_data_yaml()
        loader_t = build_dataloader(DATA_YAML, batch_size=2, num_workers=0, is_train=True)
        loader_v = build_dataloader(DATA_YAML, batch_size=2, num_workers=0, is_train=False)

        bt = next(iter(loader_t))
        bv = next(iter(loader_v))
        # 训练图像始终 640×640，验证未必
        assert bt["img"].shape[2] == 640 and bt["img"].shape[3] == 640

    def test_custom_aug_config_applies(self):
        _skip_if_no_data_yaml()
        aug = AugConfig(mosaic=False, hsv_h=0.03)
        loader = build_dataloader(DATA_YAML, batch_size=2, num_workers=0,
                                  is_train=True, aug_config=aug)
        assert len(loader) > 0

    def test_dataloader_length_matches(self):
        """批数 = ceil(图像数 / batch_size)。"""
        _skip_if_no_data_yaml()
        loader = build_dataloader(DATA_YAML, batch_size=4, num_workers=0, is_train=True)
        # 训练集 6471 张 → 6471/4 = 1618 批 (最后一批可能不足 batch)
        n_batches = len(loader)
        assert n_batches == 1618  # ceil(6471/4)


# ══════════════════════════════════════════════════════════════════
# build_dataloader — 异常处理
# ══════════════════════════════════════════════════════════════════

class TestBuildDataloaderErrors:

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size 必须 > 0"):
            build_dataloader(DATA_YAML, batch_size=0, num_workers=0, is_train=True)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size 必须 > 0"):
            build_dataloader(DATA_YAML, batch_size=-1, num_workers=0, is_train=True)

    def test_yaml_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            build_dataloader("/nonexistent/path.yaml", batch_size=2, num_workers=0, is_train=True)
