# -*- coding: utf-8 -*-
"""
E2E integration test.

50-image mini dataset -> data loading -> train 3 epochs -> eval -> predict -> ONNX export.
Target: <10 minutes.
"""

import os
import shutil
from pathlib import Path

import pytest

REAL_TRAIN_ROOT = "D:/Data/VisDrone/train"
REAL_VAL_ROOT = "D:/Data/VisDrone/val"
REAL_PRETRAINED = os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt")


def _skip_if_no_data():
    if not os.path.isdir(REAL_TRAIN_ROOT):
        pytest.skip("train data not found")
    if not os.path.isfile(REAL_PRETRAINED):
        pytest.skip("pretrained weights not found")
    try:
        import torch  # noqa
    except OSError:
        pytest.skip("torch import failed (page file too small)")


def _create_mini_dataset(src_root, dst_root, n=25):
    """Copy first n images+annotations from src to dst."""
    src = Path(src_root)
    dst = Path(dst_root)
    dst_img = dst / "images"
    dst_ann = dst / "annotations"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_ann.mkdir(parents=True, exist_ok=True)

    img_dir = src / "images"
    ann_dir = src / "annotations"
    copied = 0
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        stem = f.stem
        ann_file = ann_dir / f"{stem}.txt"
        if not ann_file.is_file():
            continue
        shutil.copy2(f, dst_img / f.name)
        shutil.copy2(ann_file, dst_ann / ann_file.name)
        copied += 1
        if copied >= n:
            break
    return copied


class TestE2E:

    def test_e2e_full_pipeline(self, tmp_path):
        """Full E2E: mini dataset -> train 3 epochs -> eval -> predict -> ONNX."""
        _skip_if_no_data()

        # 1. mini dataset (25 train + 25 val = 50 images)
        mini_train = tmp_path / "train"
        mini_val = tmp_path / "val"
        n_train = _create_mini_dataset(REAL_TRAIN_ROOT, mini_train, n=25)
        n_val = _create_mini_dataset(REAL_VAL_ROOT, mini_val, n=25)
        assert n_train == 25 and n_val == 25, "failed to create mini dataset"
        print(f"mini dataset: train={n_train}, val={n_val}")

        # 2. data conversion
        from data.data_loader import prepare_dataset
        out_base = tmp_path / "converted"
        data_yaml = prepare_dataset(
            train_root=str(mini_train), val_root=str(mini_val),
            test_root=None, output_base=str(out_base), val_split_ratio=0.0,
        )
        assert os.path.isfile(data_yaml), f"data.yaml missing: {data_yaml}"

        # 3. model build
        from models.model_builder import build_model
        model = build_model("yolov8n", REAL_PRETRAINED, nc=10)
        assert model is not None

        # 4. train 3 epochs
        from config.config_loader import load_config
        config = load_config()
        config.train.epochs = 3
        config.train.batch_size = 2
        config.train.output_dir = str(tmp_path / "e2e_runs")
        config.train.name = "e2e_test"
        config.data.output_base = str(out_base)
        config.system.num_workers = 0

        from train.trainer import run_train
        best_path, log_dir = run_train(config=config, model=model)
        assert os.path.isfile(best_path), f"best.pt missing: {best_path}"
        print(f"train done: {best_path}")

        # 5. eval
        from eval.evaluator import run_eval
        result = run_eval(best_path, data_yaml, batch_size=2)
        assert result.mAP50 > 0.0, "mAP@50 is 0"
        assert isinstance(result.per_class_AP, dict)
        print(f"eval: mAP@50={result.mAP50:.4f}")

        # 6. predict
        from inference.predictor import run_predict
        images_dir = mini_val / "images"
        results = run_predict(best_path, str(images_dir), conf=0.5)
        assert len(results) > 0, "no prediction results"
        assert all(r["mode"] == "direct" for r in results)
        print(f"predict: {len(results)} images")

        # 7. ONNX export
        from export.exporter import export_model
        onnx_path = export_model(best_path, format="onnx")
        assert os.path.isfile(onnx_path), f"ONNX missing: {onnx_path}"
        size_mb = os.path.getsize(onnx_path) / 1e6
        assert 5 < size_mb < 20, f"bad ONNX size: {size_mb:.1f} MB"
        print(f"ONNX: {size_mb:.1f} MB")

        print("E2E OK")


class TestE2EMini:

    def test_data_pipeline_only(self, tmp_path):
        """Data pipeline only (no GPU)."""
        _skip_if_no_data()
        mini = tmp_path / "mini"
        n = _create_mini_dataset(REAL_TRAIN_ROOT, mini / "train", n=10)
        assert n == 10
        from data.data_loader import load_dataset
        pairs, stats = load_dataset(str(mini / "train"))
        assert stats["paired"] == 10
        assert stats["skipped_no_annotation"] == 0
        print(f"data pipeline: paired={stats['paired']}")

    def test_model_build(self):
        """Model build + forward inference."""
        _skip_if_no_data()
        from models.model_builder import build_model
        model = build_model("yolov8n", REAL_PRETRAINED, nc=10)
        assert model is not None
        import torch, numpy as np
        model.model.eval()
        with torch.no_grad():
            dummy = torch.from_numpy(np.random.randn(1, 3, 640, 640).astype(np.float32))
            out = model.model(dummy)
        assert out is not None
        print("model forward OK")

    def test_onnx_export_only(self):
        """ONNX export using trained best.pt."""
        _skip_if_no_data()
        model_path = os.path.join(
            os.path.dirname(__file__), "..",
            "runs", "train", "visdrone_baseline", "weights", "best.pt"
        )
        if not os.path.isfile(model_path):
            pytest.skip("model not found")
        from export.exporter import export_model
        onnx_path = export_model(model_path, format="onnx")
        assert os.path.isfile(onnx_path)
        size_mb = os.path.getsize(onnx_path) / 1e6
        assert 5 < size_mb < 20
        print(f"ONNX: {size_mb:.1f} MB")
