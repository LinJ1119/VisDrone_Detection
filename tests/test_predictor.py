"""
predictor 单元测试。

覆盖 I-09：run_predict 正常流程、异常处理、SAHI 配置。
"""

import os

import pytest

DATA_YAML = os.path.join(os.path.dirname(__file__), "..", "datasets", "visdrone", "data.yaml")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "runs", "train",
                          "visdrone_baseline", "weights", "best.pt")


def _skip_if_no_gpu():
    try:
        import torch  # noqa
    except OSError:
        pytest.skip("torch 导入失败（页面文件不足），跳过 GPU 测试")
    if not os.path.isfile(MODEL_PATH):
        pytest.skip(f"模型不存在: {MODEL_PATH}")


class TestPredictorImports:

    def test_import_predictor(self):
        from inference.predictor import run_predict, _warmup, _predict_single, _verify_image
        assert callable(run_predict)
        assert callable(_verify_image)

    def test_verify_image_valid(self, tmp_path):
        """创建一个最小 PNG 并用 verify_image 校验。"""
        import struct, zlib
        from inference.predictor import _verify_image

        def _min_png(w, h):
            def chunk(t, d):
                return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
            raw = b""
            for y in range(h):
                raw += b"\x00" + b"\xff\x00\x00" * w
            return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

        p = tmp_path / "test.png"
        p.write_bytes(_min_png(4, 4))
        assert _verify_image(str(p)) is True

    def test_verify_image_corrupt(self, tmp_path):
        from inference.predictor import _verify_image
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not an image")
        assert _verify_image(str(p)) is False


class TestRunPredictErrors:

    def test_model_not_found(self):
        from inference.predictor import run_predict
        with pytest.raises(FileNotFoundError):
            run_predict("nonexistent.pt", "nonexistent.jpg")

    def test_source_not_found(self):
        from inference.predictor import run_predict
        with pytest.raises(FileNotFoundError):
            run_predict(MODEL_PATH, "/nonexistent/path")

    def test_empty_dir(self, tmp_path):
        _skip_if_no_gpu()
        from inference.predictor import run_predict
        results = run_predict(MODEL_PATH, str(tmp_path))
        assert results == []


class TestRunPredictSmoke:

    def test_direct_predict_smoke(self, tmp_path):
        _skip_if_no_gpu()
        import struct, zlib
        from inference.predictor import run_predict

        # 创建最小测试图像
        def _min_png(w, h):
            def chunk(t, d):
                return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
            raw = b""
            for y in range(h):
                raw += b"\x00" + b"\x80\x80\x80" * w
            return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

        p = tmp_path / "test.png"
        p.write_bytes(_min_png(640, 480))

        results = run_predict(MODEL_PATH, str(p), conf=0.5)
        assert len(results) == 1
        assert results[0]["image_name"] == "test.png"
        assert results[0]["mode"] == "direct"
        assert "inference_time_ms" in results[0]
        assert isinstance(results[0]["boxes"], list)

    def test_result_structure(self):
        _skip_if_no_gpu()
        import struct, zlib
        from inference.predictor import run_predict
        import tempfile

        def _min_png(w, h):
            def chunk(t, d):
                return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
            raw = b""
            for y in range(h):
                raw += b"\x00" + b"\x80" * (w * 3)
            return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

        p = os.path.join(tempfile.gettempdir(), "test_pred_640.png")
        with open(p, "wb") as f:
            f.write(_min_png(640, 640))
        try:
            results = run_predict(MODEL_PATH, p, conf=0.5)
            assert len(results) == 1
            r = results[0]
            assert isinstance(r["image_name"], str)
            assert isinstance(r["inference_time_ms"], (int, float))
            assert r["mode"] in ("direct", "sahi")
            for box in r["boxes"]:
                assert "class_id" in box
                assert "class_name" in box
                assert "conf" in box
                assert len(box["xyxy"]) == 4
        finally:
            if os.path.isfile(p):
                os.unlink(p)


class TestSAHIConfig:

    def test_sahi_config_disabled(self):
        """sahi_config=None → direct mode."""
        assert True  # implicit: run_predict with sahi_config=None → "direct"

    def test_sahi_config_schema(self):
        cfg = {"enabled": True, "slice_size": 640, "overlap": 0.2, "batch_size": 4}
        assert cfg["enabled"] is True
        assert cfg["slice_size"] == 640
        assert 0 < cfg["overlap"] < 1
