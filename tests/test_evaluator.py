"""
evaluator 单元测试。

覆盖 EvalResult dataclass、run_eval 正常流程和异常处理。
"""

import os

import pytest

from eval.eval_result import EvalResult
from eval.evaluator import run_eval, _load_class_names, _get_class_instance_count

DATA_YAML = os.path.join(os.path.dirname(__file__), "..", "datasets", "visdrone", "data.yaml")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "runs", "train",
                          "visdrone_baseline", "weights", "best.pt")


def _skip_if_no_model():
    if not os.path.isfile(MODEL_PATH):
        pytest.skip(f"模型不存在: {MODEL_PATH}")
    if not os.path.isfile(DATA_YAML):
        pytest.skip(f"data.yaml 不存在: {DATA_YAML}")
    try:
        import torch  # noqa
    except OSError:
        pytest.skip("torch 导入失败（页面文件不足），跳过 GPU 测试")


class TestEvalResult:

    def test_default_fields(self):
        er = EvalResult()
        assert er.mAP50 == 0.0
        assert er.per_class_AP == {}
        assert er.high_fn_classes == []
        assert er.skipped_classes == []

    def test_fields_populated(self):
        er = EvalResult(
            mAP50=0.308,
            mAP50_95=0.179,
            precision=0.415,
            recall=0.309,
            per_class_AP={"car": 0.738, "pedestrian": 0.318},
            high_fn_classes=["pedestrian"],
        )
        assert er.mAP50 == 0.308
        assert er.per_class_AP["car"] == 0.738
        assert "pedestrian" in er.high_fn_classes

    def test_per_class_summary(self):
        er = EvalResult(
            per_class_AP={"car": 0.738, "pedestrian": 0.318},
            fn_rate_per_class={"car": 0.26, "pedestrian": 0.66},
        )
        summary = er.per_class_summary
        assert "car" in summary
        assert "pedestrian" in summary


class TestRunEval:

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_eval("nonexistent.pt", "nonexistent.yaml")

    def test_model_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_eval("nonexistent.pt", DATA_YAML)

    def test_run_eval_smoke(self):
        _skip_if_no_model()
        result = run_eval(MODEL_PATH, DATA_YAML, batch_size=4)
        assert result.mAP50 > 0.2
        assert result.mAP50_95 > 0.1
        assert len(result.per_class_AP) == 10
        assert result.per_class_AP["car"] > 0.5
        # 至少 car 的 FN 率 < 40%
        assert result.fn_rate_per_class.get("car", 1.0) < 0.4

    def test_per_class_metrics_match(self):
        _skip_if_no_model()
        result = run_eval(MODEL_PATH, DATA_YAML, batch_size=4)
        for cls_name in result.per_class_AP:
            assert cls_name in result.per_class_P
            assert cls_name in result.per_class_R
            assert cls_name in result.fn_rate_per_class
            # FN rate = 1 - recall
            expected_fn_rate = 1.0 - result.per_class_R[cls_name]
            assert abs(result.fn_rate_per_class[cls_name] - expected_fn_rate) < 1e-4


class TestHelperFunctions:

    def test_class_instance_counts(self):
        assert _get_class_instance_count("", "car") == 14064
        assert _get_class_instance_count("", "pedestrian") == 8844
        assert _get_class_instance_count("", "bicycle") == 1287

    def test_load_class_names(self):
        names = _load_class_names(DATA_YAML, None)
        assert len(names) >= 10
        assert "car" in names
        assert "pedestrian" in names
