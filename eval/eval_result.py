"""
评估结果数据结构。

定义 EvalResult dataclass，供 I-08（evaluator）和 I-12（visualizer）共享。
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvalResult:
    """模型评估结果。

    字段与概要设计 I-08 一致。
    """
    # 总体指标
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # 每类别指标
    per_class_AP: Dict[str, float] = field(default_factory=dict)
    per_class_P: Dict[str, float] = field(default_factory=dict)
    per_class_R: Dict[str, float] = field(default_factory=dict)

    # 按目标尺寸分层
    size_stratified_mAP: Dict[str, float] = field(default_factory=dict)
    # keys: "small" (area<32²), "medium" (32²≤area<96²), "large" (area≥96²)

    # 错误分析
    tp_fp_fn: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # 格式: {"car": {"tp": N, "fp": N, "fn": N}, ...}
    fn_rate_per_class: Dict[str, float] = field(default_factory=dict)
    high_fn_classes: List[str] = field(default_factory=list)  # FN rate > 40%

    # 验证集跳过
    skipped_classes: List[str] = field(default_factory=list)

    @property
    def per_class_summary(self) -> str:
        """每类 mAP@50 的简短文本摘要。"""
        lines = []
        for cls_name in sorted(self.per_class_AP.keys()):
            ap = self.per_class_AP[cls_name]
            fn_r = self.fn_rate_per_class.get(cls_name, 0.0)
            flag = " ⚠️ FN>40%" if fn_r > 0.4 else ""
            lines.append(f"  {cls_name:20s}: mAP@50={ap:.4f}{flag}")
        return "\n".join(lines)
