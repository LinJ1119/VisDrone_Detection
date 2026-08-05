"""
nms_utils 单元测试。

覆盖标准 NMS 的边界情况。
"""

import numpy as np
import pytest

from utils.nms_utils import nms


class TestNMS:

    def test_empty_input(self):
        assert len(nms([], [], 0.5)) == 0

    def test_single_box(self):
        keep = nms([[0, 0, 10, 10]], [0.9], 0.5)
        assert list(keep) == [0]

    def test_two_non_overlapping(self):
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        keep = nms(boxes, [0.9, 0.8], 0.5)
        assert set(keep.tolist()) == {0, 1}

    def test_two_fully_overlapping_suppress_lower(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        keep = nms(boxes, [0.9, 0.8], 0.5)
        assert list(keep) == [0]

    def test_partial_overlap_below_threshold_keeps_both(self):
        # IoU = 50/150 = 0.333 < 0.5
        boxes = [[0, 0, 10, 10], [5, 0, 15, 10]]
        keep = nms(boxes, [0.9, 0.8], 0.5)
        assert set(keep.tolist()) == {0, 1}

    def test_partial_overlap_above_threshold_suppresses(self):
        # IoU = 50/150 = 0.333 > 0.3 → suppress
        boxes = [[0, 0, 10, 10], [5, 0, 15, 10]]
        keep = nms(boxes, [0.9, 0.8], 0.3)
        assert list(keep) == [0]

    def test_highest_score_first(self):
        boxes = [[0, 0, 10, 10], [1, 1, 9, 9], [2, 2, 8, 8]]
        scores = [0.5, 0.9, 0.7]
        keep = nms(boxes, scores, 0.5)
        assert list(keep) == [1]

    def test_multi_box_chain(self):
        # A overlaps B, B overlaps C, but A doesn't overlap C directly
        boxes = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [20, 20, 30, 30],
        ], dtype=np.float32)
        scores = [0.9, 0.8, 0.7]
        keep = nms(boxes, scores, 0.5)
        assert 2 in keep.tolist()  # third box is independent

    def test_ndarray_input(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = nms(boxes, scores, 0.5)
        assert set(keep.tolist()) == {0, 1}

    def test_all_suppressed_returns_only_best(self):
        # all boxes overlap heavily: IoU ≥ 0.81, threshold 0.7 → only best survives
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [0.5, 0.5, 9.5, 9.5]]
        scores = [0.3, 0.9, 0.4]
        keep = nms(boxes, scores, 0.7)
        assert len(keep) == 1
        assert keep[0] == 1  # highest score (0.9)
