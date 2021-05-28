# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import numpy as np


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    # 通过不同尺度因子的缩放，从一系列的框中获取不重复框的数组指标，用于过滤重复框
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)  # 外积后转换为一个数
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    # 用于去除无效框，即超出边界或者左右，上下，不满足几何关系的点，比如右边顶点的x坐标小于左边顶点
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()


def filter_small_boxes(boxes, min_size):
    # 用于去掉过小的检测框
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep
