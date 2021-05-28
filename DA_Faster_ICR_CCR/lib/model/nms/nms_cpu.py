from __future__ import absolute_import

import numpy as np
import torch


def nms_cpu(dets, thresh):
    # dets [12000, 5]= proposals_single, scores_single.squeeze(1)
    dets = dets.numpy()
    # x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]  # argsort函数返回的是数组值从小到大的索引值

    # 保留的结果框集合
    keep = []
    while order.size > 0:
        # 保留得分最高的一个的索引
        i = order.item(0)
        keep.append(i)      # 将其作为保留的框
        # 计算置信度最大的框（order[0]）与其它所有的框（order[1:]，即第二到最后一个）框的IOU，
        xx1 = np.maximum(x1[i], x1[order[1:]]) # 逐位比较取其大者
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h   #计算相交框的面积
        # 计算IOU：重叠面积/(面积1+面积2-重叠面积)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 取出IOU小于阈值的框
        # 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)
        inds = np.where(ovr <= thresh)[0]
        # 更新排序序列
        order = order[inds + 1]
        # 删除IOU大于阈值的框，因为从第二个数开始，当作第一个数，所以需要+1，如[1,2,3,4],将从[2,3,4]开始，
        # 若选择第一个数2，下标为0，所以需要+1，才能对应原来数[1,2,3,4],选择为2.

    return torch.IntTensor(keep)  # 返回索引值
