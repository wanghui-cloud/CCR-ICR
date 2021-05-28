from __future__ import print_function

import numpy as np

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# 根据几种尺度和比例生成的anchor

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

# 生成基础的anchor
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    # 请注意anchor的表示形式有两种，一种是记录左上角和右下角的坐标，一种是记录中心坐标和宽高
    # 这里生成一个基准anchor，采用左上角和右下角的坐标表示[0,0,15,15]
    # base_anchor   ->  array([0, 0, 15, 15])
    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    # 返回不同长宽比ratio的anchor
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    ''' 返回：[   [-3.5  2.  18.5 13. ]
                 [ 0.   0.  15.  15. ]
                 [ 2.5 -3.  12.5 18. ]   ]'''

    # 返回不同尺度scales的扩展的anchor
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    # vstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组，就是横着排起来
    '''返回： [[ -84.  -40.   99.   55.]
              [-176.  -88.  191.  103.]
              [-360. -184.  375.  199.]
              [ -56.  -56.   71.   71.]
              [-120. -120.  135.  135.]
              [-248. -248.  263.  263.]
              [ -36.  -80.   51.   95.]
              [ -80. -168.   95.  183.]
              [-168. -344.  183.  359.]]'''
    # 最后的anchor是二维数组,每行为一个框
    return anchors

# 传入anchor的左上角和右下角的坐标，返回anchor的中心坐标和长宽
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

#输入 ws[23,16,11] hs[12,16,22] 中心点坐标7.5
# 把给的anchor合在一起，按列排
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis] #np.newaxis 在使用和功能上等价于 None，其实就是 None 的一个别名。
    hs = hs[:, np.newaxis]
    # 计算以此为中心点坐标，不同anchor的左上、右下坐标
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1),))

    return anchors

# 输入：anchors:[0,0,15,15]  ratio:[0.5,1,2]
# 计算不同长宽比ratio的anchor坐标
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    # 得到base_anchor的宽,高和中心坐标 16 16 7.5 7.5
    w, h, x_ctr, y_ctr = _whctrs(anchor)   # 找到anchor的中心点和长宽
    size = w * h   # 256  # 返回anchor的面积
    # 得到一组 size_ratios 是w^2
    size_ratios = size / ratios # 计算anchor的长宽尺度设置的数组 [512,256,128]
    # 不同长宽比下的anchor的宽度 四舍五入
    # 为什么要开根号来获得ws呢？我这里理解成通过面积来计算正方形的边长作为ws
    # 使ws:hs满足对应比例的同时，并且ws*hs的大小与base_anchor的大小相近的一种策略
    ws = np.round(np.sqrt(size_ratios))  # [23,16,11]
    # 不同长宽比下的anchor的高度
    hs = np.round(ws * ratios)  # [12,16,22]
    # 创建anchors
    # 以x_ctr, y_ctr为中心点的三个比例框的左上、右下坐标值
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors   # (3,4)

# 不同尺度scales的扩展的anchor
# 在得到的ratio_anchors中的三种宽高比的anchor，分别进行三种scale的变换，
# 搭配三种scale，最终会得到9种宽高比和scale的anchors
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)  # 找到anchor的中心坐标
    ws = w * scales   # shape [3,] 得到不同尺度的新的宽
    hs = h * scales   # shape [3,] 得到不同尺度的新的高
    # 创建anchors
    # 以x_ctr, y_ctr为中心点的三个比例框的左上、右下坐标值
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)  # 把这个比例下的anchor保留下来
    return anchors


if __name__ == "__main__":
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed

    embed()
