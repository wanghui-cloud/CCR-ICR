# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import numpy as np
import torch
# ==========================================
# 函数输入: 两个区域的左下角右上角坐标，一个是预测的ROI，另一个是ground_truth
# 函数输出: x, y, w, h的偏移量
# ==========================================
# 这个函数主要计算了一个给定的ROI和一个ground_truth之间的偏移量大小
# 注意在计算坐标变换的时候是将anchor的表示形式变成中心坐标与长宽
def bbox_transform(ex_rois, gt_rois):
    # 计算的是两个N * 4的矩阵之间的相关回归矩阵，两个输入矩阵一个是anchors，一个是gt_boxes
    # 本质上是在求解每一个anchor相对于它的对应gt_box的（dx, dy, dw, dh）的四个回归值，
    # 返回结果的shape为[N, 4]，使用了log指数变换

    # 计算得到每个anchor的中心坐标和长宽
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # 计算每个anchor对应的ground truth box对应的中心坐标和长宽
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # 计算四个坐标变换值
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    # 对于每一个anchor，得到四个关系值 shape: [4, num_anchor]
    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)

    return targets

# ========================================
# 函数输入：生成的一系列区域以及GT的左下右上坐标，其中ex_rois可以不是batch
# 函数输出：返回计算他们之间的偏移量
# ========================================
# 这个函数和上一个完成的是相同的任务，但是区别是可以进行batch操作
# 其中ex_rois可以传入batch的，也可以传单个图像的一系列ROIs，但是GT都是以batch传入的
def bbox_transform_batch(ex_rois, gt_rois):
    # 这里判断了一下ex_rois是不是批量的，也就是batch_size的形式
    # 如果是2，证明不是batch的形式
    # 然后计算了这一系列的ex_rois的w,h,x,y

    # 举个例子，比如ex_rois是[10, 4]，gt_rois是[10, 10, 4]
    # 那么对于ex算出来的w是[10]，然后通过torch.view操作变成了(1, 10),再通过.expand_as变成了
    # [10, 10]这样就保证了维度相同，可以运算了
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1, -1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1, -1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1, -1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1, -1).expand_as(gt_heights))

    # 这里是如果ex也是batch的形式就不用那么复杂直接计算就可以，返回的结果是(10, 10, 4)
    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError("ex_roi input dimension is not correct.")
    # 这里是对结果的拼接
    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 2)

    return targets

# 结合RPN的输出对所有初始框anchors进行了坐标变换
# 输入:boxes:[B,N,4],表示原始anchors，即未经任何处理仅仅是经过平移之后产生的anchors
#     deltas:[B,N,4],RPN网络产生的数据，即网络'rpn_bbox_pred'层的输出，N表示anchors的数目，
# 函数输出:每个类别的pred_boxes的左下右上的坐标
def bbox_transform_inv(boxes, deltas, batch_size):
    # 用于将rpn网络产生的deltas进行变换处理，求出变换后的对应到原始图像空间的boxes，

    # 获得初始proposal的中心和长宽信息
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    # 获得坐标变换信息
    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    # 得到改变后的proposal的中心和长宽信息
    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    # 将改变后的proposal的中心和长宽信息还原成左上角和右下角的版本
    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

#=========================================
# 函数输入:一系列的待选框左下右上坐标，图像大小，batch_size
# 函数输出:限制边界后的bbox
#=========================================
# 这个函数是将boxes的边界限制在图像范围之内，防止那些越界的边界框
# 由于也是batch的形式，会有一个维度是batch_siz
def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    # 这里面先取了一下每个图像有几个rois，因为这个是boxes的第二维
    num_rois = boxes.size(1)

    # 这里判断了一下，如果预测框已经小于零了证明已经出界了，这时候让他们等于0
    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    # 由于比如256大小的图像范围是0-255，所以-1来得到坐标的最大值
    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    # 分别判断左下右上的坐标是否越过最大值，如果越过就让他等于最大值
    boxes[:, :, 0][boxes[:, :, 0] > batch_x] = batch_x
    boxes[:, :, 1][boxes[:, :, 1] > batch_y] = batch_y
    boxes[:, :, 2][boxes[:, :, 2] > batch_x] = batch_x
    boxes[:, :, 3][boxes[:, :, 3] > batch_y] = batch_y

    return boxes

 #严格限制proposal的四个角在图像边界内
# =================================================
# 函数输入:预测框的坐标，图像大小(这里可以使不同大小的batch的形式)，batch_size
# 函数输出:限制之后的预测框
# =================================================
def clip_boxes(boxes, im_shape, batch_size):
    # 迭代每一个图像大小，进行限制
    # .clamp函数就是限制函数，参数是最小值最大值，这样将他们限制在图像中
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0] - 1)

    return boxes

 ##计算重合程度，两个框之间的重合区域的面积 / 两个区域一共加起来的面

# ====================================================
# 函数输入:一系列的anchors和一系列的gt的左下右上坐标
# 函数输出:这一系列anchors和gt之间交并比IOU返回的结果是(N, K)大小的，即两两之间都做了比较
# ====================================================
def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    # N是anchors的数量，K是gt_boxes的数量
    N = anchors.size(0)
    K = gt_boxes.size(0)

    # 这里首先先将两个区域的面积计算出来，然后将gt的面积转化为(1, k)的维度
    # 将anchors的维度转换为(N, 1)的维度
    gt_boxes_area = ((gt_boxes[:, 2] - gt_boxes[:, 0] + 1) *
                     (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)).view(1, K)

    anchors_area = ((anchors[:, 2] - anchors[:, 0] + 1) *
                    (anchors[:, 3] - anchors[:, 1] + 1)).view(N, 1)

    # 这里将待选框和gt转换为同一维度都是(N,K,4)
    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    # 这里找到右上x坐标与左下x坐标的最小值以及右上x坐标的最大值
    # 这样就能保正得到的数不是负数
    iw = (torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
          - torch.max(boxes[:, :, 0], query_boxes[:, :, 0]) + 1)
    # 如果小于0，证明两个框没有交集，则等于零
    iw[iw < 0] = 0

    # 同样，找到y坐标的最小最大值，然后得到h的最大值
    ih = ( torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
           - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])+ 1)
    # 如果小于0，证明两个框没有交集，则等于零
    ih[ih < 0] = 0

    # 这样根据交并比的公式，ua计算了两个区域的面积总和
    # 用iw*ih得到相交区域的面积，除以总面积，得到交并比，即IOU
    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    # 返回最大的IOU
    return overlaps


#==============================================
# 函数输入：batch_size形式的gt，anchors可以是batch也可以不是
# 函数输出：batch形式的ROIs, 维度为batch_size, N, K
#==============================================
# 这个函数还是计算IOU，但是gt可以是batch_size的形式
def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    # 这里多的一维是这个gt物体的类别
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    # 首先计算出batch_size的大小
    batch_size = gt_boxes.size(0)

    # 这里判断一下anchors是不是batch的形式，如果是两维证明不是batch的形式
    if anchors.dim() == 2:

        # N和K还是看一张图有多少个anchor和ground_truth
        N = anchors.size(0)
        K = gt_boxes.size(1)

        # 这里的contiguous相当于对原来的tensor进行了一下深拷贝，其实就相当于reshape
        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()

        # 这里计算出gt的宽和高，并且同时计算出面积，将其面积变成(batch, 1, k)的维度
        gt_boxes_x = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        gt_boxes_y = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        # 同样，这里计算出anchors的宽和高，也计算出面积，维度是(batch, N, 1)
        anchors_boxes_x = anchors[:, :, 2] - anchors[:, :, 0] + 1
        anchors_boxes_y = anchors[:, :, 3] - anchors[:, :, 1] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        # 这里判断了一下gt和anchors是不是0，但是我很奇怪为什么这里要判断一下他们是不是0
        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        # 这里就和上面的代码一样了，将两个扩充到一样的维度
        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        # 计算宽度
        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2])
              - torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0])+ 1)
        iw[iw < 0] = 0
        # 计算高度
        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3])
              - torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1])+ 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # 这里是做的一个填补，如果gt是0，让这部分的IOU变成0
        # 如果anchors是0，那么就用-1去替换 IOU的值
        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K)
                              .expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1)
                              .expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        # anchor的最后一维如果是4那么就是四个坐标
        if anchors.size(2) == 4:
            anchors = anchors[:, :, :4].contiguous()
        # 不然的或就是后四个数是坐标，现在还没理解第一维是什么
        else:
            anchors = anchors[:, :, 1:5].contiguous()
        # 得到GT的前思维坐标
        gt_boxes = gt_boxes[:, :, :4].contiguous()

        # 后面的计算过程和前面的基本上没有差别了
        gt_boxes_x = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        gt_boxes_y = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = anchors[:, :, 2] - anchors[:, :, 0] + 1
        anchors_boxes_y = anchors[:, :, 3] - anchors[:, :, 1] + 1
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2])
              - torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3])
              - torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_( gt_area_zero.view(batch_size, 1, K)
                               .expand(batch_size, N, K), 0)
        overlaps.masked_fill_( anchors_area_zero.view(batch_size, N, 1)
                               .expand(batch_size, N, K), -1)
    else:
        raise ValueError("anchors input dimension is not correct.")

    return overlaps
