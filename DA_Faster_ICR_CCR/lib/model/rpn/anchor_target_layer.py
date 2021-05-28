from __future__ import absolute_import

import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from model.utils.config import cfg

from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch, clip_boxes
from .generate_anchors import generate_anchors

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------


DEBUG = False

try:
    long  # Python 2
except NameError:
    long = int  # Python 3

# 为anchor找到训练所需的ground truth、类别和坐标变换信息
class _AnchorTargetLayer(nn.Module):
    # 与generate_anchors配合使用，共同产生anchors的样本rpn，用于rpn的分类和回归任务
    # 将anchor对应ground truth。生成anchor分类标签和边界框回归目标。为anchor找到训练所需的ground truth类别和坐标变换信息
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    """
            给所有的anchors赋对应的gt目标，制造anchor二分类的labels和bbox的回归用的targets
            targets 包括:dx dy dw dh
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride  # 16,vgg下采样倍数
        self._scales = scales
        anchor_scales = scales
        # 得到基础的anchor,并转化为张量  [9,bs]
        # !这里从np转为了float_tensor，方便运用torch中函数
        # 传入的scales和ratios是元组形式，需要np.array转换
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        # 不允许超过边界
        self._allowed_border = 0  # default is 0

    # 输入:rpn_cls_prob：[B,nc_score_out18,H,W], 每层的像素是fg/bg的概率   gt_boxes:[bs, 数量, 5]
    #     im_info：[bs,3] 图片的大小H, W, ratio    num_boxes:图片中包含的目标的个数
    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0]    # [bs,18,H,W], 分类概率
        gt_boxes = input[1]    # [bs,数量,5] bbox的标注信息5= 上、右下坐标4+类别1
        im_info = input[2]     # [bs,3]图片的大小H, W, ratio
        num_boxes = input[3]   # 图片中包含的目标的个数

        # map of shape (..., H, W) 得到图片的宽度、高度  一个bs一致
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        # 在原图上生成anchor
        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        # 偏移位置:将anchors在特征图上进行滑动的偏移量
        # 特征图相对于原图的偏移，在原图上生成anchor
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        # 生成网格 shift_x shape: [height, width], shift_y shape: [height, width]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(),
                                             shift_y.ravel(),
                                             shift_x.ravel(),
                                             shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors  # anchor个数  9
        K = shifts.size(0)      # 网格坐标点的多少 2775=H*W

        self._anchors = self._anchors.type_as(gt_boxes)  # move to specific gpu.改变数据类型
        # 得到单张图上的anchors，三维[K, A, 4]
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # 拓展至一个batch上的anchors，三维[K*A, 4]
        all_anchors = all_anchors.view(K * A, 4)  # 左上、右下坐标

        total_anchors = int(K * A)   # total_anchors记录anchor的数目

        # keep only inside anchors
        # 没有过界的anchors索引   xmin>=0 ymin>=0  xmax<w+0 ymax<h+0
        keep = ( (all_anchors[:, 0] >= -self._allowed_border)&
                 (all_anchors[:, 1] >= -self._allowed_border)&
                 (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border)&     # 右下角坐标<width
                 (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))     # 右下角坐标<height
        # inds_inside：没有过界的anchors索引
        inds_inside = torch.nonzero(keep).view(-1)
        # anchors：没有过界的anchors
        anchors = all_anchors[inds_inside, :]    # 基本删去一半

        # 通过overlaps制作labels，找到anchor对应的gt最大索引，和gt对应的anchor最大索引
        # iou>0.7, label: 1 is positive, iou<0.3, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)  # 全-1的矩阵
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        # 计算anchors和gt_boxes的IOU（overlap），返回: [bs，len(anchors), len(gt_boxes)]
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        # 每个anchor对应最大的那个IOU值、索引，返回:[bs，len(anchors)]
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)

        # 每个gt对应最大的IOU值，返回:[bs，len(gt_boxes)]
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:    # False
            #  iou<0.3，负样本标记为0
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # 如果gt_max_overlaps是0,则让他等于一个很小的值，证明gt没有anchor与其重叠
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5

        # keep:每个gt对应最大的IOU值所在位置处，返回:[bs，len(anchors)]
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
        # eq相等返回1，不相等返回0，缩减的维度：按照dim=2相加

        if torch.sum(keep) > 0:
            # 找出与gt相交最大且iou不为0的那个anchor，作为正样本
            labels[keep > 0] = 1

        # fg label: above threshold IOU
        # iou>0.7，正样本标记为1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # 计算出一个训练batch中需要的前景的数量
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)

        # 得到目前已经确定的前景和背景的数量
        sum_fg = torch.sum((labels == 1).int(), 1)    # 前景的anchors数量
        sum_bg = torch.sum((labels == 0).int(), 1)    # 背景的anchors数量

        # 这里对一个batch_size进行迭代，看看选择的前景和背景数量是够符合规定要求
        for i in range(batch_size):
            # 如果得到的正样本太多，则需要二次采样
            # subsample positive labels if we have too many
            # 如果正样本的数量超过了预期的设置
            if sum_fg[i] > num_fg:
                # 首先获取所有的非零元素的索引
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)    # 前景的anchors
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                # 然后将他们用随机数的方式进行排列
                rand_num = (torch.from_numpy(np.random.permutation(fg_inds.size(0))).
                            type_as(gt_boxes).long())
                # 这里就去前num_fg个作为正样本，其他的设置成-1也就是不关心
                disable_inds = fg_inds[rand_num[: fg_inds.size(0) - num_fg]]
                labels[i][disable_inds] = -1

            #           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            # 如果得到的负样本太多，也要进行二次采样
            # 下面就是和上面一样的方法，对越界的那些样本设置为-1
            if sum_bg[i] > num_bg:   #如果事实存在的背景anchor大于了所需值，就随机抛弃一些背景anchor
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = (torch.from_numpy(np.random.permutation(bg_inds.size(0))).
                            type_as(gt_boxes).long())
                disable_inds = bg_inds[rand_num[: bg_inds.size(0) - num_bg]]
                labels[i][disable_inds] = -1

        # 假设每个batch_size的gt_boxes都是20的话  [0,20,40,...,(batch_size-1)*20]
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)

        # argmax_overlaps：[bs，len(anchors)]每个anchor对应最大IOU的索引，加上20，大小不变
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        # 计算anchors与对应最大iou的GT之间的偏移量
        # gt_boxes处理: 1、.view(-1, 5) ->(batch_size*20, 5)
        #               2、[argmax_overlaps.view(-1), :]选出与每个anchorIOU最大的GT,这个时候offset就起作用了
        bbox_targets = _compute_targets_batch(anchors,     # [len(anchors)，4]
                                              gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5),)
        # 返回:[bs,len(anchors)，4]  anchor与他最近的gt的回归四要素tx,ty,tw,th

        # use a single value instead of 4 values for easy index.
        # 所有前景的anchors，将他们的权重初始化  [1.0, 1.0, 1.0, 1.0]   [len(anchors),4]
        bbox_inside_weights[labels == 1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        # 参数默认定义的是-1，如果小于零，positive和negative的权重设置成相同的,1/num_example
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)  # 正样本、负样本之和
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)

        bbox_outside_weights[labels == 1] = positive_weights    # 正样本
        bbox_outside_weights[labels == 0] = negative_weights    # 负样本

        # 输入： total_anchors：记录anchor的数目  inds_inside：没有过界的anchors索引
        #       labels：[bs,len(inds_inside)] 包含0.1,-1，代表正负样本分类

        # 之前取labels的操作都是在对于图像范围内的边框进行的，需要将图像外填充
        # labels和bbox_targets是区域的分类和回归标签
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        # bbox_targets：anchor与他最近的gt的回归四要素tx,ty,tw,th
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        # 计算损失时的权重
        # 正样本回归loss的权重，默认为1，负样本为0，表明在回归任务中，只采用正样本进行计算
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        # 平衡正负样本的权重，它们将在计算SmoothL1Loss的时候被使用
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        # labels转换为[bs,1,9*H,W]
        labels = (labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous())
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        # bbox_targets ->[bs,4*A,h,w]
        bbox_targets = (bbox_targets.view(batch_size, height, width, A * 4).permute(0, 3, 1, 2).contiguous())
        outputs.append(bbox_targets)

        # bbox_inside_weights也转换为4维 [bs,4*A,h,w]
        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).\
            expand(batch_size, anchors_count, 4)
        bbox_inside_weights = (bbox_inside_weights.contiguous()
                               .view(batch_size, height, width, 4 * A).permute(0, 3, 1, 2).contiguous())
        outputs.append(bbox_inside_weights)

        # bbox_outside_weights也转换为4维 [bs,4*A,h,w]
        bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1)\
            .expand(batch_size, anchors_count, 4)
        bbox_outside_weights = (bbox_outside_weights.contiguous().view(batch_size, height, width, 4 * A)
                                .permute(0, 3, 1, 2).contiguous())
        outputs.append(bbox_outside_weights)

        return outputs
        # 输出: 返回output列表[labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
        #      labels： [bs,1,9*H,W]: 标签(0:背景, 1: 前景, -1: 屏蔽)
        #      bbox_targets: [bs,9*4,H,W] 每个anchor与IOU最大的GTbbox的位移和缩放参数
        #      bbox_inside_weights:[bs,9*4,H,W] in权重,label是1，它就是1。其他为0，只就算正样本的损失
        #      bbox_outside_weights:[bs,9*4,H,W] out权重 样本权重归一化后正负样本都是1/256.其他背景为0,计算SmoothL1Loss的时候被使用

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""

# _unmap函数将图像内部的anchor映射回到生成的所有的anchor
def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret

# _compute_targets函数计算anchor和对应的gt_box的位置映射
def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
