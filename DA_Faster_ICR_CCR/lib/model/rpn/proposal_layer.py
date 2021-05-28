from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import yaml

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.utils.config import cfg

from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
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

# 通过将估计的边界框变换应用于一组常规框（称为“锚点”）来输出对象检测候选区域,选出合适的ROIS
class _ProposalLayer(nn.Module):
    # 从特征图中生成anchor
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        # 得到特征步长
        self._feat_stride = feat_stride
        # 得到基础的anchor,并转化为张量  [9,bs]  左上、右下坐标
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        # anchors的行数就是所有anchor的个数 9
        self._num_anchors = self._anchors.size(0)  # 每个像素点的anchor个数

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)


    # 输入:rpn_cls_prob：[B,nc_score_out,H,W], 每层的像素是fg/bg的概率
    #      rpn_bbox_pred：预测bbox的feature_map(大小:[B,nc_bbox_out,H,W]
    #      im_info：图片的大小H, W, ratio    cfg_key :'TRAIN' or 'TEST'
    def forward(self, input):  # input 含有四个元素的tuple
        # 根据anchor得到候选区域，这里有NMS
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs

        # 读入数据
        scores = input[0][:, self._num_anchors :, :, :]  # [B,9,H,W], 分类概率
        bbox_deltas = input[1]    # [B,nc_bbox_out36,H,W]，预测bbox的偏移量
        im_info = input[2]        # 图像信息：H, W, ratio
        cfg_key = input[3]        # 'TRAIN' or 'TEST'
        #设置参数                                         # TRAIN :   TEST
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N    # 12000 :   6000  NMS前保留的proposal
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N  # 2000  :   300   NMS后保留的proposal
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH         # 0.7   :   0.7
        min_size = cfg[cfg_key].RPN_MIN_SIZE             # 8     :   16   推荐框的最小尺寸

        batch_size = bbox_deltas.size(0)    # 批尺寸

        # 在原图上生成anchor
        feat_height, feat_width = scores.size(2), scores.size(3)
        # 特征图相对于原图的偏移，在原图上生成anchor
        shift_x = np.arange(0, feat_width) * self._feat_stride  #shape: [width,]
        shift_y = np.arange(0, feat_height) * self._feat_stride  #shape: [height,]
        # 生成网格 shift_x shape: [height, width], shift_y shape: [height, width]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(),
                                             shift_y.ravel(),
                                             shift_x.ravel(),
                                             shift_y.ravel())).transpose()) #shape[height*width, 4]
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors  # anchor个数  9
        K = shifts.size(0)   # 网格坐标点的多少 2775=H*W

        self._anchors = self._anchors.type_as(scores)  # 改变数据类型
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        # 得到单张图上的anchors，三维[K, A, 4]
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)   # [K, A, 4]
        # 拓展至一个batch上的anchors，三维[4, K*A, 4]
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4) # [1,K*A,4]->[B,K*A,4]

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        # 转置和重塑预测的bbox转换，使与锚点的顺序相同：  [B,nc_bbox_out36,H,W]->[B,K*A,4]
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()  # [B,H,W,nc_bbox_out36]
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)  # [B,H*W*9,4]=[B,K*A,4]  左上、右下坐标

        # Same story for the scores:使与锚点的顺序相同
        scores = scores.permute(0, 2, 3, 1).contiguous()  # [B,9,H,W]-> [B,H,W,9]
        scores = scores.view(batch_size, -1)  # [B,K*A]

        # Convert anchors into proposals via bbox transformations
        # 结合RPN的输出,通过bbox转换将锚点anchors转换为候选区域
        # 输入：anchors：[B,K*A,4] bbox_deltas:[B,K*A,4]
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        # 输出：[B,K*A,4]  左上、右下坐标

        # 2. clip predicted boxes to image
        # 裁剪预测框到图像
        # 严格限制proposal的四个角在图像边界内
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)
        # 按照评分对前景图像进行排序，选择K个最高的送入NMS网络，将定义好的前N个NMS的输出保存下来，再计算出不符合规定的高度和宽度大小的边框
        scores_keep = scores   # 分类概率 [B,K*A]
        proposals_keep = proposals  # 加完偏移量的anchors [B,K*A,4]
        # scores_keep按照列递减排序,返回value、key
        _, order = torch.sort(scores_keep, 1, True)

        # 定义一个全0矩阵,存放topN个得分最高的  [bs,2000,5]
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        output_da = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            # 删除高度或宽度<阈值的预测框（注意：将min_size转换为存储在im_info [2]中的输入图像比例）

            # 获取了一张图像的proposals、前景的评分
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # 按分数从最高到最低排序所有（h候选区域，得分）对
            # 一张图像的所有proposals前景评分从大到小的排名
            order_single = order[i]

            # # 5. take top pre_nms_topN (e.g. 6000)
            # 根据得分排序，取顶部pre_nms_topN
            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]  # 索引值

            # 按照order_single中的index，取出对应proposals、scores的value
            proposals_single = proposals_single[order_single, :]   # 推荐框 [12000,4]
            scores_single = scores_single[order_single].view(-1, 1)   # 得分 [12000,1]

            # 6. apply nms (e.g. threshold = 0.7)
            # 经过nms的操作得到这张图像保留下来的proposal  nms_thresh=0.7
            # NMS就是去除冗余的检测框,保留最好的一个,nms返回的是索引值
            keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)  # 2770
            keep_idx_i = keep_idx_i.long().view(-1)

            # keep_idx_i_da = nms(proposals_single, scores_single.squeeze(1), 0.5)
            # keep_idx_i_da = keep_idx_i_da.long().view(-1)
            # if post_nms_topN > 0:
            #     keep_idx_i_da = keep_idx_i_da[:post_nms_topN]
            # proposals_single_da = proposals_single[keep_idx_i_da, :]

            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            # 取到的是经过nms保留下来的proposals以及他们的分数
            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            # 存储单张图片的proposal
            output[i, :, 0] = i  # 之前是0填充，现在是对应的第几张图片填充
            output[i, :num_proposal, 1:] = proposals_single

            # num_proposal_da = proposals_single_da.size(0)
            # output_da[i, :, 0] = i
            # output_da[i, :num_proposal_da, 1:] = proposals_single_da
        # 输出:  output  ->  size([bs, num_proposal, 5])  第0维度上，代表的是第几张图片？
        return (output, scores_single)      # 源域和目标域的保留的num_proposal个数不同,以源域为例子做注释

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""

    # 删除任何小于min_size的边框
    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        # expand_as(ws) 将tensor扩展为参数ws的大小
        keep = (ws >= min_size.view(-1, 1).expand_as(ws)) \
               & (hs >= min_size.view(-1, 1).expand_as(hs))
        return keep
