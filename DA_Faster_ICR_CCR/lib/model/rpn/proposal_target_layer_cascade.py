from __future__ import absolute_import

import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn

from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

# 将对象检测候选分配给ground truth目标。生成候选分类标签和边界框回归目标。
# 为选出的ROIS找到训练所需的ground truth类别和坐标变换信息
class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)  # 均值
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)    # 方差
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)    # 权重

    # 输入:all_rois：[bs,num_proposal,5]，RPN生成的候选区域
    #                经过fg/bg预测+nms后的proposal,num_proposal<=2000(目标域是300)
    #                5=[第一个元素恒定为0/1/2/3/4,x1,y1,x2,y2],产生的RoI都是正样本
    #     gt_boxes:[bs,num_boxes,5],真实边框,5=左上、右下坐标4+类别1   num_boxes:目标框的数量
    def forward(self, all_rois, gt_boxes, num_boxes):
        # 作用: 再次对roi进行筛选(到256个vgg16.yml中设定) 2000 proposal —> 128
        #   1、roi对应的GT标签(之前的步骤只有fg,bg,这里得到的是class)
        #   2、roi的GT变化量(之后就是要通过这个做回归)  3、得到权重
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        # gt_boxes_append 全0矩阵 [bs,num_boxes,5]
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        # 最后一个维度，第一个元素保持为0,和all_rois保持一致,仅赋值了1:5
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]

        # Include ground-truth boxes in the set of candidate rois
        # 拼接all_rois[bs,num_proposal,5](前景框) + gt_boxes_append[bs,num_boxes,5](gt框)
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)
        # 返回：all_rois (bs,num_proposal+num_boxes,5)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)  # 256
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))  # 0.25=64
        # 每张图片至少一个roi
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        # 生成包含前景和背景的roi的随机样本示例
        # 输入:all_rois:[bs,num_proposal+num_boxes,5]、gt_boxes:[bs,num_boxes,5] 左上、右下坐标4+类别1
        #     fg_rois_per_image:每张图片的前景roi(64)、rois_per_image:每张图片的roi(256) _num_classes:9
        labels, rois, bbox_targets, bbox_inside_weights \
            = self._sample_rois_pytorch(all_rois, gt_boxes, fg_rois_per_image,
                                        rois_per_image, self._num_classes)
        # 输出:labels：[bs,256],正样本对应的标签,负样本均设置为0
        #      rois：  [bs,256,5] 记录预测正负ROI，其值来自于RPN回归分支输出,最后一维,前1:batch编号,后4:坐标
        # bbox_targets:[bs,256,4] 正负ROI对应的偏移量,2个平移变化量，两个缩放变化量，仅设置了前景部分，背景为0
        # bbox_inside_weights:[bs,256,4],存在有真实物体对应ROI的回归权重，最后一维度均为:(1,1,1,1),背景部分设置为0

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
        # 输出: rois：  [bs,256,5] 记录预测正负ROI，其值来自于RPN回归分支输出,最后一维,前1:batch编号,后4:坐标
        #      labels：[bs,256],正样本对应的标签,负样本均设置为0
        #      bbox_targets：[bs,256,4] 正负ROI对应的偏移量,2个平移变化量，两个缩放变化量，仅设置了前景部分，背景为0
        #      bbox_inside_weights：[bs,256,4],存在有真实物体对应ROI的回归权重，最后一维度均为:(1,1,1,1),背景部分设置为0
        #      bbox_outside_weights：[bs,256,4],存在有真实物体对应ROI的权重，和上面相等？？？


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""

    # 将背景类对应的ROI偏移量置0、记录对应有真实物体对应ROI的权重
    def _get_bbox_regression_labels_pytorch( self, bbox_target_data,  # [bs,256,4] 两个平移变化量，两个缩放变化量
                                             labels_batch,            # [bs,256] proposals对应的标签
                                             num_classes):            # 类别数量 9
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)    # 批处理大小
        rois_per_image = labels_batch.size(1)   # 每张图片的ROI数量
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()  # [bs,256,4]
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()     # [bs,256,4]

        # 遍历每一张图片的标签
        for b in range(batch_size):
            # 如果一张图片全是背景，则跳过
            if clss[b].sum() == 0:
                continue
            # 返回一张图片存在物体的索引
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):   # numel()函数：返回数组中元素的个数
                ind = inds[i]
                # 仅设置了前景部分，背景部分默认为0
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                # 为前景的对应box,设置BBOX_INSIDE_WEIGHTS为偏移量权重
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS
            # 其余未设置部分为背景，默认为0

        return bbox_targets, bbox_inside_weights
        # 输出：
        #    # bbox_targets          -> bbox_target_data的部分    -> 仅设置了前景部分，背景部分设置为0
        #    # bbox_inside_weights   -> size([1,256,4])     ->  最后一维度:(1.0, 1.0, 1.0, 1.0)

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""
        '''
           计算每一张图片中ROI的偏移量.
		   ex_rois： 预测的ROI，shape(batch, 256, 4)
		   gt_rois: 对应ROI的真实边框， shape(batch, 256, 4)
        '''
        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)        # 批处理大小
        rois_per_image = ex_rois.size(1)    # 每张图片的ROI数目

        # targets -> size([1,256,4]) -> 两个平移变化量，两个缩放变化量
        # 计算ROI与真实边框的偏移量，[batch, 256, 4]
        targets = bbox_transform_batch(ex_rois, gt_rois)

        # 是否对偏移量进行标准化操作（减均值除标准差操作）
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = (targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))\
                      / self.BBOX_NORMALIZE_STDS.expand_as(targets)

        return targets

    # 生成包含前景和背景的roi的随机样本示例
    def _sample_rois_pytorch(self, all_rois,      # [bs,num_proposal+num_boxes,5] 真实边框和候选框在通道上拼接起来
                             gt_boxes,            # [bs,num_boxes,5] 左上、右下坐标4+类别1  真实边框
                             fg_rois_per_image,   # 每张图片的前景roi(64)
                             rois_per_image,      # 每张图片的roi(256)
                             num_classes):        # 类别数量 9
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        # 利用Proposal与标签生成IoU矩阵
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)  # [bs,num_proposal+num_boxes,num_boxes]

        # 每个rois对应的最大gt_bbox的IOU值及索引
        max_overlaps, gt_assignment = torch.max(overlaps, 2) # [bs,num_proposal+num_boxes]

        batch_size = overlaps.size(0)         # 批处理大小 bs
        num_proposal = overlaps.size(1)       # 候选框数目 num_proposal+num_boxes
        num_boxes_per_img = overlaps.size(2)  # 真实边框数目 num_boxes

        # offset是对IoU矩阵的每一行的最大值索引加上其所在的batch的编号*每个batch的真实边框数（即K）
        # 假设num_boxes=20, offset = [0,20,40,...,(batch_size-1)*20]
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        # offset bs-> (bs,1) + (bs,2050) -> (bs,2050)  广播操作
        # 每个rois对应的最大交并比的gt_bbox的标号   [bs,num_proposal+num_boxes]
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        # 根据交并比把gt_bbox的cls的标签分配给每个proposal
        # 取出IoU矩阵中每一行对应着有最大IoU的真实边框的类别标签，shape(batch, 2000+K)
        labels = (gt_boxes[:, :, 4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1))
        #  labels:[bs,num_proposal+num_boxes] 每个proposal的labels

        # 记录标签，其值来自于真实边框的类别标签
        labels_batch = labels.new(batch_size, rois_per_image).zero_()  # [bs,256]
        # 记录预测ROI，其值来自于RPN回归分支输出
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()  # [bs,256,5]
        # 记录对应的真实边框，其值来自于真实边框
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()  # [bs,256,5]

        # 选择满足条件的正负样本
        for i in range(batch_size):
            # 前景索引,IoU>=0.5的框
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()    # numel()函数：返回数组中元素的个数

            # 背景索引:IoU在[0.1，0.5]的框
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI)&
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            # 既有前景fg又有背景bg, fg多于设定值64，则进行下采样随机选取
            if fg_num_rois > 0 and bg_num_rois > 0:
                # 对正样本进行采样 64
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_num_rois).long().cuda()
                # 把前景进行随机排列
                rand_num = (torch.from_numpy(np.random.permutation(fg_num_rois))
                            .type_as(gt_boxes).long())
                # 最多取前64个
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # 对负样本进行采样,根据fg个数计算bg个数
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                # rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)

                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:       # 全是前景fg
                # 采样256个正样本
                # rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                # 生成随机数组size([256]) -> [1~fg_num_rois(2050)]
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                # 随机抽取256个，可能重复
                fg_inds = fg_inds[rand_num]
                # 256个全是前景
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:   # 全是背景bg
                # 采样256个负样本
                # sampling bg
                # rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            # 将正负样本索引拼接在一起   [256],在for i in batch_size循环里
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            # 取出正负样本Proposal对应的标签     [bs,256]
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            # 前景框数量 < 总框数 => 后面的都是背景 设置为0
            # 保证将背景ROI的标签置为0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            # 记录取出对应的预测ROI,保存要保留的坐标   最后一维 前1:0   后4:坐标
            rois_batch[i] = all_rois[i][keep_inds]
            # i=0,后四位是坐标，第一位都设定为i  rois_batch -> size([bs,256,5])
            rois_batch[i, :, 0] = i

            # gt_assignment[i][keep_inds] -> size([256])    -> 选出的proposal分别对应的gt_box的编号
            # gt_rois_batch -> size([1,256,5])    ->选出的proposal分别对应的gt_box, 最后一维度前4:坐标 后1:类别
            # 记录取出对应的真实边框    最后一维 前4:坐标 后1:类别
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        # 计算每一个Proposal相对于其标签的偏移量，[bs,256,4]
        bbox_target_data = self._compute_targets_pytorch(rois_batch[:, :, 1:5],
                                                         gt_rois_batch[:, :, :4])
        # 输出：bbox_target_data:[1,256,4]  两个平移变化量，两个缩放变化量

        # 将背景类对应的ROI偏移量置0、记录对应有真实物体对应ROI的权重
        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data,
                                                     labels_batch, num_classes)
        # 输出： bbox_targets：[bs,256,4] 仅设置了前景部分，背景部分设置为0，同下
        #       bbox_inside_weights:[bs,256,4],ROI的权重，最后一维度均为:(1.0, 1.0, 1.0, 1.0)

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
        # 输出:labels_batch：[bs,256],正负样本Proposal对应的标签,负样本均设置为0
        #      rois_batch：[bs,256,5] 记录预测ROI，其值来自于RPN回归分支输出,最后一维 前1:i + 后4:坐标
        #      bbox_targets：[bs,256,4] 2个平移变化量，两个缩放变化量，仅设置了前景部分，背景部分设置为0，同下
        #      bbox_inside_weights:[bs,256,4],ROI的权重，最后一维度均为:(1,1,1,1),背景部分设置为0