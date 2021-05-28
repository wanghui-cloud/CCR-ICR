from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
from model.utils.net_utils import _smooth_l1_loss
from torch.autograd import Variable

from .anchor_target_layer import _AnchorTargetLayer
from .proposal_layer import _ProposalLayer


class _RPN(nn.Module):
    """ region proposal network """

    def __init__(self, din):
        super(_RPN, self).__init__()

        # 得到输入特征图的深度
        self.din = din  # get depth of input feature map, e.g., 512
        # anchor的尺度
        self.anchor_scales = cfg.ANCHOR_SCALES   # [8, 16, 32]
        # anchor的比例
        self.anchor_ratios = cfg.ANCHOR_RATIOS   # [0.5, 1, 2]
        # 特征步长 1个像素对应原图的大小
        self.feat_stride = cfg.FEAT_STRIDE[0]   # [16],vgg下采样倍数

        # 处理base_feat的convrelu层 input_channel=512
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # RPN目标分数预测层，背景和前景分类得分
        # define bg/fg classifcation score layer
        self.nc_score_out = (len(self.anchor_scales)
                             * len(self.anchor_ratios) * 2)  # 19
        # 上面是RPN卷积 这里是分类， 网络输入是512 输出是参数个数
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # RPN目标边界框offset参数预测层
        # define anchor box offset prediction layer
        self.nc_bbox_out = (len(self.anchor_scales)
                            * len(self.anchor_ratios) * 4)  # 36
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        # 定义候选区域层，对bbox进行分类, 维度3*3*3=27
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        # 定义anchor目标层，实例化 proposallayer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod #静态方法
    # 将x reshape
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0], int(d), int(float(input_shape[1] * input_shape[2])
                                               / float(d)), input_shape[3],)
        return x

    # 输入: base_feat: [1,512,H,W] 基础特征图      im_info: [1,3]  W, H, ratio
    #      gt_boxes:[bs,num_boxes,5],5=左上、右下坐标4+类别1   num_boxes:目标框的数量
    def forward(self, base_feat, im_info, gt_boxes, num_boxes, need_cls_score=False):

        # base_feat是四维 [Batch_size, 512, W, H]
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        # RPN_Conv经过RPN卷积（conv3X3），输出：[Batch_size, 512, H, W]
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        # 接下来，特征图rpn_conv1经过1、RPN目标分数预测层  2、RPN目标边界框offset参数预测层

        # get rpn classification score
        # 1、RPN目标分数预测层（conv1X1），得到RPN分类得分
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # [bs,18,H,w]
        # 输出：[B,nc_score_out,H,W]，并更改形状->[bs,2,H*nc_score_out/2,W]
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)  # [bs,2,9*H,w]
        # dim=1上softmax, 得到两类fg/bg的概率，再恢复形状[bs,nc_score_out,H,W]
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)   # [bs,2,9*H,w]
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out) # [bs,18,H,w]

        # 2、RPN目标边界框offset参数预测层，进行bbox的预测，4个参数的偏移
        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)  # [bs,nc_bbox_out,H,W] [bs,36,H,W]

        # proposal layer
        # 3、进行roi的预测
        cfg_key = "TRAIN" if self.training else "TEST"

        # 用anchor提取候选区域
        # 输入:rpn_cls_prob：[B,nc_score_out18,H,W], 每层的像素是fg/bg的概率
        #      rpn_bbox_pred：[B,nc_bbox_out36,H,W] 预测bbox的feature_map  im_info：图片的大小H, W, ratio
        (rois, rois_binary_cls_score)= self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                                          im_info, cfg_key))
        # 输出:rois：[bs,num_proposal,5]，经过fg/bg预测+nms后的proposal,num_proposal<=2000(目标域是300)
        #                               5=[第一个元素恒定为0/1/2/3/4,x1,y1,x2,y2],产生的RoI都是正样本
        #     rois_binary_cls_score:[num_proposal,1]为对应proposal的前景得分，num_proposal为指定的nms参数值

        self.rpn_loss_cls = 0  # 分类损失
        self.rpn_loss_box = 0  # 回归损失

        # generating training labels and build the rpn loss
        # 生成训练标签并构建rpn损失，源域需要, 目标域跳过
        if self.training:
            assert gt_boxes is not None  # 如果 gt_boxes 不存在就警告

            # 为anchor找到训练所需的ground truth、类别和坐标变换信息
            # 输入:rpn_cls_prob：[B,nc_score_out18,H,W], 每层的像素是fg/bg的概率   gt_boxes:[1, 数量, 5]
            #     im_info：[1,3] 图片的大小H, W, ratio    num_boxes:图片中包含的目标的个数
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            # 输出: 返回rpn_data列表,包含[labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
            #      labels： [bs,1,9*H,W]: 标签(0:背景, 1: 前景, -1: 屏蔽)
            #      bbox_targets: [bs,9*4,H,W] anchor与他最近的gt的回归四要素tx,ty,tw,th
            #      bbox_inside_weights:[bs,9*4,H,W] in权重,label是1，它就是1。其他为0，只就算正样本的损失
            #      bbox_outside_weights:[bs,9*4,H,W] out权重，正负样本都是1/256.其他背景为0,计算SmoothL1Loss的时候被使用

            # 计算分类损失 compute classification loss
            # 前景背景分类得分 [bs,2,9*H,w] -> [bs,9*H,W,2] -> [bs,9*H*W,2]  未经softmax的
            rpn_cls_score = (rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2))

            # 取出rpn_data的label,-> [bs, 9*H*W]
            rpn_label = rpn_data[0].view(batch_size, -1)

            # 找到rpn_label中的非屏蔽项(≠-1)的索引号index,包含所有的计算损失时的正负样本 [256*bs]
            # ne去除掉-1，返回非0的索引,并展平
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            # 前背景分类得分-> [9*H*W, 2] ,index_select 从第0轴按照rpn_keep索引找
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            # 前景背景label -> [9*H*W] 对第0维度 寻找需要匹配的anchor   [256*bs]
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            # 变成变量(新版本中就不用了)
            rpn_label = Variable(rpn_label.long())
            # 计算交叉损失(分类损失)
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            # 计算前景的个数
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            # 取出[bbox坐标变化, in权重, out权重]
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, = rpn_data[1:]

            # compute bbox regression loss 都变成变量(新版本中就不用了)
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            # 计算L1loss(回归损失)
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets,
                                                rpn_bbox_inside_weights, rpn_bbox_outside_weights,
                                                sigma=3, dim=[1, 2, 3],)

        if need_cls_score:
            return rois, self.rpn_loss_cls, self.rpn_loss_box, rois_binary_cls_score
        else:
            # 返回roi,分类和回归的损失
            return rois, self.rpn_loss_cls, self.rpn_loss_box
        # 输出:rois：[bs,num_proposal,5]，经过fg/bg预测+nms后的proposal,num_proposal<=2000(目标域是300)
        #                                 最后一维5 [第一个元素恒定为0/1/2/3/4, x1, y1, x2, y2]
        #      rpn_loss_cls：分类损失   rpn_loss_box：回归损失
