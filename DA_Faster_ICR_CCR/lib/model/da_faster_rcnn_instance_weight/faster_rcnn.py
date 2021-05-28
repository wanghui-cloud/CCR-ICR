import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.da_faster_rcnn_instance_weight.DA import _ImageDA, _InstanceDA
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import ( _affine_grid_gen, _affine_theta, _crop_pool_layer, _smooth_l1_loss,)
from torch.autograd import Variable


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic, in_channel=4096): # class-agnostic 方式只回归2类bounding box,即前景和背景
        super(_fasterRCNN, self).__init__()
        self.classes = classes  #类别
        self.n_classes = len(classes)   # 类别数9
        self.class_agnostic = class_agnostic   # 类无关 前景背景类

        #  loss 两种loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)  # dout_base_model=512 输入特征映射的channel ????
        # 为选择出的rois找到训练所需的ground truth类别和坐标变换信息  proposal_layer 选择出合适的rois
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)  # 候选区域对应gt

        # 以下是3种裁剪特征图的方式
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign( (cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        self.grid_size = (cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE)
        # self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA(in_channel)
        self.consistency_loss = torch.nn.MSELoss(reduction='sum')
        # Conv2d(input_channel、output_channel=8、kernel_size=1X1、stride=1、padding=0)
        self.conv_lst = nn.Conv2d(self.dout_base_model, self.n_classes - 1, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    # 输入：源域及目标域的：
    # im_data:图片np数据、im_info:图片的长、宽、缩放比、im_cls_lb:某个类别在该图片中是否存在目标
    # gt_boxes:bbox的标注信息 weight_value：图片级、目标级权重
    def forward(self, im_data, im_info, im_cls_lb, gt_boxes, num_boxes, need_backprop,    # 源域
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop, # 目标域
                weight_value=1.0,):   # 权重因子

        # 合理性检查   源域需要BP,目标域不需要BP时正常
        # if not (need_backprop.detach() == 1 and tgt_need_backprop.detach() == 0):
        # # if not (need_backprop.detach() == (1,1,1,1) and tgt_need_backprop.detach() == (0,0,0,0)):
        #     need_backprop = torch.Tensor([1]).cuda()
        #     tgt_need_backprop = torch.Tensor([0]).cuda()
        #
        # assert need_backprop.detach() == 1 and tgt_need_backprop.detach() == 0

        #============================= for source ==========================

        batch_size = im_data.size(0)  # 图片个数  batch_size/gpu数目
        im_info = im_info.data  # [4,3]  (size1,size2, image ratio(new image / original image) )
        # im_cls_lb.shape:[batch_size/gpu数目,num_class]
        im_cls_lb = im_cls_lb.data  # gt_classes转化的计算交叉熵需要使用的 某个类别在该图片中是否存在目标，有1，无0
        gt_boxes = gt_boxes.data    # [bs,num_boxes,5] bbox的标注信息 5=左上、右下坐标4+类别1
        num_boxes = num_boxes.data  # 每张图片中目标个数
        need_backprop = need_backprop.data  # 是否需要BP

        # feed image data to base model to obtain base feature map
        # 将图像数据馈送到基础模型以获得基础特征图 [4,512,W/16,H/16]
        base_feat = self.RCNN_base(im_data)   # im_data图片数据

        # 图像分类 判断图片中是否含有某类，输出[bs, n_classes-1],squeeze(-1) 指定维度压缩一个维度
        cls_feat = self.conv_lst(self.avg_pool(base_feat)).squeeze(-1).squeeze(-1)
        # 图像级类别正则化  第一个括号为定义。初始化__init__（），后面为调用forward（）函数
        img_cls_loss = nn.BCEWithLogitsLoss()(cls_feat, im_cls_lb)

        # feed base feature map tp RPN to obtain rois
        # 特征图反馈到RPN得到ROIS
        self.RCNN_rpn.train()   # 训练、测试过程RPN起的作用不一样
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # 输出:rois：[bs,num_proposal,5]，经过fg/bg预测+nms后的proposal,num_proposal<=2000(目标域是300)
        #            5=[第一个元素恒定为0/1/2/3/4,x1,y1,x2,y2],产生的RoI都是正样本
        #     rpn_loss_cls：分类损失   rpn_loss_box：回归损失

        # 训练模式下，用ground truth回归
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            # 输出: rois：  [bs,256,5] 记录预测正负ROI，其值来自于RPN回归分支输出,最后一维,前1:batch编号,后4:坐标
            #      labels： [bs,256],真实框标签，正样本对应的标签,负样本均设置为0
            #      bbox_targets：[bs,256,4] 正负ROI对应的偏移量,2个平移变化量，两个缩放变化量，仅设置了前景部分，背景为0
            # bbox_inside_weights：[bs,256,4],存在有真实物体对应ROI的回归权重，最后一维度均为:(1,1,1,1),背景部分设置为0
            # bbox_outside_weights：[bs,256,4],存在有真实物体对应ROI的权重，和上面相等？？？
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            # 更改形状
            rois_label = Variable(rois_label.view(-1).long())                              # [256*bs] 1024
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))              # [1024,bs]
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))     # [1024,bs]
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))  # [1024,bs]

        else:  # 验证模式下
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        # 根据预选框的位置坐标在特征图中将相应区域池化为固定尺寸的特征图，以便进行后续的分类和bbox回归操作
        if cfg.POOLING_MODE == "align":   # base_feat： [4,512,W/16,H/16]  rois —> [256*bs，4]
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            # 返回 [256*bs, 512, 7, 7]
        elif cfg.POOLING_MODE == "pool":
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        # pooling后的特征反馈到上次模型,   经过了 RCNN_top
        pooled_feat = self._head_to_tail(pooled_feat)    # [256*bs, 4096]

        # 计算bbox的偏移  经过全连接层
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)  # [256*bs,36=4*9]
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            # 根据roi标签选择相应的列  返回：[256*bs,9,4]
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            # torch.gather 收集输入的特定维度指定位置的数值  返回：[256*bs,1,4]
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4),)
            # 去除第二个维度 [256*bs,1,4] -> [256*bs,4] 获得bs*256所对应的回归参数
            bbox_pred = bbox_pred_select.squeeze(1)

        # 计算对象分类概率   返回：[256*bs,9]
        cls_score = self.RCNN_cls_score(pooled_feat)  # 计算分数
        cls_prob = F.softmax(cls_score, 1)     # [256*bs,1] 计算概率

        RCNN_loss_cls = 0   # RCNN分类损失
        RCNN_loss_bbox = 0  # RCNN回归损失

        if self.training:
            # 分类损失  classification loss 交叉熵
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # 计算的是一个batch的所有损失

            # 回归损失  bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        # 预测的类别  [4,256,9]
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        # 预测的边界框的回归参数   [bs,256,4]
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        # ============================ for target ================================"""
        tgt_batch_size = tgt_im_data.size(0)   # 图片个数  batch_size/gpu数目
        tgt_im_info = (tgt_im_info.data)  #  [4,3]  (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data  # [4,5] bbox的标注信息 5=左上、右下坐标4+类别1
        tgt_num_boxes = tgt_num_boxes.data   # 一维4个数字，全为0，每张图片中目标个数
        tgt_need_backprop = tgt_need_backprop.data  # 是否需要BP  一维4个数字，全为0，

        # feed image data to base model to obtain base feature map
        # 将图像数据馈送到基础模型以获得基础特征图 [bs,512,W/16,H/16]
        tgt_base_feat = self.RCNN_base(tgt_im_data)

        # .detach()从当前计算图中分离下来的,不会具有梯度grad
        # 图像分类 判断图片中是否含有某类， 输出[bs, n_classes-1]
        tgt_image_cls_feat = (self.conv_lst(self.avg_pool(tgt_base_feat)).squeeze(-1).squeeze(-1).detach() )
        # 经过一个sigmod激活函数？？？  挖掘目标域实例中的难样本（对实例赋予不同的损失权重）
        tgt_image_cls_feat = torch.sigmoid(tgt_image_cls_feat[0]).detach()

        # feed base feature map tp RPN to obtain rois
        # 特征图反馈到RPN得到ROIS
        self.RCNN_rpn.eval()  # 验证、测试模式
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(tgt_base_feat, tgt_im_info,
                                                                      tgt_gt_boxes, tgt_num_boxes)
        # 输出:rois：[bs,num_proposal,5]，经过fg/bg预测+nms后的proposal,num_proposal<=300(源域是2000)
        #                               5=[第一个元素恒定为0/1/2/3/4,x1,y1,x2,y2],产生的RoI都是正样本
        #           rpn_loss_cls=0   rpn_loss_box=0

        # if it is training phrase, then use ground trubut bboxes for refining

        tgt_rois_label = None
        tgt_rois_target = None
        tgt_rois_inside_ws = None
        tgt_rois_outside_ws = None
        tgt_rpn_loss_cls = 0
        tgt_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == "crop":
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            tgt_grid_xy = _affine_grid_gen(tgt_rois.view(-1, 5),
                                           tgt_base_feat.size()[2:], self.grid_size)
            tgt_grid_yx = torch.stack([tgt_grid_xy.data[:, :, :, 1],
                                       tgt_grid_xy.data[:, :, :, 0]], 3).contiguous()
            tgt_pooled_feat = self.RCNN_roi_crop(tgt_base_feat, Variable(tgt_grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                tgt_pooled_feat = F.max_pool2d(tgt_pooled_feat, 2, 2)
        # 根据预选框的位置坐标在特征图中将相应区域池化为固定尺寸的特征图，以便进行后续的分类和bbox回归操作
        elif cfg.POOLING_MODE == "align": # base_feat： [bs,512,W/16,H/16]  tgt_rois —> [300*bs，4]
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
            # 返回 [300*bs, 512, 7, 7]
        elif cfg.POOLING_MODE == "pool":
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        # feed pooled features to top model
        # pooling后的特征反馈到上次模型,   经过了 RCNN_top
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)   # [300*bs, 4096]

        # 计算对象分类概率   返回：[300*bs,9]
        tgt_cls_score = self.RCNN_cls_score(tgt_pooled_feat).detach()  # [300*bs,9]计算分数
        tgt_prob = F.softmax(tgt_cls_score, 1).detach()      # [300*bs,9] 计算概率
        tgt_pre_label = tgt_prob.argmax(1).detach()   # [300*bs]预测的类别即为最大值

        """  ************** DA source loss ****************  """
        DA_img_loss_cls = 0   # 源域图像级损失
        DA_ins_loss_cls = 0   # 源域实例级损失
        tgt_DA_img_loss_cls = 0  # 目标域图像级损失
        tgt_DA_ins_loss_cls = 0  # 目标域实例级损失

        # 1) 图像级域分类器
        # base_feat： [bs,512,W/16,H/16]
        base_score, base_label = self.RCNN_imageDA(base_feat, need_backprop)
        # 图像级损失函数计算
        base_prob = F.log_softmax(base_score, dim=1)
        DA_img_loss_cls = F.nll_loss(base_prob, base_label)

        # 2) 实例级域分类器
        instance_sigmoid, same_size_label = self.RCNN_instanceDA(pooled_feat, need_backprop)
        # 实例级级损失函数计算
        instance_loss = nn.BCELoss()
        DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

        # 3) 一致性损失函数计算
        # consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
        consistency_prob = F.softmax(base_score, dim=1)[:, 1, :, :]
        consistency_prob = torch.mean(consistency_prob)
        consistency_prob = consistency_prob.repeat(instance_sigmoid.size())

        DA_cst_loss = self.consistency_loss(instance_sigmoid, consistency_prob.detach())

        """  ************** DA taget loss ****************  """
        # 1) 图像级域分类器
        tgt_base_score, tgt_base_label = self.RCNN_imageDA( tgt_base_feat, tgt_need_backprop)
        # 图像级损失函数计算
        tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
        tgt_DA_img_loss_cls = F.nll_loss(tgt_base_prob, tgt_base_label)

        # 2) 实例级域分类器
        tgt_instance_sigmoid, tgt_same_size_label = self.RCNN_instanceDA(tgt_pooled_feat,
                                                                         tgt_need_backprop)
        # 将图像级的多标签结果与实例级的预测结果进行监督，挖掘目标域实例中的难样本（对实例赋予不同的损失权重）
        target_weight = []    # 难样本权重
        for i in range(len(tgt_pre_label)):
            label_i = tgt_pre_label[i].item()
            if label_i > 0:
                diff_value = torch.exp(
                    weight_value * torch.abs(tgt_image_cls_feat[label_i - 1] -
                                             tgt_prob[i][label_i])).item()
                target_weight.append(diff_value)
            else:
                target_weight.append(1.0)
        # 实例级级损失函数计算
        tgt_instance_loss = nn.BCELoss(weight=torch.Tensor(target_weight).view(-1, 1).cuda())
        tgt_DA_ins_loss_cls = tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)

        # 3) 一致性损失函数计算
        tgt_consistency_prob = F.softmax(tgt_base_score, dim=1)[:, 0, :, :]
        tgt_consistency_prob = torch.mean(tgt_consistency_prob)
        tgt_consistency_prob = tgt_consistency_prob.repeat(tgt_instance_sigmoid.size())

        tgt_DA_cst_loss = self.consistency_loss(tgt_instance_sigmoid, tgt_consistency_prob.detach())

        return (rois,         # [bs,256,5] 源域rpn推荐的候选框，其值来自于RPN回归分支输出
                              # 最后一维,前1:batch编号,后4:坐标
                rois_label,   # [256*bs] rpn推荐的候选框标签，正样本对应的标签,负样本均设置为0
                cls_prob,     # [256*bs,1] 类别预测
                bbox_pred,    # [bs,256,4] 预测的边界框的回归参数
                img_cls_loss, # 图像级类别正则化,判断图片中是否含有某类
                # faster RCNN原有损失
                rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox,
                # 源域、目标域图片级域自适应 实例级域自适应  一致性约束损失函数
                DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_img_loss_cls, tgt_DA_ins_loss_cls,
                DA_cst_loss, tgt_DA_cst_loss, )

    def _init_weights(self):
        # 输入：网络名称、均值、方差、是否初始化截断正态分布的权值
        def normal_init(m, mean, stddev, truncated=False):  #均值 标准差
            # 截断正态 随机正态
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # backbone=VGG16，标准库的vgg16，参数加载的预训练参数
        # RCNN_rpn
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)   # 是否初始化截断正态分布的权值
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # RCNN
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        normal_init(self.conv_lst, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # 图片级域自适应
        normal_init(self.RCNN_imageDA.Conv1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_imageDA.Conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # 实例级域适应
        normal_init(self.RCNN_instanceDA.dc_ip1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_instanceDA.dc_ip2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        # 域分类器
        normal_init(self.RCNN_instanceDA.clssifer, 0, 0.05, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()  # 调用初始化模型参数，有预训练，则导入参数，在VGG16类中重写
        self._init_weights()  # 调用初始化参数
