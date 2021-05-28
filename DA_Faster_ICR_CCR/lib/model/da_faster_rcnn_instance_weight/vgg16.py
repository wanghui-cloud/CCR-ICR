# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------

# 统一python3的执行代码（使用python2运行也是python3的结果）
from __future__ import absolute_import, division, print_function

# pytorch必备的数据库-标准库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# 依赖的外部库，本文件只是一个特征提取模型，要嵌入到_fasterRCNN这个大模型中
from model.da_faster_rcnn_instance_weight.faster_rcnn import _fasterRCNN

# 继承父类_fasterRCNN，父类的构造函数定义了rpn，ROIPooling层
class vgg16(_fasterRCNN):
    # 输入： 数据集类名称元组（背景＋类别名称）、是否使用预训练模型、预训练模型地址、是否采用class_agnostic
    def __init__(self, classes, pretrained_path, pretrained=False, class_agnostic=False):
        self.model_path = "./data/pretrained_model/vgg16_caffe.pth"    # 预训练模型地址
        self.dout_base_model = 512           # 输入特征映射的channel
        self.pretrained = pretrained         # 是否使用预训练模型
        self.class_agnostic = class_agnostic   # 类无关

        _fasterRCNN.__init__(self, classes, class_agnostic)

    # 初始化模型参数，有预训练，则导入参数。
    def _init_modules(self):
        vgg = models.vgg16()   # 调用了标准库的vgg16
        # VGG分2大块，特征提取部分vgg.features，分类器部分vgg.classifier
        if self.pretrained:    # 仅加载了backbone=VGG16的预训练参数
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            # 加载模型参数
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        # 每个网络的分类任务不同，需要定制个性化
        # 重新定义分类器部分，左闭右开，取出vgg.classifier，舍弃最后一个linear
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        # ._modules取出网络转为字典显示    .values() 将字典的对应键值都取出，就是各个小层的网络参数都取出。
        # list() 将这些网络层参数强制转为list类型  [:-1])遍历网络小层，舍弃最后一个小层结构
        # list前加*，把list分解成独立的参数传入

        # 特征提取器部分，取出vgg.features，舍弃最后一个池化层 not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        # 此时的特征图是输入图片的1/16
        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        # Fix the layers before conv3:   VGG又按照表格分块为conv1、conv2、conv3、conv4、conv5
        # 固定特征提取器CNN_base  conv3之前的网络参数，不进行训练
        for layer in range(10):   # 固定0-9
            for p in self.RCNN_base[layer].parameters():
                p.requires_grad = False

        # 分类回归层定义，回归出5数据，分类+坐标值
        self.RCNN_top = vgg.classifier

        # 由于丢弃了最后的一层，重新定义分类预测及回归预测 not using the last maxpool layer
        # 分类预测层
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)   # k
        # 回归预测层
        if self.class_agnostic:   # false 控制bbox的回归方式，是否类无关，与之对应的是class_specific
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
            # agnostic就是不管啥类别，把bbox调整到有东西(类别非0)即可
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)  # 4K
            # specific的话，必须要调整到确定的class

    # 定义进入全连接前，把特征图拉直的函数
    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)  # [1024, 25088]
        fc7 = self.RCNN_top(pool5_flat)             # [1024, 4096]

        return fc7
