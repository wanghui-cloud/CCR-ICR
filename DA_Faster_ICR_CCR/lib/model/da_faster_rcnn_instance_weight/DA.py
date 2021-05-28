from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.da_faster_rcnn_instance_weight.LabelResizeLayer import (
    ImageLabelResizeLayer,
    InstanceLabelResizeLayer,)
from model.utils.config import cfg
from torch.autograd import Function, Variable

# 定义了三个类 GRLayer、_ImageDA、_InstanceDA
class GRLayer(Function):
    # 这里的ctx，其实就是self
    @staticmethod
    def forward(ctx, input):
        # 设定一个参数，input保持不变的传输
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        # 梯度反向，并乘上系数
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    # 新样式，不用先实例化    !!参考 https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845
    return GRLayer.apply(x)

# 图像级对齐
class _ImageDA(nn.Module):
    def __init__(self, dim):
        super(_ImageDA, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=True)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
        self.reLu = nn.ReLU(inplace=False)
        self.LabelResizeLayer = ImageLabelResizeLayer()

    # x -> size([1,512,H,W])    FeatureMap
    def forward(self, x, need_backprop):
        # 梯度反转
        x = grad_reverse(x)
        # 两层卷积 维度:512 -> 512 -> 2
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)   # [1, 2, H, W]
        # 根据图片数量生成label数组
        label = self.LabelResizeLayer(x, need_backprop)
        return x, label

# 实例级对齐
class _InstanceDA(nn.Module):
    def __init__(self, in_channel=4096):
        super(_InstanceDA, self).__init__()
        self.dc_ip1 = nn.Linear(in_channel, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(1024, 1024)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer = nn.Linear(1024, 1)
        self.LabelResizeLayer = InstanceLabelResizeLayer()

    def forward(self, x, need_backprop):
        # x -> size([256,4096])
        x = grad_reverse(x)
        # 3层全连接 维度:4096 -> 1024 -> 1024 -> 1
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = torch.sigmoid(self.clssifer(x))
        # x -> size([256,1])
        label = self.LabelResizeLayer(x, need_backprop)
        # label -> size([256,1])
        return x, label
