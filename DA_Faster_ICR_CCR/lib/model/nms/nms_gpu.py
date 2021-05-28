from __future__ import absolute_import

import numpy as np
import torch

from ._ext import nms

# 方法3：先将全部变量定义为静态类型，再利用Cython模块编译
# 方法4：在方法3的基础上再加入cuda加速模块， 再利用Cython模块编译，即利用gpu加速
def nms_gpu(dets, thresh):
    keep = dets.new(dets.size(0), 1).zero_().int()
    num_out = dets.new(1).zero_().int()
    nms.nms_cuda(keep, dets, num_out, thresh)
    keep = keep[: num_out[0]]
    return keep
