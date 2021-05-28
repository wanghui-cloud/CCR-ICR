# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

# from scipy.misc import imread, imresize
import cv2
import numpy as np

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    把图片列表变化为适合网络的输入格式
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    # 取出每张图片的最大的长宽和深度
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # 求出图片的个数
    num_images = len(ims)
    # 创建一个np数组4维,(图片序号,长,宽,深度)(最大的),用for循环填入图片数据
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0 : im.shape[0], 0 : im.shape[1], :] = im
    # 返回图片的np数组
    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)   # .astype是转numpy数据的命令
    # 减去中值 RGB三个通道的均值
    im -= pixel_means
    # im = im[:, :, ::-1]
    # 记录维度
    im_shape = im.shape
    # 取前两个维度的最大值和最小值
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # 得到图像的短边的缩放比     target是短边像素
    im_scale = float(target_size) / float(im_size_min)
    # 先用短边固定尺寸计算缩放比例，如果长边超最大比例，用长边最大除以长边作为缩放比例。
    # Prevent the biggest axis from being more than MAX_SIZE
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    # im = imresize(im, im_scale)
    # 沿x,y轴缩放的系数都是im_scale
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

    return im, im_scale
