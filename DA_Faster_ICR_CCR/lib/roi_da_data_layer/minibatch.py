# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import, division, print_function

# from scipy.misc import imread
import cv2
import numpy as np
import numpy.random as npr
from model.utils.blob import im_list_to_blob, prep_im_for_blob
from model.utils.config import cfg

# roidb：一个列表包含[一张图片的roi的字典]
def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    # 有几张图片的字典,根据输入只有一张 -> 文件名
    num_images = len(roidb)   # num_images=1

    # Sample random scales to use for each image in this batch
    # 产生np随机数组,大小是num_images(这里是1),从0到len-1,(cfg.TRAIN.SCALES)是图片短边的像素是个元组
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)   # （600，）

    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        "num_images ({}) must divide BATCH_SIZE ({})".format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    # 输入某张图片的roidb，及最长边限制，返回图片的np数组,和图片的缩放比
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # blobs添加data属性: 图片np数组
    blobs = {"data": im_blob}

    # blobs添加need_backprop属性:target不需要反传BP  source需要反传BP
    im_name = roidb[0]["image"]  # 图片的路径+名称
    if im_name.find("source_") == -1:  # target domain
        blobs["need_backprop"] = np.zeros((1,), dtype=np.float32)
    else:
        blobs["need_backprop"] = np.ones((1,), dtype=np.float32)

    assert len(im_scales) == 1, "Single batch only"  # ？？？？单批次
    assert len(roidb) == 1, "Single batch only"

    # blobs添加gt_boxes属性  (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:      # 是否使用所有的GroundTruth
        # 返回gt box真实类比标签索引,[0]取第一个维度(一维数组)，仅取索引idx   box_num
        gt_inds = np.where(roidb[0]["gt_classes"] != 0)[0]  # where返回idx、val
    else:
        # 返回gt box中属性gt_overlaps（类别得分大于-1）的真实类比标签索引    box_num
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]["gt_classes"] != 0) &
                           np.all(roidb[0]["gt_overlaps"].toarray() > -1.0, axis=1))[0]
    # 合并gt boxes坐标（缩放后的坐标）  gt_class
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)   # (box_num,5)
    gt_boxes[:, 0:4] = roidb[0]["boxes"][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]["gt_classes"][gt_inds]
    blobs["gt_boxes"] = gt_boxes

    # blobs添加im_info属性  图片长,宽,缩放比
    blobs["im_info"] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    # blobs添加图片的ID序号
    blobs["img_id"] = roidb[0]["img_id"]

    # change gt_classes to one hot
    def gt_classes2cls_lb_onehot(array):    # box的真实类别标签  大小：box_num
        cls_lb = np.zeros((num_classes - 1,), np.float32)
        for i in array:
            cls_lb[i - 1] = 1  # 某个类别在该图片中是否存在目标
        return cls_lb

    # change gt_classes to cross entropy need lb
    # 将gt变成交叉熵需要的lb
    def gt_classes2cls_lb(array):
        cls_lb = np.array([i - 1 for i in set(array)])
        return cls_lb

    # blobs添加cls_lb   某个类别在该图片中是否存在目标，有1，无0
    blobs["cls_lb"] = gt_classes2cls_lb_onehot(roidb[0]["gt_classes"])

    return blobs

# 返回图片的np数组,和缩放比,并进行图片的缩放 和 去均值处理
def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """

    num_images = len(roidb)  # 图片数量,根据输入->1
    processed_ims = []       # 缩放后图片存放的列表
    im_scales = []           # 图片的缩放比存放的列表
    for i in range(num_images):
        im = cv2.imread(roidb[i]["image"])   # 读取字典中image的键值 -> 文件的路径,读取图片
        # cv2.imwrite("2 example.jpg", im)
        if len(im.shape) == 2:      # 如果图像是二维(无色彩信息),
            # 对第三个维度进行扩展(为了程序兼容2维图像)
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        if len(im.shape) == 4:  # 防止混入四通道图片,
            im = im.conver('RGB')

        # flip the channel, since the original one using cv2
        # 使im倒叙(对第三个通道)[i:j:s(步长)]
        im = im[:, :, ::-1]    # im.shape=(1024,2048,3)  通道3上 rgb -> bgr

        # 如果需要反转对第二通道进行倒叙
        if roidb[i]["flipped"]:
            im = im[:, ::-1, :]   # 仅对高度方向的像素点翻转,

        # 获取短边像素
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]    # 600
        #  输入:图片像素矩阵  PIXEL_MEANS:RGB三个通道的均值
        #  MAX_SIZE:长边最大尺寸1000   target_size： 短边最大尺寸
        #  返回缩放后的图片和缩放比，统一按照最短边缩放的
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE)
        # 图片的缩放比存放的列表
        im_scales.append(im_scale)
        # 缩放后图片存放的列表
        processed_ims.append(im)

    # Create a blob to hold the input images
    # 把图片转为blob格式，即转成统一长宽的图片格式，选择最大长和最大宽，最后图片填充不够的用0补充
    # blob格式就是 [图片数量，所有图片的最大长，所有图片的最大宽，RGB3通道 ]
    # 其实这里只有一个图片，im_list_to_blob好像什么都没做
    blob = im_list_to_blob(processed_ims)
    # 返回图片的np数组,和缩放比
    return blob, im_scales
