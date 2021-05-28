# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import os
import os.path as osp

# from model.utils.cython_bbox import bbox_overlaps
import numpy as np
import PIL
import scipy.sparse
from model.utils.config import cfg

ROOT_DIR = osp.join(osp.dirname(__file__), "..", "..")

'''
imdb class为所有数据集的父类，包含了所有数据集共有的特性。
eg:数据集名称（_name）、数据集类名列表（_classes）、数据集的文件名列表（_image_index）、_roidb、_config等

roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含到roi信息
以roidb[img_index]为例说明：
    boxes：          box位置信息，box_num*4的np array  左上、右下坐标
    gt_classes：     所有box的真实类别，box_num长度的list
    gt_overlaps：    所有box在不同类别的得分，box_num*class_num矩阵
    filpped：        是否翻转
    max_overlaps：   每个box的在所有类别的得分最大值，box_num长度
    max_classes：    每个box的得分最高所对应的类，box_num长度
'''

class imdb(object):
    """Image database.  数据集对象"""

    def __init__(self, name, classes=None):
        self._name = name  # 数据集名称
        self._num_classes = 0   # 数据集类别个数
        # 如果没有输入类别-->建立空list
        if not classes:
            self._classes = []
        else:
            self._classes = classes    # 数据集类别数量
        # 数据集图片文件名列表 例如 data/VOCdevkit2007/VOC2007/ImageSets/Main/{image_set}.txt
        self._image_index = []
        self._obj_proposer = "gt"   # 候选区域的选取方法
        # 字典 包含了gt_box、真实标签、gt_overlaps和翻转标签 flipped: true,代表图片被水平反转
        self._roidb = None    # 开始时_roidb 设置为 空
        self._roidb_handler = self.default_roidb  # 候选框提取默认采用了selective_search方法
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name  # 返回数据集名称

    @property
    def num_classes(self):
        return len(self._classes)  # 返回数据集类别数量

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index  # 返回数据集图片数量

    @property
    def roidb_handler(self):
        return self._roidb_handler  # 返回ground-truth每个ROI构成的数据集

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    # 设置候选框的方法
    def set_proposal_method(self, method):
        method = eval("self." + method + "_roidb")   #  method = 'gt'
        self.roidb_handler = method   #  self.roidb_handler = self.gt_roidb

    @property
    def roidb(self):
        #   A roidb is a list of dictionaries, each with the following keys:
        #   roidb 是字典的列表，有索引值：boxes  gt_overlaps  gt_classes  flipped
        if self._roidb is not None:
            return self._roidb
        # self._roidb 是 cityscape.py中的gt_roidb,标注信息,从annotation文件中读取
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    # 缓存数据集注释信息
    # cache_path用来生成roidb缓存文件的文件夹，用来存储数据集的roi
    def cache_path(self):
        cache_path = osp.abspath(osp.join("./data/experiments/domain-adaptation", "cache"))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path   # 返回缓存地址 cfg.DATA_DIR+'cache'，不存在就创建

    @property
    def num_images(self):
        return len(self.image_index)

    def image_path_at(self, i):
        # 返回错误，子类没有实现父类要求一定要实现的接口
        raise NotImplementedError

    def image_id_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
        raise NotImplementedError

    def _get_widths(self):  # 返回图像的size[0]，即宽度值
        return [ PIL.Image.open(self.image_path_at(i)).size[0]
                 for i in range(self.num_images) ]

    # 对图像数据进行水平翻转，进行数据增强
    def append_flipped_images(self):

        num_images = self.num_images   # 返回图片数量 2975
        print(" flipped_images：{}" .format(num_images))

        widths = self._get_widths()    # 返回图片宽度，根据图片宽度去变化坐标
        # 遍历每个图片
        for i in range(num_images):
            # 用copy，父对象不会因为dict的改变而改变，而子对象会
            # roidb['boxes']有四个元素，xmin,ymin,xmax,ymax
            boxes = self.roidb[i]["boxes"].copy()
            # 假设boxes=([1,2,4,2]),oldx1=[1],oldx2=[4]
            oldx1 = boxes[:, 0].copy()  #  左上角x坐标
            oldx2 = boxes[:, 2].copy()  # 右下角x坐标
            # 变换坐标，将xmax变成xmin,xmin变成xmax关于x=xmin对称的点,翻转后boxes变成[-2,2,1,4]
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            # modified
            for b in range(len(boxes)):
                if boxes[b][2] < boxes[b][0]:   # 检查，防止自己标注的数据集出错，可能出现x坐标为0的情况
                    boxes[b][0] = 0

            assert (boxes[:, 2] >= boxes[:, 0]).all()  # # 翻转后的xmax肯定大于xmi
            entry = {"boxes": boxes,
                     "gt_overlaps": self.roidb[i]["gt_overlaps"], # 所有box在不同类别的得分，box_num*class_num矩阵
                     "gt_classes": self.roidb[i]["gt_classes"],  # 所有box的真实类别，box_num长度的list
                     "flipped": True,}  # flipped变为True代表水平翻转
            # 将翻转后的图像数据的roidb也加入   翻转的图片的roidb的flipped属性不一样
            self.roidb.append(entry)
        # 因为是按顺序翻转，所有只需要将原来的扩大一倍，roidb里面的图片信息索引与image_index索引对应
        self._image_index = self._image_index * 2 # _image_index指的是每个图片的名称


    # # 根据RP来确定候选框的recall值
    # def evaluate_recall(self, candidate_boxes=None, thresholds=None,
    #                     area='all', limit=None):
    #   """Evaluate detection proposal recall metrics. 评估recall值
    #
    #   Returns:
    #       results: dictionary of results with keys 返回结果是下面4个指标的字典
    #           'ar': average recall
    #           'recalls': vector recalls at each IoU overlap threshold
    #           'thresholds': vector of IoU overlap thresholds
    #           'gt_overlaps': vector of all ground-truth overlaps
    #   """
    #   # Record max overlap value for each gt box
    #   # Return vector of overlap values
    #   # 制定了一些area范围，先根据area找到index，再通过area_ranges[index]找到范围
    #   areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
    #            '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
    #   area_ranges = [[0 ** 2, 1e5 ** 2],  # all
    #                  [0 ** 2, 32 ** 2],  # small
    #                  [32 ** 2, 96 ** 2],  # medium
    #                  [96 ** 2, 1e5 ** 2],  # large
    #                  [96 ** 2, 128 ** 2],  # 96-128
    #                  [128 ** 2, 256 ** 2],  # 128-256
    #                  [256 ** 2, 512 ** 2],  # 256-512
    #                  [512 ** 2, 1e5 ** 2],  # 512-inf
    #                  ]
    #   assert area in areas, 'unknown area range: {}'.format(area)
    #   area_range = area_ranges[areas[area]]
    #   gt_overlaps = np.zeros(0)
    #   num_pos = 0
    # '''
    #  roidb是由字典组成的list，roidb[img_index]包含了该图片索引所包含到roi信息，下面以roidb[img_index]为例说明：
    #     boxes：box位置信息，box_num*4的np array
    #     gt_overlaps：所有box在不同类别的得分，box_num*class_num矩阵
    #     gt_classes：所有box的真实类别，box_num长度的list
    #     filpped：是否翻转
    #     max_overlaps：每个box的在所有类别的得分最大值，box_num长度
    #     max_classes：每个box的得分最高所对应的类，box_num长度
    # '''
    #   for i in range(self.num_images):
    #     # Checking for max_overlaps == 1 avoids including crowd annotations
    #     # (...pretty hacking :/)
    #     max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
    #     gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
    #                        (max_gt_overlaps == 1))[0]   # 首先获得需要评估的index
    #     gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
    #     gt_areas = self.roidb[i]['seg_areas'][gt_inds]
    #     valid_gt_inds = np.where((gt_areas >= area_range[0]) &
    #                              (gt_areas <= area_range[1]))[0]  # 得到符合面积约束的index
    #     gt_boxes = gt_boxes[valid_gt_inds, :]
    #     num_pos += len(valid_gt_inds)  # 记录符合条件的框的个数
    #
    #     if candidate_boxes is None:
    #       # If candidate_boxes is not supplied, the default is to use the
    #       # non-ground-truth boxes from this roidb
    #       non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
    #       boxes = self.roidb[i]['boxes'][non_gt_inds, :]
    #     else:
    #       boxes = candidate_boxes[i]
    #     if boxes.shape[0] == 0:
    #       continue
    #     if limit is not None and boxes.shape[0] > limit:
    #       boxes = boxes[:limit, :]
    #
    #     overlaps = bbox_overlaps(boxes.astype(np.float),
    #                              gt_boxes.astype(np.float))
    #
    #     _gt_overlaps = np.zeros((gt_boxes.shape[0]))
    #     for j in range(gt_boxes.shape[0]):
    #       # find which proposal box maximally covers each gt box
    #       argmax_overlaps = overlaps.argmax(axis=0)
    #       # and get the iou amount of coverage for each gt box
    #       max_overlaps = overlaps.max(axis=0)
    #       # find which gt box is 'best' covered (i.e. 'best' = most iou)
    #       gt_ind = max_overlaps.argmax()
    #       gt_ovr = max_overlaps.max()
    #       assert (gt_ovr >= 0)
    #       # find the proposal box that covers the best covered gt box
    #       box_ind = argmax_overlaps[gt_ind]
    #       # record the iou coverage of this gt box
    #       _gt_overlaps[j] = overlaps[box_ind, gt_ind]
    #       assert (_gt_overlaps[j] == gt_ovr)
    #       # mark the proposal box and the gt box as used
    #       overlaps[box_ind, :] = -1
    #       overlaps[:, gt_ind] = -1
    #     # append recorded iou coverage level
    #     gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))
    #
    #   gt_overlaps = np.sort(gt_overlaps)
    #   if thresholds is None:
    #     step = 0.05
    #     thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    #   recalls = np.zeros_like(thresholds)
    #   # compute recall for each iou threshold
    #   for i, t in enumerate(thresholds):
    #     recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    #   # ar = 2 * np.trapz(recalls, thresholds)
    #   ar = recalls.mean()
    #   return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
    #           'gt_overlaps': gt_overlaps}

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        # box_list的长度必须跟图片的数量相同，相当于为每个图片创造roi，各图像要一一对应
        assert (len(box_list) == self.num_images), \
            "Number of boxes must match number of ground-truth images"
        roidb = []
        for i in range(self.num_images):
            # 遍历每张图片，boxes代表当前图像中的box
            boxes = box_list[i]
            num_boxes = boxes.shape[0]  # 代表当前boxes中box的个数
            # overlaps:所有box在不同类别的得分,shape:num_boxes × num_classes
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]["boxes"].size > 0:
                gt_boxes = gt_roidb[i]["boxes"]
                gt_classes = gt_roidb[i]["gt_classes"]
                gt_overlaps = bbox_overlaps(boxes.astype(np.float), gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append( {"boxes": boxes,
                           "gt_classes": np.zeros((num_boxes,), dtype=np.int32), # 所有box的真实类别
                           "gt_overlaps": overlaps,     # 所有box在不同类别的得分
                           "flipped": False,            #  是否经过翻转
                           "seg_areas": np.zeros((num_boxes,), dtype=np.float32),} )
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]["boxes"] = np.vstack((a[i]["boxes"], b[i]["boxes"]))
            a[i]["gt_classes"] = np.hstack((a[i]["gt_classes"], b[i]["gt_classes"]))
            a[i]["gt_overlaps"] = scipy.sparse.vstack( [a[i]["gt_overlaps"],
                                                        b[i]["gt_overlaps"]] )
            a[i]["seg_areas"] = np.hstack((a[i]["seg_areas"], b[i]["seg_areas"]))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
