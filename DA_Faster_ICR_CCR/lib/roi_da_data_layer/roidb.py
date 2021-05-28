"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import, division, print_function

import datasets
import numpy as np
import PIL
from datasets.factory import get_imdb
from model.utils.config import cfg

# 丰富imdb.roidb
def prepare_roidb(imdb):
    """通过添加一些对训练有用的派生量来丰富imdb的roidb。
    此函数预先计算每个ROI和每个GT框之间在地面真值框上获得的最大重叠量。
    还记录了具有最大重叠的类。    增加height...标签"""
    # roidb列表:
    # boxes:box位置信息，box_num*4的np array  左上、右下坐标
    # gt_classes:所有box的真实类别    gt_overlaps:所有box在不同类别的得分，box_num*class_num矩阵
    # flipped:False,不翻转    gt_ishard:判断难易样本  seg_areas:box的面积

    roidb = imdb.roidb
    # 如果不是coco数据集的话  startswith() 方法用于检查字符串是否是以指定子字符串开头
    if ( not (imdb.name.startswith("coco")) or "car" in imdb.name or "sim10k" in imdb.name):   # 不执行
        # 打开每一张图片获取图片大小
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size for i in range(imdb.num_images)]
    # 对全部的图片进行遍历  self._image_index列表中,每个元素代表一张图片的名称，无后缀
    for i in range(len(imdb.image_index)):
        # 对第i张图片,ID和路径,长度和宽度,对roidb中的字典进行添加
        roidb[i]["img_id"] = imdb.image_id_at(i)   # 返回ID=i
        roidb[i]["image"] = imdb.image_path_at(i)  # 返回图片路径
        if (not (imdb.name.startswith("coco")) or "car" in imdb.name or "sim10k" in imdb.name):
            # 加入图片的宽度、高度
            roidb[i]["width"] = sizes[i][0]
            roidb[i]["height"] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        # 每个目标对类的置信度
        gt_overlaps = roidb[i]["gt_overlaps"].toarray() # toArray()返回 一个新的数组对象，修改不改变原来

        # max overlap with gt over classes (columns)
        # 每个目标(行)最大的置信度
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        # 每个目标(行)最大的置信度的那类的索引值
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]["max_classes"] = max_classes
        roidb[i]["max_overlaps"] = max_overlaps

        # 合理性检查
        # max overlap of 0 => class should be zero (background)
        # np.where(max_overlaps == 0)-->找到数组中0的位置,返回二维数组,组成坐标,[0]只取第一维
        zero_inds = np.where(max_overlaps == 0)[0]    # 列出有置信度存在0的行(目标)
        assert all(max_classes[zero_inds] == 0)       # 在置信度为0的中间，类别是背景的找出背景
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)     #在置信度不为0的中间，找出前景


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    # 设置长宽比的极限
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]["width"]   # 取出第 i 个图像的字典 -> 'width'键值对应的矩阵
        height = roidb[i]["height"]
        ratio = width / float(height)   # 计算每个框的长宽比 -> 得到一张图片的长宽比矩阵

        if ratio > ratio_large:
            roidb[i]["need_crop"] = 1    # 是否需要裁剪
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]["need_crop"] = 1
            ratio = ratio_small
        else:
            roidb[i]["need_crop"] = 0    # 超过极限的进行裁剪标记,添加到字典中
        # 把修改后的长宽比,加入列表[把所有图片的所有框的长宽比放在一张表中]
        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)     # 产生数组，输出ndarray，满足指定要求的数组对象
    ratio_index = np.argsort(ratio_list)   # 进行长宽比排列
    return ratio_list[ratio_index], ratio_index   # 返回排列后的长宽比,和排列顺序，对应图片(从小到大)


def filter_roidb(roidb):
    # filter the image without bounding box.
    # print("before filtering： %d images" % (len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]["boxes"]) == 0:   # 过滤没有目标框的图片
            del roidb[i]
            i -= 1
        i += 1

    print("after filtering： %d images" % (len(roidb)))
    return roidb

# imdb_name="cityscape_2007_train_s" 融合多个数据集的db
def combined_roidb(imdb_names, training=True):  # dataset name

    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        # 如果使用翻转,数据增广 2975张 -> 5950张
        if cfg.TRAIN.USE_FLIPPED:  # Ture
            print("Appending horizontally-flipped training examples:")
            # 水平翻转数据增强,翻转boxes坐标
            # 仅处理了roidb标注信息部分,加了一倍图片的roidb,编号为对应的2倍
            imdb.append_flipped_images()  # 仅处理了roidb标注信息部分

        print("Preparing training data:")

        prepare_roidb(imdb)  # 丰富imdb.roidb
        # ratio_index = rank_roidb_ratio(imdb)

        return imdb.roidb

    # 获取roidb和imdb格式的训练数据
    def get_roidb(imdb_name):
        # 建立imdb对象：数据集标注的实例化对象
        imdb = get_imdb(imdb_name)   # factory.py,通过名称指定数据集对象imdb = cityscape(train_s, 2007)
        # 包含 boxes, gt_classes, ishards,  overlaps, flipped, seg_areas,
        print("Loaded dataset `{:s}` for training".format(imdb.name))
        # 设置产生proposla的方法  导致 imdb.roidb_handler = self.gt_roidb
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print("Set proposal method: {:s}".format(cfg.TRAIN.PROPOSAL_METHOD)) # gt
        # 处理imdb中的roidb，水平翻转操作 2975张 -> 5950张
        roidb = get_training_roidb(imdb)  # 此时才建立翻转数据增强后的注释文件的缓存,
        return roidb

    # 对字符串进行分割，有的数据集中是多个数据集名用‘+’相连，先分开处理。一个数据集，没有变化
    # 比如：voc_2007_trainval+voc_2012_trainval
    roidbs = [get_roidb(s) for s in imdb_names.split("+")]  # cityscape_2007_train_s
    # 最终返回GT的roidbs,形式[ 第一种数据集->[{ 第一张图片的字典 },{ 第二张图片的字典 },{...}],第二种数据集-> [{},...],[...]]

    roidb = roidbs[0]   # 这里因为只有一个数据集,即cityscape_2007_train_s
    if len(roidbs) > 1:   # 不会执行，指定有二个数据集合并时，才会执行
        # r是每个roidb列表
        for r in roidbs[1:]:
            # 在第一个数据集的列表中追加(后面数据集的图片标注)字典
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split("+")[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes) # 数据集合并
    else:
        imdb = get_imdb(imdb_names)  #  数据集 一个cityscape对象，cityscape(split, year)

    if training:
        # 过滤没有目标框的图片    !!对cityscape : 5950张 -> 5930张
        roidb = filter_roidb(roidb)  # filter samples without bbox
        # print(len(roidb))

    #  进行长宽比的排列,排列后的长宽比列表ratio_list & 长宽比的次序ratio_index
    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return (imdb, roidb, ratio_list, ratio_index,)
    # imdb  dataset
    # roidb dict   由imdb得到丰富后的imdb中roidb
    # ratio_list(0.5,0.5,0.5......2,2,2,)   长宽比列表r
    # ratio_increase_index(4518,6421,.....)   长宽比的次序 升序
