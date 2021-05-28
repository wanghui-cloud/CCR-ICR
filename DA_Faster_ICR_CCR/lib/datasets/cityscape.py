from __future__ import absolute_import, print_function

import os
import pickle  # 持久化存储模块
import subprocess  # 以不同的方式创建子进程
import uuid   # 通用唯一识别码
import xml.etree.ElementTree as ET # 元素树对xml文件解析

# import PIL
import numpy as np
import scipy.io as sio
import scipy.sparse  # 稀疏矩阵

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

from . import ds_utils
from .imdb import ROOT_DIR, imdb
from .voc_eval import voc_eval

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete

 # 继承imdb
class cityscape(imdb):
    # def __init__(self, image_set, year, devkit_path=None):
    def __init__(self, image_set, year, devkit_path=None):
        # 使用父类的初始化方法，仅给出数据集名称
        imdb.__init__(self, "cityscape_" + year + "_" + image_set)
        self._year = year
        self._image_set = image_set
        # 数据集路径 VOC格式
        self._devkit_path = (self._get_default_path() if devkit_path is None else devkit_path)
        self._data_path = os.path.join(self._devkit_path, "VOCdevkit")
        # cityscape大类有8个+1个背景
        self._classes = ( "__background__",  # always index 0
                          "person", "rider", "car", "truck", "bus",
                          "train", "motorcycle", "bicycle",)
        # 给每一个类别分类赋予一个对应的整数
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ".jpg"
        # 把所有的图片名称加载，放在list中，便于索引读取图片
        # 把对应域和训练测试的txt文件中的图片名，->导入self._image_index列表中,每个元素代表一张图片的名称，无后缀
        self._image_index = ( self._load_image_set_index() )  # train image name without .jpg

        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        # 根据地址打开标注文件(Annoation),并以字典的形式导入缓存文件中,并返回gt_roidb字典
        self._roidb_handler = self.gt_roidb   #  dict 调用gt_roidb，返回gt_roidb字典
        # 定义一个随机的标识符
        self._salt = str(uuid.uuid4())        # 生成当前版本号,每次不同->作用??
        self._comp_id = "comp4"

        # PASCAL specific config options    一些不知道是什么的配置????
        self.config = { "cleanup": True, "use_salt": True, "use_diff": False, "matlab_eval": False,
                        "rpn_file": None, "min_size": 2,}

        assert os.path.exists(self._devkit_path), "VOCdevkit path does not exist: {}".format(self._devkit_path)
        assert os.path.exists(self._data_path), "Path does not exist: {}".format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, "JPEGImages", index + self._image_ext)

        assert os.path.exists(image_path), "Path does not exist: {}".format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        把对应域和训练测试的txt文件中的图片名，->导入self._image_index列表中,每个元素代表一张图片
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt   数据集列表
        image_set_file = os.path.join(self._data_path, "ImageSets", "Main", self._image_set + ".txt")
        assert os.path.exists(image_set_file), "Path does not exist: {}".format(image_set_file)

        image_index = []
        with open(image_set_file) as f:
            for x in f.readlines():
                if len(x) > 1:   # 为去除一些空行
                    image_index.append(x.strip())
            # image_index = [x.strip() for x in f.readlines()]

        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, "cityscape")

    # 根据地址打开标注文件(Annoation),并以字典的形式导入缓存文件中,并返回gt_roidb字典
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        # 创建imdb数据集对象时，创建的文件夹
        cache_file = os.path.join(self.cache_path, self.name + "_gt_roidb.pkl")
        # 继承imdb,初始化时，仅会调用init函数，此处调用self.cache_path才会去新建文件夹
        if os.path.exists(cache_file):    # 如果已经有缓存pkl文件，则跳过，第一次时未跳过
            # "rb"   以二进制读方式打开，只能读文件 ， 如果文件不存在，会发生异常
            with open(cache_file, "rb") as fid:
                # 将二进制对象转换成 Python 对象
                roidb = pickle.load(fid)
            print("{} gt roidb loaded from {}".format(self.name, cache_file))
            return roidb  # 退出程序

        # 第一次执行无缓存文件，故执行以下的代码，缓存数据
        # 把对应域和训练测试的txt文件中的图片名，->导入self._image_index列表中,每个元素代表一张图片的名称，无后缀
        gt_roidb = [self._load_pascal_annotation(index) for index in self.image_index]  # 返回self._image_index
        with open(cache_file, "wb") as fid:
            # "wb" 以二进制写方式打开，只能写文件，如果文件不存在，创建该文件；如果文件已存在，先清空
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            # 序列化对象，将对象obj保存到文件file中去 pickle.dump(obj, file, [,protocol])
        print("缓存数据：wrote gt roidb to {} ".format(cache_file))
        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + "_selective_search_roidb.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fidf:
                roidb = pickle.load(fid)
            print("{} ss roidb loaded from {}".format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != "test":
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, "wb") as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print("wrote ss roidb to {}".format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != "test":
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config["rpn_file"]
        print("loading {}".format(filename))
        assert os.path.exists(filename), "rpn data not found at: {}".format(filename)
        with open(filename, "rb") as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR, "selective_search_data", self.name + ".mat"))
        assert os.path.exists(filename), "Selective search data not found at: {}".format(filename)
        raw_data = sio.loadmat(filename)["boxes"].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config["min_size"])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC format.
        以PASCAL VOC格式从XML文件中加载图像和边框信息。
        index :list 一张无后缀的图片名称，
        """
        filename = os.path.join(self._data_path, "Annotations", index + ".xml")
        tree = ET.parse(filename)   # 对xml解析（元素树）
        # 在元素树中找 'object' ，object就是图片中被框出的物体
        objs = tree.findall("object")
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)   # 判断 物体个数
        # 建立空np数组（num_objs*4）->存放GT目标框用
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        # 建立空np数组（num_objs）->存放GT类别
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # 建立空np数组（num_objs（目标数量）*num_classes（类别数量））->所有obj(目标)在不同class(类别)的置信度
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        # np数组->box框的面积
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        # np数组->目标是否是难目标
        ishards = np.zeros((num_objs), dtype=np.int32)

        # 对每个目标(被框出的物体)分别进行读取,
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):  # enumerate枚举:返回序号,元素
            bbox = obj.find("bndbox")    # 找<object>标签下的<bndbox>标签
            # Make pixel indexes 0-based
            x1 = float(bbox.find("xmin").text) - 1  # 坐标从零开始所以要减去
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1

            diffc = obj.find("difficult")    # 找<object>标签下的<difficult>标签
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult    # 写入判断难易样本的数组

            # <name>标签下的类别信息全部小写,除去空格
            # 通过self._class_to_ind字典找到类别对应的编号,赋值给cls(cls是个数值临时变量)
            cls = self._class_to_ind[obj.find("name").text.lower().strip()]  # 1-8之间的数值
            boxes[ix, :] = [x1, y1, x2, y2]    # 对每个obj目标,写入GT的boxes数组
            if boxes[ix, 0] > 2048 or boxes[ix, 1] > 1024:  # 最小值超越边界检测  必要性??
                print(boxes[ix, :])
                print(filename)
                p = input()

            gt_classes[ix] = cls   # GT标签赋值
            overlaps[ix, cls] = 1.0   # 目标对每个class的置信度初始化,GT是1,其他是0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)  # GT的box的面积(像素)

        overlaps = scipy.sparse.csr_matrix(overlaps)   # 使用scipy.sparse的稀疏矩阵csr_matrix()
        # 返回读到的数据
        return { "boxes": boxes, "gt_classes": gt_classes, "gt_ishard": ishards,
                 "gt_overlaps": overlaps, "flipped": False,"seg_areas": seg_areas,}

    def _get_comp_id(self):
        comp_id = (self._comp_id + "_" + self._salt if self.config["use_salt"]
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + "_det_" + self._image_set + "_{:s}.txt"
        filedir = os.path.join(self._devkit_path, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write("{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                            index, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1,
                            dets[k, 2] + 1, dets[k, 3] + 1,))

    def _do_python_eval(self, output_dir="output"):
        annopath = os.path.join(self._devkit_path, "VOC" + self._year, "Annotations", "{:s}.xml")
        imagesetfile = os.path.join(self._devkit_path, "VOC" + self._year, "ImageSets", "Main",
                                    self._image_set + ".txt",)
        cachedir = os.path.join(self._devkit_path, "annotations_cache")
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print("VOC07 metric? " + ("Yes" if use_07_metric else "No"))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == "__background__":
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls,
                                     cachedir, ovthresh=0.5, use_07_metric=use_07_metric,)
            aps += [ap]
            print("AP for {} = {:.4f}".format(cls, ap))
            with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        print("Mean AP = {:.4f}".format(np.mean(aps)))
        print("~~~~~~~~")
        print("Results:")
        for ap in aps:
            print("{:.3f}".format(ap))
        print("{:.3f}".format(np.mean(aps)))
        print("~~~~~~~~")
        print("")
        print("--------------------------------------------------------------")
        print("Results computed with the **unofficial** Python eval code.")
        print("Results should be very close to the official MATLAB eval code.")
        print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
        print("-- Thanks, The Management")
        print("--------------------------------------------------------------")

    def _do_matlab_eval(self, output_dir="output"):
        print("-----------------------------------------------------")
        print("Computing results with the official MATLAB eval code.")
        print("-----------------------------------------------------")
        path = os.path.join(cfg.ROOT_DIR, "lib", "datasets", "VOCdevkit-matlab-wrapper")
        cmd = "cd {} && ".format(path)
        cmd += "{:s} -nodisplay -nodesktop ".format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += "voc_eval('{:s}','{:s}','{:s}','{:s}'); quit;\"".format(
            self._devkit_path, self._get_comp_id(), self._image_set, output_dir)
        print("Running:\n{}".format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config["matlab_eval"]:
            self._do_matlab_eval(output_dir)
        if self.config["cleanup"]:
            for cls in self._classes:
                if cls == "__background__":
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config["use_salt"] = False
            self.config["cleanup"] = False
        else:
            self.config["use_salt"] = True
            self.config["cleanup"] = True
