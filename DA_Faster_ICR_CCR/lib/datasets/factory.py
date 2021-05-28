# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
    Factory method for easily getting imdbs by name.
    能够更简单的用名字找到数据集, 里面设置了对多种数据集的预处理
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape
from datasets.clipart import clipart
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.KITTI import KITTI
from datasets.bdd100k import bdd100k
from datasets.sim10k import sim10k
from datasets.vg import vg
from datasets.water import water

__sets = {}

# Set up cityscapes
for year in ["2007", "2012"]:
    for split in ["train_s", "train_t", "train_all", "test_s", "test_t", "test_all"]:
        name = "cityscape_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscape(split, year)
# Set up cityscapes_car
for split in ['train', 'trainval', 'val', 'test']:
    name = 'cityscape_car_{}'.format(split)
    __sets[name] = (lambda split=split : cityscape_car(split))
# Set up foggy_cityscapes
for split in ['train', 'trainval','test']:
    name = 'foggy_cityscape_{}'.format(split)
    __sets[name] = (lambda split=split : foggy_cityscape(split))

# Set up voc_<year>_<split>
for year in ["2007", "2012"]:  # pascal_voc
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc(split, year)
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))
for year in ["2007", "2012"]:  # pascal_voc_water
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_water_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_water(split, year)

# Set up SIM10K
for split in ['train','val']:
    name = 'sim10k_{}'.format(split)
    __sets[name] = (lambda split=split : sim10k(split))

# Set up KITTI
for split in ['train', 'val', 'synthCity', 'trainval']:
  name = 'KITTI_{}'.format(split)
  __sets[name] = (lambda split=split, year=year: KITTI(split))

# Set up bdd100k
for split in ['train', 'val', 'daytrain', 'dayval', 'nighttrain', 'nightval', 'citydaytrain', 'citydayval',
              'cleardaytrain', 'cleardayval', 'rainydaytrain', 'rainydayval']:
    name = 'bdd100k_{}'.format(split)
    __sets[name] = (lambda split=split, year=year: bdd100k(split))

# Set up clipart
for year in ["2007"]:
    for split in ["trainval", "train", "test"]:
        name = "clipart_{}".format(split)
        __sets[name] = lambda split=split: clipart(split, year)

# Set up water
for year in ["2007"]:
    for split in ["train", "test"]:
        name = "water_{}".format(split)
        __sets[name] = lambda split=split: water(split, year)

# # Set up sim10k coco style and cityscapes coco style
# for year in ["2019"]:
#     for split in ["train", "val"]:
#         name = "sim10k_{}_{}".format(year, split)
#         __sets[name] = lambda split=split, year=year: sim10k(split, year)

# Set up coco_2014_<split>
for year in ["2014"]:
    for split in ["train", "val", "minival", "valminusminival", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2014_cap_<split>
for year in ["2014"]:
    for split in ["train", "val", "capval", "valminuscapval", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2015_<split>
for year in ["2015"]:
    for split in ["test", "test-dev"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ["150-50-20", "150-50-50", "500-150-80", "750-250-150", "1750-700-450", "1600-400-20",]:
    for split in ["minitrain", "smalltrain", "train", "minival", "smallval", "val", "test",]:
        name = "vg_{}_{}".format(version, split)
        __sets[name] = lambda split=split, version=version: vg(version, split)

# set up imagenet.
for split in ["train", "val", "val1", "val2", "test"]:
    name = "imagenet_{}".format(split)
    devkit_path = "data/imagenet/ILSVRC/devkit"
    data_path = "data/imagenet/ILSVRC"
    __sets[name] = lambda split=split, devkit_path=devkit_path, data_path=data_path: \
        imagenet(split, devkit_path, data_path)


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()  # 返回imdb数据集，一个函数指针，一个lambda表达式


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
