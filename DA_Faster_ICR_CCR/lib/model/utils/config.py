from __future__ import absolute_import, division, print_function

import os
import os.path as osp

import numpy as np

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()   # EasyDict的实例化,可以使得以属性的方式去访问字典的值
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate  初始学习率
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum  动量
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization   正则化DATA_DIR
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate  LR的降低因子
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# 显示配置 Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
# 是否为了偏误而将学习速率加倍
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution
# 是否初始化截断正态分布的权值
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
# 采样区域时是否在池中添加GT boxs
__C.TRAIN.USE_GT = False

# Whether to use aspect-ratio grouping of training images, introduced merely for saving
# GPU memory   是否使用长宽比分组训练图像
__C.TRAIN.ASPECT_GROUPING = False  #一个batch中选择尺度相似的样本

# The number of snapshots kept, older ones are deleted to save space
# 保留的快照数量，旧的删除，以节省空间
__C.TRAIN.SNAPSHOT_KEPT = 3

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)  #训练尺度，可以配置为一个数组

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000 #缩放后图像最长边的上限

# Trim size for input images to create minibatch
__C.TRAIN.TRIM_HEIGHT = 600
__C.TRAIN.TRIM_WIDTH = 600

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1  #每一个batch使用的图像数量

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128  #每一个batch中前景的比例

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5  #ROI前景阈值

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5   #ROI背景高阈值
__C.TRAIN.BG_THRESH_LO = 0.1  #ROI背景低阈值

# Use horizontally-flipped images during training?
# 数据增广,图像翻转
__C.TRAIN.USE_FLIPPED = True  #训练时是否进行水平翻转

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True  #是否训练回归

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5  #用于训练回归的roi与真值box的重叠阈值

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000   #snapshot间隔

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = "res101_faster_rcnn"
# __C.TRAIN.SNAPSHOT_INFIX = ''    #snapshot前缀

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
# __C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True   #bbox归一化方法，去均值和方差
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)    #rpn 前景box权重

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = "gt"    #默认proposal方法

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True  #是否使用RPN
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7  #正样本IoU阈值
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3  #负样本IoU阈值
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5  #前景样本的比例
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256   #RPN样本数量
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7    #NMS阈值
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000   #使用NMS前，要保留的top scores的box数量
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000   #使用NMS后，要保留的top scores的box数量
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 8    #原始图像空间中的proposal最小尺寸阈值
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

# Whether to tune the batch normalization parameters during training
__C.TRAIN.BN_TRAIN = False

#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = "gt"

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = "nms"

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000

#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize.
# if true, the region will be resized to a square of 2xPOOLING_SIZE,
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

#
# MobileNet options
#

__C.MOBILENET = edict()

# Whether to regularize the depth-wise filters during training
__C.MOBILENET.REGU_DEPTH = False

# Number of fixed layers during training, by default the first of all 14 layers is fixed
# Range: 0 (none) to 12 (all)
__C.MOBILENET.FIXED_LAYERS = 5

# Weight decay for the mobilenet weights
__C.MOBILENET.WEIGHT_DECAY = 0.00004

# Depth multiplier
__C.MOBILENET.DEPTH_MULTIPLIER = 1.0

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1.0 / 16.0

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# 对所有的网络使用相同的像素均值，尽管它并不是真正的像素均值
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, "data/datasets"))

# Name (or path to) the matlab executable
__C.MATLAB = "matlab"

# Place outputs under an experiments directory
__C.EXP_DIR = "default"

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0

__C.POOLING_MODE = "crop"

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Maximal number of gt rois in an image during Training
# 训练过程中图像gt roi的最大值
__C.MAX_NUM_GT_BOXES = 20

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8, 16, 32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5, 1, 2]

# Feature stride for RPN
__C.FEAT_STRIDE = [16]

__C.CUDA = False

__C.CROP_RESIZE_WITH_MAX_POOL = True


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, "output", __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = "default"
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, "tensorboard", __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = "default"
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(("Type mismatch ({} vs. {}) " "for config key: {}")
                                 .format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(("Error under config key: {}".format(k)))
                raise
        else:
            b[k] = v

# 载入设置文件
def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    # 用来解析.yml数据结构文件
    import yaml

    with open(filename, "r") as f:
        # 以属性的方式访问字典
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    # 融合两个设置文件
    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval

    assert len(cfg_list) % 2 == 0   # 确保以字典对的形式存在
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split(".")
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
            # print(d)
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), "type {} does not match original type {}"\
            .format(type(value), type(d[subkey]))
        # type <class 'tuple'> does not match original type <class 'list'>
        d[subkey] = value


# if __name__ == "__main__":
#     set_cfgs = [ "ANCHOR_RATIOS", "[0.5,1,2]",  "TRAIN.SCALES",  "[800,]",
#                  "TRAIN.MAX_SIZE", "1600", "TEST.SCALES",  "[800,]", "TEST.MAX_SIZE", "1600",]
#
#     cfg_from_list(set_cfgs)
