# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# 20200309跑通
# 20200505代码阅读完毕
# CUDA_VISIBLE_DEVICES=0,1  在终端执行程序时指定GPU
# 问题:MAP过小  20200519:修改先验框
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pprint
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model.da_faster_rcnn_instance_weight.resnet import resnet
from model.da_faster_rcnn_instance_weight.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import ( adjust_learning_rate, clip_gradient, load_net,
                                    save_checkpoint, save_net, weights_normal_init,)
from roi_da_data_layer.roibatchLoader import roibatchLoader
from roi_da_data_layer.roidb import combined_roidb
from torch.utils.data.sampler import Sampler
from tensorboardX import SummaryWriter

#  未使用
def infinite_data_loader(data_loader):
    while True:
        for batch in data_loader:
            yield batch   # 带yield的函数是一个生成器

def parse_args():
    # dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.
    # 变量总体描述
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")

    # 传入已训练过的模型 resume trained model  C2F_20.pth
    parser.add_argument("--r", dest="resume",
                        default='',
                        help="resume from which model", type=str,)

    # 使用的数据集   C2F  S2C  V2C  K2C  C2BDD
    parser.add_argument("--dataset", dest="dataset",
                        default="V2C", help="train dataset", type=str,)
    # 选用的模型和存放的位置
    parser.add_argument("--net", dest="net",
                        default="vgg16", help="vgg16, res101", type=str)
    parser.add_argument("--pretrained_path", dest="pretrained_path",
                        default="",
                        help="vgg16, res101", type=str,)
    # 保存模型的迭代次数
    parser.add_argument("--checkpoint_interval", dest="checkpoint_interval",
                        default=1,
                        help="number of iterations to save checkpoint",type=int,)
    # 模型的保存位置
    parser.add_argument("--save_dir", dest="save_dir",
                        default="./data/experiments/DA_Faster_ICR_CCR/V2C_fix1/model",
                        help="directory to save models", type=str,)
    # 训练epoch配置
    parser.add_argument("--max_epochs", dest="max_epochs",
                        default=50, help="max epoch for train", type=int, )
    parser.add_argument("--start_epoch", dest="start_epoch",
                        default=1, help="starting epoch", type=int)
    # tenrosbord显示
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        default='True', help='whether use tensorboard', action='store_true')
    # 加载数据时使用的多线程加载的进程数
    parser.add_argument("--nw", dest="num_workers",
                        default=8, help="number of worker to load data", type=int,)
    # 是否使用cuda   命令行遇动作时，为ture，默认值False
    parser.add_argument("--cuda", dest="cuda",
                        default='True', action="store_true", help="whether use CUDA")
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple GPUs', action='store_true')
    # 是否启用大尺度训练？？，默认False，选择时调用的config也不同
    # large_scale:True 读取res101_ls.yml   large_scale:False 读取vgg16/res101/50.yml
    parser.add_argument("--ls", dest="large_scale", action="store_true", help="whether use large imag scale",)
    # batch_size是不是只可以为1?? 设置其他报错
    parser.add_argument("--bs", dest="batch_size",
                        default=1, help="batch_size", type=int)
    # 是否执行类无关的bbox回归
    parser.add_argument("--cag", dest="class_agnostic", action="store_true",
                        help="whether perform class_agnostic bbox regression",)
    # config optimization
    parser.add_argument("--max_iter", dest="max_iter",
                        default=10000, help="max iteration for train", type=int,)
    parser.add_argument("--o", dest="optimizer",
                        default="sgd", help="training optimizer", type=str)
    # 学习率设置
    parser.add_argument("--lr", dest="lr",
                        default=0.001, help="starting learning rate", type=float) # 初始学习率
    parser.add_argument("--lr_decay_step", dest="lr_decay_step",
                        default=5,
                        help="step to do learning rate decay, unit is iter", type=int,)
    parser.add_argument("--lr_decay_gamma", dest="lr_decay_gamma",
                        default=0.1,
                        help="learning rate decay ratio", type=float,)    # 学习率下降率
    # 实例级权重？？
    parser.add_argument("--instance_weight_value", dest="instance_weight_value",
                        default=1.0,
                        help="instance_weight_value", type=float,)
    # DA损失函数参数
    parser.add_argument("--lamda", dest="lamda",
                        default=0.1, help="DA loss param", type=float)

    # 设置显示配置, 代中多少个batch显示
    parser.add_argument("--disp_interval", dest="disp_interval",
                        default=1000,
                        help="number of iterations to display", type=int, )

    # set training session 训练会话，针对多GPU
    parser.add_argument("--s", dest="session", default=1, help="training session", type=int)

    args = parser.parse_args()  # 获取终端交互及默认值，并返回
    return args

# 采样器,继承自pytorch模块
class sampler(Sampler):
    # 初始化 参数(训练集,batch的大小) -> 图片个数
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)   # batch的个数
        self.batch_size = batch_size
        # arange(0,batch_size)整型,不包含batch_size -> view(1, batch_size)从1到batch_size ->long()64位整型
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False   # 没有图片剩余
        if train_size % batch_size:  # 有图片剩余
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True   # 剩余

    # 用来产生迭代索引值的，也就是指定每个step需要读取哪些数据
    def __iter__(self):
        # randperm返回0~num_per_batch-1的随机序列   数字大小以batch_size大小间隔
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        #  分成一组一组的  expand扩张到(batch个数*大小) -> 使得rand_num的每一行是一组banch的连续序号
        self.rand_num = (rand_num.expand(self.num_per_batch, self.batch_size) + self.range)
        self.rand_num_view = self.rand_num.view(-1)   # 变成1维

        if self.leftover_flag:   # 加入不足一个batch剩余的部分样本
            # 在维度dim=0上拼接
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        # 返回一个迭代器（取图片的顺序）
        return iter(self.rand_num_view)

    # 返回每次迭代器的长度
    def __len__(self):
        return self.num_data

if __name__ == "__main__":

    args = parse_args()  # 获取训练命令行的解析参数
    print(args)

    # 读入数据集，设置名称和格式  anchor-based的scale、ratio，允许boxs的最大尺寸
    # Normal-to-Foggy:Cityscapes——>Foggy Cityscapes
    if args.dataset == "C2F":
        args.s_imdb_name = "cityscape_2007_train_s"
        args.t_imdb_name = "cityscape_2007_train_t"
        args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "cityscape_2007_test_t"
        args.set_cfgs = ["ANCHOR_SCALES", "[8,16,32]",
                         "ANCHOR_RATIOS", "[0.5 ,1 ,2]",
                         "MAX_NUM_GT_BOXES", "30", ]
    # Synthetic-to-Real:sim10k——>cityscapes_car
    elif args.dataset == "S2C":
        args.s_imdb_name = "sim10k_train"
        args.t_imdb_name = "cityscape_car_trainval"
        args.s_imdbtest_name = "sim10k_val"
        args.t_imdbtest_name = "cityscapes_car_test"
        # 使用时，需要把元组换成列表
        args.set_cfgs = ["ANCHOR_SCALES", "[8,16,32]","ANCHOR_RATIOS", "[0.5,1,2]",
                         "TRAIN.SCALES", "[800,]", "TRAIN.MAX_SIZE", "1600",
                         "TEST.SCALES", "[800,]", "TEST.MAX_SIZE", "1600", ]
    # 大差别图像的域偏移:voc_2007+voc_2012——>clipart
    elif args.dataset == "V2C":  # 剪切画
        args.s_imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.t_imdb_name = "clipart_trainval"
        args.t_imdbtest_name = "clipart_trainval"
        args.set_cfgs  = ["ANCHOR_SCALES", "[8,16,32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "20", ]
    # Cross-Camera: K2C
    elif args.dataset == "K2C":  # bdd车辆数据集
        args.s_imdb_name = "KITTI_train+KITTI_val"
        args.t_imdb_name = "cityscape_train"
        # args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "cityscapes_val"
        args.set_cfgs = ["ANCHOR_SCALES", "[4,8,16,32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "30", ]
    # Cross-Camera: C2BDD
    elif args.dataset == "C2BDD":  # bdd车辆数据集
        args.s_imdb_name = "cityscapes_train+cityscapes_val"
        args.t_imdb_name = "bdd100k_daytrain"
        # args.s_imdbtest_name = "cityscape_2007_test_s"
        args.t_imdbtest_name = "bdd100k_dayval"
        args.set_cfgs = ["ANCHOR_SCALES", "[8,16,32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "30", ]
    # voc_water_2007——>water
    elif args.dataset == "water":  # 水彩画
        args.s_imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
        args.t_imdb_name = "water_train"
        args.t_imdbtest_name = "water_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[8,16,32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "20", ]
    # 仅VOC_2007
    elif args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_train"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[4,8,16,32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "50", ]
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ["ANCHOR_SCALES", "[8,16,32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "20", ]
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ["ANCHOR_SCALES", "[4, 8, 16, 32]", "ANCHOR_RATIOS","[0.5,1,2]",
                         "MAX_NUM_GT_BOXES","50", ]
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ["ANCHOR_SCALES", "[4, 8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "30", ]
    elif args.dataset == "vg":  #？？？？？
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ["ANCHOR_SCALES", "[4, 8, 16, 32]", "ANCHOR_RATIOS", "[0.5,1,2]",
                         "MAX_NUM_GT_BOXES", "50",]

    # 载入设置文件的地址  默认 False
    print(args.net)
    args.cfg_file = ("cfgs/{}_ls.yml".format(args.net) if args.large_scale
                     else "cfgs/{}.yml".format(args.net))
    # True 读取res101_ls.yml  False 读取vgg16/res101/50.yml
    print(args.cfg_file)

    # 在设置文件地址读取设置文件
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)   # vgg16.yml
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("\033[1;45m*****************  config:   ********************\033[0m")
    pprint.pprint(cfg)   # 分成每个小项都单行显示
    np.random.seed(cfg.RNG_SEED)  # 3 利用随机数种子，每次生成的随机数相同，设置相同

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:   # 判断是否使用GPU训练
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 训练集加载及处理
    cfg.TRAIN.USE_FLIPPED = True  # 数据集增强方式
    cfg.USE_GPU_NMS = args.cuda  # True

    # 源域   db融合，包含数据增强及数据筛选,去除没有object的图片  2975->5930张
    s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.s_imdb_name)  # roidb.py
    # imdb：实例化后的数据集 imdb = cityscape(train_s, 2007)  roidb：imdb的一个属性，每张图片标注字典的列表
    # ratio_list：排列后的长宽比列表    ratio_index：排列顺序，对应图片(从小到大)
    s_train_size = len(s_roidb)  # add flipped

    # 目标域 2975->5930张  image_index*2
    t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name)
    t_train_size = len(t_roidb)

    print("\033[1;45msource {:d} target {:d} roidb entries\033[0m".format(len(s_roidb), len(t_roidb)))

    # 模型的保存位置  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    output_dir = args.save_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # batch采样器的实例化：自定义的采样函数，定义从数据集中抽取样本的策略
    s_sampler_batch = sampler(s_train_size, args.batch_size)
    t_sampler_batch = sampler(t_train_size, args.batch_size)

    # roibatchloader实例化：自定义的数据集，继承自data.Dataset，实现方法（__len__、__getitem__），每个batch中图片的长宽比相同
    s_dataset = roibatchLoader(s_roidb, s_ratio_list, s_ratio_index, args.batch_size,
                               s_imdb.num_classes, training=True,)
    # __getitem__一次只能获取一个数据，所以需要DataLoader定义的迭代器，实现batch读取
    # dataloader实例化：定义一个迭代器，实现批量读取，打乱数据并提供并行加速等功能
    s_dataloader = torch.utils.data.DataLoader(s_dataset, batch_size=args.batch_size,
                                               sampler=s_sampler_batch,
                                               num_workers=args.num_workers, drop_last=True)

    t_dataset = roibatchLoader(t_roidb, t_ratio_list, t_ratio_index, args.batch_size,
                               t_imdb.num_classes, training=False,)
    t_dataloader = torch.utils.data.DataLoader( t_dataset, batch_size=args.batch_size,
                                                sampler=t_sampler_batch,
                                                num_workers=args.num_workers, drop_last=True)

    # 源域tensor变量初始化
    im_data = torch.FloatTensor(1)    # 图片信息
    im_info = torch.FloatTensor(1)    # 缩放后图片的长、宽、缩放比
    im_cls_lb = torch.FloatTensor(1)  # 计算交叉熵时需要的lb，某个类别在该图片中是否存在目标，有1，无0
    num_boxes = torch.LongTensor(1)   # 图片中目标个数
    gt_boxes = torch.FloatTensor(1)   # box坐标
    need_backprop = torch.FloatTensor(1)  # 是否需要反向传播  源域1，目标域0

    # 目标域tensor变量初始化
    tgt_im_data = torch.FloatTensor(1)
    tgt_im_info = torch.FloatTensor(1)
    tgt_im_cls_lb = torch.FloatTensor(1)
    tgt_num_boxes = torch.LongTensor(1)
    tgt_gt_boxes = torch.FloatTensor(1)
    tgt_need_backprop = torch.FloatTensor(1)

    # Tensor转cuda（GPU运算）
    if args.cuda:
        # 源域
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_cls_lb = im_cls_lb.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        need_backprop = need_backprop.cuda()

        # 目标域
        tgt_im_data = tgt_im_data.cuda()
        tgt_im_info = tgt_im_info.cuda()
        tgt_im_cls_lb = tgt_im_cls_lb.cuda()
        tgt_num_boxes = tgt_num_boxes.cuda()
        tgt_gt_boxes = tgt_gt_boxes.cuda()
        tgt_need_backprop = tgt_need_backprop.cuda()

    if args.cuda:
        cfg.CUDA = True

    # 初始化网络  注意:网络的预训练模型在网络的__init__方法中定义
    if args.net == "vgg16":
        # 输入： 数据集类名称元组（背景＋类别名称）、是否使用预训练模型、预训练模型地址、是否采用类无关回归
        # 初始化VGG16及其父类_fasterRCNN
        fasterRCNN = vgg16(s_imdb.classes, pretrained_path=args.pretrained_path,
                           pretrained=True, class_agnostic=args.class_agnostic, )
    elif args.net == "res101":
        fasterRCNN = resnet(s_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)

    elif args.net == "res50":
        fasterRCNN = resnet(s_imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)

    elif args.net == "res152":
        fasterRCNN = resnet(s_imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
        # class_agnostic:控制bbox的回归方式，是否类无关，与之对应的是class_specific
        # agnostic：就是不管啥类别，把bbox调整到有东西(类别非0)，specific的话，必须要调整到确定的class
        # 一般使用class_agnostic   1、模型（代码）简单  2、参数数量少内存开销小运行速度快
    else:
        print("network is not defined")
        # 交互式调试使用pdb
        pdb.set_trace()

    # 创建模型  调用初始化模型及初始化参数
    fasterRCNN.create_architecture()   # faster_rcnn.py
    print("model created !!!!")

    # 学习率
    lr = cfg.TRAIN.LEARNING_RATE   # 0.001
    lr = args.lr     # 0.001

    # Module.parameters()和named_parameters() 二者都是迭代器
    # 前者返回模型的模块参数，后者返回参数名称+值
    params = []  # 创建空列表->用于存放参数
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:  # 需要训练的参数requires_grad = True
            if "bias" in key:     # 如果带有偏置b,在参数列表中添加字典
                # params为输入优化器函数的参数（params,lr,weight_decay）
                params += [ {"params": [value], "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),   # True  0.002
                             "weight_decay": cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0, } ]  # 0
            else:
                params += [ {"params": [value], "lr": lr,     # 0.001
                             "weight_decay": cfg.TRAIN.WEIGHT_DECAY, } ]   # 0.0005

    # 选择优化器
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    # # 是否使用GPU(cuda)   防止报错
    # if args.cuda:
    #     fasterRCNN.cuda()

    # 传入训练过的模型
    if args.resume != '':
        load_name = os.path.join(output_dir, args.resume)
        print("loading resume checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)        # 加载模型数据
        args.session = checkpoint["session"]
        args.start_epoch = checkpoint["epoch"] + 1  # 训练开始的epoch
        fasterRCNN.load_state_dict(checkpoint["model"])    # 加载模型参数
        optimizer.load_state_dict(checkpoint["optimizer"])  # 使用的优化器及其参数
        # add: 重载optimizer的参数时将所有的tensor都放到cuda上（加载时默认放在cpu上了）
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        lr = optimizer.param_groups[0]["lr"]                # 学习率
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (load_name))

    # 是否使用多GPU训练
    if args.mGPUs:
        # 指定特定的GPU训练时使用
        # device_ids = [2, 3]
        # fasterRCNN = nn.DataParallel(fasterRCNN, device_ids=device_ids)
        fasterRCNN = nn.DataParallel(fasterRCNN)

    # 是否使用GPU(cuda)
    if args.cuda:
        fasterRCNN.cuda()

    # 生成迭代对象
    # s_dataloader仅实现了批量读取，可以像迭代器一样使用，能通过循环操作读取
    # 不过不是迭代器，通过iter命令转换为迭代器，后面通过next可更方便的读取出data
    data_iter_s = iter(s_dataloader)
    data_iter_t = iter(t_dataloader)
    loss_temp = 0

    # 计算每个epoach的batch的个数
    # iters_per_epoch = int(10000 / args.batch_size)
    iters_per_epoch = int(s_train_size / args.batch_size)

    # 展示可视化训练
    if args.use_tfboard:
        log_name = os.path.join("logs",args.save_dir.split("/")[-2])
        logger = SummaryWriter("logs")  # 实例化SummaryWriter

    # 每一个epoch计算11个loss，然后用loss.backward反向传播，optimizer进行优化
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        epoch_start = time.time()

        print("\033[1;46m    start train !!!!\033[0m", end='\r')
        fasterRCNN.train()  # 设置训练的标志位
        loss_temp = 0
        start = time.time()  # 记录开始时间

        # 学习率下降,每间隔lr_decay_step个epoch下调学习率
        if epoch % (args.lr_decay_step + 1) == 0:
            # 更改学习率
            print("\033[1;46m    adjust lr !!!!\033[0m", end='\r')
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            # 记录更改的学习率
            lr *= args.lr_decay_gamma

        print("\033[1;46m   start   iters_per_epoch\033[0m", end='\r')
        for step in range(iters_per_epoch):  # 分为batch训练
            try:
                # 源域：取得一个batch的数据
                data = next(data_iter_s)
                # data是个list包含6个tensor的list:  roibatchLoder中__getitem__返回数据
                # train:padding_data, im_info, cls_lb, gt_boxes_padding, num_boxes, need_backprop,
                # test:data, im_info, cls_lb, gt_boxes, num_boxes, need_backprop
            except:
                data_iter_s = iter(s_dataloader)
                data = next(data_iter_s)
            try:
                # 目标域：取得一个batch的数据
                tgt_data = next(data_iter_t)
            except:
                data_iter_t = iter(t_dataloader)
                tgt_data = next(data_iter_t)
            # 根据data数据的维度改变自定义的数据，并赋值
            im_data.data.resize_(data[0].size()).copy_(data[0])  # 图片数据
            im_info.data.resize_(data[1].size()).copy_(data[1])   # 缩放后图片的长、宽、缩放比
            im_cls_lb.data.resize_(data[2].size()).copy_(data[2])   # 计算交叉熵时需要的lb
            gt_boxes.data.resize_(data[3].size()).copy_(data[3])    # box坐标  5
            num_boxes.data.resize_(data[4].size()).copy_(data[4])    # 图片中目标个数
            need_backprop.data.resize_(data[5].size()).copy_(data[5])  # 是否需要反向传播

            tgt_im_data.data.resize_(tgt_data[0].size()).copy_(tgt_data[0])  # change holder size
            tgt_im_info.data.resize_(tgt_data[1].size()).copy_(tgt_data[1])
            tgt_im_cls_lb.data.resize_(data[2].size()).copy_(data[2])
            tgt_gt_boxes.data.resize_(tgt_data[3].size()).copy_(tgt_data[3])
            tgt_num_boxes.data.resize_(tgt_data[4].size()).copy_(tgt_data[4])
            tgt_need_backprop.data.resize_(tgt_data[5].size()).copy_(tgt_data[5])

            """   faster-rcnn loss + DA loss for source and  DA loss for target    """
            # 梯度清零
            fasterRCNN.zero_grad()

            # 正向传播计算各个损失
            # 图像级类别正则化：用于实例级对齐时增加目标域内硬对齐的权值？？？
            # 判断实例级对齐难度
            print("\033[1;46m     forward propagation !!!!!!!!!!\033[0m", end='\r')

            (rois,         # [bs,256,5] 源域rpn推荐的候选框，其值来自于RPN回归分支输出
                           # 最后一维,前1:batch编号,后4:坐标
             rois_label,   # [256*bs] rpn推荐的候选框标签，正样本对应的标签,负样本均设置为0
             cls_prob,     # [256*bs,1] 类别预测
             bbox_pred,    # [bs,256,4] 预测的边界框的回归参数
             img_cls_loss, # 图像级类别正则化,判断图片中是否含有某类
             # faster RCNN原有损失
             rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox,
             # 源域、目标域图片级域自适应 实例级域自适应  一致性约束损失函数
             DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_img_loss_cls, tgt_DA_ins_loss_cls,
             DA_cst_loss, tgt_DA_cst_loss,) = fasterRCNN(
                im_data=im_data, im_info=im_info, im_cls_lb=im_cls_lb,
                gt_boxes=gt_boxes, num_boxes=im_cls_lb, need_backprop=need_backprop,
                tgt_im_data=tgt_im_data, tgt_im_info=tgt_im_info, tgt_gt_boxes=tgt_gt_boxes,
                tgt_num_boxes=tgt_num_boxes, tgt_need_backprop=tgt_need_backprop,
                weight_value=args.instance_weight_value, )

            # 计算目标损失，各个损失间存在权重
            loss = (img_cls_loss.mean()+ rpn_loss_cls.mean() + rpn_loss_box.mean() +
                    RCNN_loss_cls.mean()+ RCNN_loss_bbox.mean() +
                    args.lamda*(DA_img_loss_cls.mean() + DA_ins_loss_cls.mean() +
                                tgt_DA_img_loss_cls.mean()+ tgt_DA_ins_loss_cls.mean() +
                                DA_cst_loss.mean() + tgt_DA_cst_loss.mean()))
            loss_temp += loss.item()   # (.mean()加不加无所谓，因为都是一个数的tensor)

            # backward 反向传播,梯度清零
            optimizer.zero_grad()
            # backward  反向传播,自动计算梯度，执行后计算图会自动清空
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.0)
            # 更新参数，优化器利用损失值更新权重参数
            optimizer.step()

            # 终端显示展示部分
            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    # 本间隔的平均loss
                    loss_temp /= args.disp_interval + 1

                # 是否多GPU,并计算各部分的loss
                if args.mGPUs:
                    loss_category_cls = img_cls_loss.mean().item()  # ？？？？
                    # faster RCNN原有损失函数 RPN分类回归损失 RCNN分类回归损失
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    # 图像级域自适应损失函数
                    loss_DA_img_cls = args.lamda * (DA_img_loss_cls.mean().item() +
                                                    tgt_DA_img_loss_cls.mean().item()) / 2
                    # 实例级域自适应损失函数
                    loss_DA_ins_cls = args.lamda * (DA_ins_loss_cls.mean().item() +
                                                    tgt_DA_ins_loss_cls.mean().item()) / 2
                    # 一致性约束损失函数
                    loss_DA_cst = args.lamda * (DA_cst_loss.mean().item() + tgt_DA_cst_loss.mean().item()) / 2
                    # 计算rois中的前景、背景数目
                    fg_cnt = torch.sum(rois_label.data.ne(0))  # 前景数：rois_label中不为0的部分
                    bg_cnt = rois_label.data.numel() - fg_cnt  # 背景数：rois_label的数量 - 前景数
                else:
                    loss_category_cls = img_cls_loss.item()
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    loss_DA_img_cls = args.lamda* (DA_img_loss_cls.item() + tgt_DA_img_loss_cls.item())/ 2
                    loss_DA_ins_cls = args.lamda * (DA_ins_loss_cls.item() + tgt_DA_ins_loss_cls.item())/ 2
                    loss_DA_cst = args.lamda * (DA_cst_loss.item() + tgt_DA_cst_loss.item()) / 2
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                # 终端显示，选用的GPU、正在进行的epoch、epoch中batch迭代进行的次数、当前间隔的平均loss、当前学习率
                # 显示本间隔前景和背景比,和本间隔的花费时间
                print("epoch:[%2d][%3d/%3d] loss: %.4f, lr: %.2e, fg/bg=(%d/%d), time cost: %f"%
                      (epoch, step, iters_per_epoch, loss_temp, lr, fg_cnt, bg_cnt, end - start))
                # 显示各个部分的loss
                print("\t category_cls: %.4f, rpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f," 
                      "rcnn_box: %.4f, DA_img_loss: %.4f, DA_ins_loss: %.4f, cst_loss: %.4f"
                      %(loss_category_cls, loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,
                        loss_DA_img_cls, loss_DA_ins_cls, loss_DA_cst,))

                # 如果使用可视化进程的话添加参数
                if args.use_tfboard:
                    info = {'loss': loss_temp,
                            'category_cls':loss_category_cls,
                            'loss_rpn_cls': loss_rpn_cls,
                            'loss_rpn_box': loss_rpn_box,
                            'loss_rcnn_cls': loss_rcnn_cls,
                            'loss_rcnn_box': loss_rcnn_box,
                            'DA_img_loss':loss_DA_img_cls,
                            'DA_ins_loss':loss_DA_ins_cls,
                            'cst_loss':loss_DA_cst,}

                    logger.add_scalars("logs_s_{}/losses".format(args.save_dir.split("/")[-2]),
                                       info, (epoch - 1) * iters_per_epoch + step)
                # 清零累计loss
                loss_temp = 0
                # 时间重新开始
                start = time.time()
        epoch_end = time.time()
        print('[epoch%2d] training end, time: %d'%(epoch, (epoch_end-epoch_start)))

        # 训练到一定epoch或训练满epoch,保存模型
        if epoch % args.checkpoint_interval == 0 or epoch == args.max_epochs:
            # 设置保存路径和名称
            save_name = os.path.join(output_dir, "{}.pth".format(args.save_dir.split("/")[-2] + "_" + str(epoch)),)
            # 保存模型和其他参数
            save_checkpoint( {"session": args.session,
                              "epoch": epoch,
                              "model": fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
                              "optimizer": optimizer.state_dict(),
                              "pooling_mode": cfg.POOLING_MODE,
                              "class_agnostic": args.class_agnostic, } , save_name, )
            # 打印已经保存!
            print("save model: {}".format(save_name))

        # if args.use_tfboard:
        #     logger.close()