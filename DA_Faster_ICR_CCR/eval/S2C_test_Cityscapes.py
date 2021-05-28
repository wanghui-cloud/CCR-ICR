# coding=utf-8
from __future__ import absolute_import, division, print_function

import _init_paths
import argparse
import os
import pdb
import pickle
import pprint
import sys
import time

sys.path.append("..")
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import load_net, save_net, vis_detections
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from torch.autograd import Variable

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Test a Faster R-CNN network")
    # test的数据集
    parser.add_argument("--dataset", dest="dataset", help="test dataset", default="S2C", type=str,)
    # 训练的模型.pth文件的存放地址
    parser.add_argument("--model_dir", dest="model_dir", help="directory to load models",
                        default="../data/experiments/DA_Faster_ICR_CCR/S2C/model/S2C_", type=str, )
    # begin_epoch
    parser.add_argument("--begin_epoch", dest="begin_epoch", help="begin epoch", default=12, type=str, )
    # end_epoch
    parser.add_argument("--end_epoch", dest="end_epoch", help="end epoch", default=22, type=str, )
    # test_all、test_s、test_t
    parser.add_argument("--part", dest="part", default="test_t",
                        help="test_s or test_t or test_all", type=str, )
    # backbone网络部分
    parser.add_argument("--net", dest="net", help="vgg16, res50, res101, res152",
                        default="vgg16", type=str, )
    # 载入设置文件的地址
    parser.add_argument("--cfg", dest="cfg_file", help="optional config file",
                        default="../cfgs/vgg16.yml", type=str,)
    parser.add_argument("--ls", dest="large_scale", help="whether use large imag scale",
                        action="store_true", )
    parser.add_argument("--cuda", dest="cuda", default='True', help="whether use CUDA", action="store_true")
    parser.add_argument("--mGPUs", dest="mGPUs", help="whether use multiple GPUs", action="store_true")
    # 是否执行类无关的bbox回归
    parser.add_argument("--cag", dest="class_agnostic", help="whether perform class_agnostic bbox regression",
                        action="store_true", )
    # 可视化
    parser.add_argument("--vis", default='True', dest="vis", help="visualization mode", action="store_true")
    parser.add_argument( "--set", dest="set_cfgs", help="set config keys",
                         default=None, nargs=argparse.REMAINDER, )

    # 模型的哪个部分要并行，0:全部，1:ROI池前的模型
    parser.add_argument("--parallel_type", dest="parallel_type",
                        help="which part of model to parallel, 0:all, 1:model before roi pooling",
                        default=0, type=int,)
    # 定义了训好的检测模型名称
    parser.add_argument("--checksession", dest="checksession", help="checksession to load model",
                        default=1, type=int,)
    parser.add_argument("--checkepoch", dest="checkepoch", help="checkepoch to load network",
                        default=1, type=int,)
    parser.add_argument("--checkpoint", dest="checkpoint", help="checkpoint to load network",
                        default=10021, type=int,)

    # parser.add_argument("--model_name", dest="model_name", help="model file name",
    #                     default="res101.bs1.pth", type=str,)
    # 是否使用分类限制？？？？
    parser.add_argument("--USE_cls_cotrain", dest="USE_cls_cotrain", help="USE_cls_cotrain",
                        default=True, type=bool,)
    # 是否使用回归限制？？
    parser.add_argument("--USE_box_cotrain", dest="USE_box_cotrain",  help="USE_box_cotrain",
                        default=True, type=bool,)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == "__main__":

    args = parse_args()
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    # sim10k
    if args.dataset == "S2C":
        print("loading our dataset...........")
        args.s_imdb_name = "sim10k_train"
        args.t_imdb_name = "cityscape_car_train"
        args.s_imdbtest_name = "sim10k_val"
        args.t_imdbtest_name = "cityscape_car_val"
        # args.set_cfgs = ["ANCHOR_SCALES", "[4,8,16,32]", "ANCHOR_RATIOS", "[0.5,1,2]",
        #                  "MAX_NUM_GT_BOXES", "50", ]
        args.set_cfgs = [ "ANCHOR_RATIOS", "[0.5,1,2]",
                          "MAX_NUM_GT_BOXES", "50", ]

    # True 读取res101_ls.yml  False 读取vgg16/res101/50.yml
    args.cfg_file = ("/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/cfgs/{}_ls.yml".format(args.net) if args.large_scale
                     else "/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/cfgs/{}.yml".format(args.net))
    print(args.net)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("\033[1;45m*****************  config:   ********************\033[0m")
    pprint.pprint(cfg)   # 分成每个小项都单行显示
    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = False  # 不进行数据增强

    # 根据设置不同，读取包含不同内容的数据集 s_imdbtest_name、t_imdbtest_name、all_imdbtest_name
    if args.part == "test_s":
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.s_imdbtest_name, False)
    elif args.part == "test_t":
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.t_imdbtest_name, False)
    elif args.part == "test_all":
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.all_imdbtest_name, False)
    else:
        print("don't have the test part !")
        # 交互式调试使用pdb
        pdb.set_trace()
    print("\033[1;45mtest_size {:d} roidb entries\033[0m".format(len(roidb)))

    imdb.competition_mode(on=True)
    # 调用了数据集中competition_mode()函数，设置了use_salt=False  cleanup=False

    print("{:d} roidb entries".format(len(roidb)))

    # input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    # print(input_dir)
    # if not os.path.exists(input_dir):
    #   raise Exception('There is no input directory for loading network from ' + input_dir)
    # load_name = os.path.join(input_dir,
    #   'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == "res101":
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == "res50":
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == "res152":
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    print("model created !!!!")

    begin_epoch = args.begin_epoch
    end_epoch = args.end_epoch
    model_dir = args.model_dir
    for i in range(begin_epoch, end_epoch + 1):
        # 加载模型
        load_name = model_dir + str(i)
        print("load checkpoint %s" % (load_name + ".pth"))
        checkpoint = torch.load(load_name + ".pth")
        fasterRCNN.load_state_dict({k: v for k, v in checkpoint["model"].items()
                                    if k in fasterRCNN.state_dict()})
        # fasterRCNN.load_state_dict(checkpoint['model'])
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("load model successfully!")

        # 变量初始化
        im_data = torch.FloatTensor(1)   # 图片信息
        im_info = torch.FloatTensor(1)   # 图片的长、宽、缩放比
        num_boxes = torch.LongTensor(1)   # 图片中目标个数
        gt_boxes = torch.FloatTensor(1)   # box坐标、分类 5

        # ship to cuda
        if args.cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        if args.cuda:
            cfg.CUDA = True
        if args.cuda:
            fasterRCNN.cuda()

        start = time.time()
        max_per_image = 100

        save_name = os.path.join(load_name.split("/")[-3], load_name.split("/")[-1] + ".pth")
        num_images = len(imdb.image_index)   # 图片数
        all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

        output_dir = get_output_dir(imdb, save_name)
        # roibatchloader实例化
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes,
                                 training=False, normalize=False,)
        # dataloader实例化
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)
        # 生成迭代对象
        data_iter = iter(dataloader)

        # _t 字典中二个时间im_detect、misc
        _t = {"im_detect": time.time(), "misc": time.time()}
        det_file = os.path.join(output_dir, "detections.pkl")

        fasterRCNN.eval()
        print("\033[1;46m    start test !!!!\033[0m")
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
        for i in range(num_images):

            data = next(data_iter)

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            det_tic = time.time()
            # 正向传播
            (
                rois,        # [1,256,5] 源域rpn推荐的候选框，其值来自于RPN回归分支输出
                          # 最后一维,前1:batch编号,后4:坐标
                cls_prob,    # [256,1] 类别预测
                bbox_pred,   # [1,256,4] 预测的边界框的回归参数
                rpn_loss_cls, rpn_loss_box,
                RCNN_loss_cls, RCNN_loss_bbox,
                rois_label,  # [256] rpn推荐的候选框标签，正样本对应的标签,负样本均设置为0
            ) = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            # 预测分数
            scores = cls_prob.data         # [256,1]
            # 预测的边界框
            boxes = rois.data[:, :, 1:5]   # [1,256,4]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    if args.class_agnostic:
                        box_deltas = (box_deltas.view(-1, 4)*
                                      torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() +
                                      torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda())
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = (box_deltas.view(-1, 4) *
                                      torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() +
                                      torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda())
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2].item()

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()

            # nms设置的不同权重？？？？
            vis = args.vis
            if vis:
                thresh = 0.05
            else:
                thresh = 0.0

            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)

            for j in xrange(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    # keep = nms(cls_dets, cfg.TEST.NMS)
                    keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]

                    if vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack( [all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)])

                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            sys.stdout.write("im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r".
                             format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()

            if vis:
                cv2.imwrite("result.png", im2show)
                # pdb.set_trace()
                # cv2.imshow('test', im2show)
                # cv2.waitKey(0)

        with open(det_file, "wb") as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print("Evaluating detections")
        imdb.evaluate_detections(all_boxes, output_dir)

        end = time.time()
        print("test time: %0.4fs" % (end - start))
