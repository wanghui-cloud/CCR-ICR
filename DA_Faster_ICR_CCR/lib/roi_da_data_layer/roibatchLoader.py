"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.utils.data as data
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.config import cfg
from PIL import Image
from roi_da_data_layer.minibatch import get_minibatch
import cv2


class roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None,):
        self._roidb = roidb
        self._num_classes = num_classes   # class的数量  包含背景 +1
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT   #  600
        self.trim_width = cfg.TRAIN.TRIM_WIDTH     # 600
        self.max_num_box = cfg.MAX_NUM_GT_BOXES    # 30
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)   # 所有图片的个数

        # given the ratio_list, we want to make the ratio same for each batch.
        # 希望每个batch中图片的长宽比相同
        # 创建一个和图片个数相同的0张量
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()

        num_batch = int(np.ceil(len(ratio_index) / batch_size))  # 向上取整
        for i in range(num_batch):
            # batch的头索引(每个batch的第一张图)
            left_idx = i * batch_size
            # batch的尾索引(最后一个batch不超过总体的尾部)(每个batch的最后一张图)
            right_idx = min((i + 1) * batch_size - 1, self.data_size - 1)  # 防止最后一个batch序号不够

            # (长宽比列表是从小到大)
            # 尾图高,跟随尾图,低,跟随首图，首尾图不同,设为1

            # batch最后一张图的长宽比 < 1  (高图片)，图片长宽比选择第一张
            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            # batch最前第一张图(宽图片)，图片长宽比选择
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            # trainset ratio list ,each batch is same number
            # 由于ratio_list已是升序，故存入的是此batch中
            self.ratio_list_batch[left_idx : (right_idx + 1)] = target_ratio

    # 通过给定索引获取数据和标签,一张图片的
    def __getitem__(self, index):  # only one sample
        # 如果在训练过程中
        if self.training:
            # ratio_list  -> 排列后的长宽比列表(从小到大)
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        '''
            根据长宽比(从小到大)取出图片对应roi参数的字典{}
        '''
        # minibatch_db：一张图片的roi字典
        minibatch_db = [self._roidb[index_ratio]]   # 得到单张图的roi字典，batch的个数
        blobs = get_minibatch(minibatch_db, self._num_classes)
        # blobs字典包含:data need_backprop  im_info  img_id  gt_boxes  cls_lb
        # img_id:int图片序号、gt_boxes:Reg+cls用，box_num*5
        # need_backprop:是否要BP,0或者1,target不需要反传,source需要
        # data：图片缩放后得四维np信息，方便后续一个batch拼合

        data = torch.from_numpy(blobs["data"])
        # add 图片应该是RGB三个通道的，判断有没有混入第三个维度是4的图片，重新保存一遍即可
        if data.size(0) == 4 :
            print(data)
        im_info = torch.from_numpy(blobs["im_info"])  # im_info：图片缩放后的长,宽,缩放比
        cls_lb = torch.from_numpy(blobs["cls_lb"])    # 计算交叉熵使用,某个类别在该图片中是否存在目标，有1，无0
        data_height, data_width = data.size(1), data.size(2)

        if self.training:
            """
            da-faster-rcnn layer............
            """
            np.random.shuffle(blobs["gt_boxes"])   # 随机打乱blobs中图片bbox的顺序
            gt_boxes = torch.from_numpy(blobs["gt_boxes"])  # numpy转移到torch
            need_backprop = blobs["need_backprop"][0]

            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################
            # 将输入图像填充到每个组的固定大小
            # NOTE1: need to cope with the case where a group cover both conditions. (done)
            # NOTE2: need to consider the situation for the tail samples. (no worry)
            # NOTE3: need to implement a parallel data loader. (no worry)
            # get the index range

            # 读入一个batch的目标长宽比  ratio = width / float(height)
            ratio = self.ratio_list_batch[index]

            # roidb.need_crop属性判断,尽可能多的包含bbox的面积
            # Ture：data裁剪+gt_boxes坐标改变
            # 裁剪图片
            if self._roidb[index_ratio]["need_crop"]:
                if ratio < 1:   # 目标长宽比，宽度<<高度，需要裁剪高度 height
                    # means that data_width << data_height, need to crop the data_height
                    # 读取bbox的最高点和最低点
                    min_y = int(torch.min(gt_boxes[:, 1]))
                    max_y = int(torch.max(gt_boxes[:, 3]))
                    # 长边height需要裁剪成为的大小
                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:   # data_height blobs缩放后的
                        trim_size = data_height
                    # bbox的最大距离
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
                        # bbox的最大距离 < 裁剪范围
                        if (box_region - trim_size) < 0:
                            # 设点裁剪最低点的范围,并在范围中随机选择
                            y_s_min = max(max_y - trim_size, 0)
                            y_s_max = min(min_y, data_height - trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        # bbox的最大距离 >= 裁剪范围
                        else:
                            y_s_add = int((box_region - trim_size) / 2)
                            # 刚好相等
                            if y_s_add == 0:
                                y_s = min_y
                            # bbox的最大距离 > 裁剪范围
                            else:
                                y_s = np.random.choice(range(min_y, min_y + y_s_add))
                    # 进行裁剪,按照以上原则,保证长宽比确定,->尽可能多的包含bbox的面积
                    data = data[:, y_s : (y_s + trim_size), :, :]

                    # 改变blobs中bbox的坐标跟随着裁剪进行变更  仅y坐标
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                    # update gt bounding box according the trip
                    # 防止超出图片的边界(bbox的最大距离 > 裁剪范围)的情况下
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    # clamp_（），将输入张量每个元素的夹紧到区间[min, max]
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)

                else:   # 宽度>>高度  需要裁剪宽度
                    # means that data_width >> data_height, need to crop the data_width
                    # 读取bbox的最左点和最右点
                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))
                    # 长边height需要裁剪成为的大小
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region - trim_size) < 0:
                            x_s_min = max(max_x - trim_size, 0)
                            x_s_max = min(min_x, data_width - trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region - trim_size) / 2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x + x_s_add))
                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    # 改变blobs中bbox的坐标跟随着裁剪进行变更  仅x坐标
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    # update gt bounding box according the trip
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    # clamp_（），将输入张量每个元素的夹紧到区间[min, max]
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

            # based on the ratio, padding the image.
            # padding 图片
            if ratio < 1:  # 高图片  data_width < data_height
                trim_size = int(np.floor(data_width / ratio))

                # 创建一个空矩阵(高*宽*3)  根据宽度和batch的目标长宽比定义一个空tensor
                padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), data_width, 3).zero_()
                # 把图片放入空tensor，即其余位置填0
                padding_data[:data_height, :, :] = data[0]
                # update im_info  更改图片长度
                im_info[0, 0] = padding_data.size(0)
                # print("height %d %d \n" %(index, anchor_idx))

            elif ratio > 1:    # 宽图片  data_width > data_height
                # 创建一个矩阵(高*宽*3)  根据宽度和batch的目标长宽比定义一个空tensor
                padding_data = torch.FloatTensor(data_height, int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                # update im_info  更改图片宽度
                im_info[0, 1] = padding_data.size(1)

            else:   # batch的目标长宽比为1，以最短边
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
                padding_data = data[0][:trim_size, :trim_size, :]
                # gt_boxes.clamp_(0, trim_size)
                gt_boxes[:, :4].clamp_(0, trim_size)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size

            # check the bounding box:  去除面积为0的bbox,选出有面积的bbox，形成列表
            not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | (gt_boxes[:, 1] == gt_boxes[:, 3])
            keep = torch.nonzero(not_keep == 0).view(-1)

            # 创建数组.存放所有blobs中的有面积的bbox的坐标  bbox_num * 5
            gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
            # 如果keep张量的元素个数不为0，图片中存在面积不为0的bbox
            if keep.numel() != 0:
                # 取出bbox的值
                gt_boxes = gt_boxes[keep]
                # 取出bbox的数量，有限制单张图片的目标框数量不超过多少
                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                # 写入张量中
                gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0

            # permute trim_data to adapt to downstream processing
            # 进行维度转化,通道数放在最前
            # view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，
            # 需要用contiguous()来返回一个contiguous copy
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)
            # add 查看dataloader单张图片大小
            # print("padding_data:", padding_data.shape[0:2])


            return (padding_data, im_info, cls_lb, gt_boxes_padding, num_boxes, need_backprop,)
            # padding后图像数据  im_info：图片的长、宽、缩放比  gt_boxes_padding：bbox的5个标注
            # cls_lb：gt_classes转化的计算交叉熵需要使用的 某个类别在该图片中是否存在目标，有1，无0
            # num_boxes -> bbox的数量 need_backprop -> 是否需要反向传播

        # 不是训练过程 -> 并不加载GT
        else:

            data = (data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width))
            im_info = im_info.view(3)

            # 预测的图片没有对应的标签
            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0
            need_backprop = 0   # 不需要反向传播

            return data, im_info, cls_lb, gt_boxes, num_boxes, need_backprop
     # 返回：padding后图像数据，图片的长、宽、缩放比，bbox的5个标注，bbox数量=0 不需要反向传播


    # 提供数据集的大小
    def __len__(self):
        return len(self._roidb)



