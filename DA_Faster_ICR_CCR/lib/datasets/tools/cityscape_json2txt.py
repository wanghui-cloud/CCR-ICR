# convert cityscape dataset to pascal voc format dataset

# 1. convert every cityscape image label '.json' to '.txt'

import json
import os
from os import listdir, getcwd
from os.path import join
import os.path

rootdir = '/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/data/datasets/cityscapes_car'  # 写自己存放图片的数据地址


def position(pos):
    # 该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x = []
    y = []
    nums = len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    # print(x_max,y_max,x_min,y_min)
    b = (float(x_min), float(y_min), float(x_max), float(y_max))
    # print(b)
    return b

# pascal voc 标准格式
# < xmin > 174 < / xmin >
# < ymin > 101 < / ymin >
# < xmax > 349 < / xmax >
# < ymax > 351 < / ymax >

def convert(size, box):
    # 该函数将xmin,ymin,xmax,ymax转为x,y,w,h中心点坐标和宽高
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    print((x, y, w, h))
    return (x, y, w, h)


def convert_annotation(image_id):
    # load_f = open("/home/ubuntu/PycharmProjects/city2pascal/source/train/tubingen/%s_gtFine_polygons.json" % (image_id), 'r')  # 导入json标签的地址
    load_f = open("/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/data/datasets/cityscapes_car/Annotations_json/%s_gtFine_polygons.json" % (image_id),
                  'r')  # 导入json标签的地址
    load_dict = json.load(load_f)
    out_file = open('/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/data/datasets/cityscapes_car/txt/%s_leftImg8bit.txt' % (image_id), 'w')  # 输出标签的地址
    # keys=tuple(load_dict.keys())
    w = load_dict['imgWidth']  # 原图的宽，用于归一化
    h = load_dict['imgHeight']
    # print(h)
    objects = load_dict['objects']
    nums = len(objects)
    # print(nums)
    # object_key=tuple(objects.keys()
    cls_id = ''
    for i in range(0, nums):
        labels = objects[i]['label']
        # print(i)
        if (labels in ['person', 'rider']):
            # print(labels)
            pos = objects[i]['polygon']
            bb = position(pos)
            # bb = convert((w, h), b)
            cls_id = 'pedestrian'  # 我这里把行人和骑自行车的人都设为类别pedestrian
            out_file.write(cls_id + " " + " ".join([str(a) for a in bb]) + '\n')
            # print(type(pos))
        elif (labels in ['car', 'truck', 'bus', 'caravan', 'trailer']):
            # print(labels)
            pos = objects[i]['polygon']
            bb = position(pos)
            # bb = convert((w, h), b)
            cls_id = 'car'  # 我这里把各种类型的车都设为类别car
            out_file.write(cls_id + " " + " ".join([str(a) for a in bb]) + '\n')

    if cls_id == '':
        print('no label json:', "%s_gtFine_polygons.json" % (image_id))


def image_id(rootdir):
    a = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # print(filename)

            filename = filename[:-16]
            # filename = filename.strip('_leftImg8bit.png')
            a.append(filename)
    return a


if __name__ == '__main__':
    names = image_id(rootdir)
    for image_id in names:
        print(image_id)
        convert_annotation(image_id)


# import os
# import numpy as np
# import codecs
# import json
# from glob import glob
# import cv2
# import shutil
# from sklearn.model_selection import train_test_split
# #1.标签路径
# #原始labelme标注数据路径
# labelme_path = "/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/data/datasets/cityscapes_car/Annotations_json/"
# # 保存路径
# saved_path = "/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/data/datasets/cityscapes_car/VOC/"
#
# #2.创建要求文件夹
# if not os.path.exists(saved_path + "Annotations"):
#     os.makedirs(saved_path + "Annotations")
# if not os.path.exists(saved_path + "JPEGImages/"):
#     os.makedirs(saved_path + "JPEGImages/")
# if not os.path.exists(saved_path + "ImageSets/Main/"):
#     os.makedirs(saved_path + "ImageSets/Main/")
#
# #3.获取待处理文件
# files = glob(labelme_path + "*.json")
# files = [i.split("/")[-1].split("gtFine_polygons.json")[0] for i in files]
# # munich_000045_000019_
#
# #4.读取标注信息并写入 xml
# for json_file_ in files:
#     json_filename = labelme_path + json_file_ + "gtFine_polygons.json"
#     json_file = json.load(open(json_filename,"r",encoding="utf-8"))
#     height, width, channels = cv2.imread(labelme_path + json_file_ +"leftImg8bit.png").shape
#     with codecs.open(saved_path + "Annotations/"+json_file_ + "gtFine_polygons" + ".xml","w","utf-8") as xml:
#         xml.write('<annotation>\n')
#         xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
#         xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
#         xml.write('\t<source>\n')
#         xml.write('\t\t<database>The UAV autolanding</database>\n')
#         xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
#         xml.write('\t\t<image>flickr</image>\n')
#         xml.write('\t\t<flickrid>NULL</flickrid>\n')
#         xml.write('\t</source>\n')
#         xml.write('\t<owner>\n')
#         xml.write('\t\t<flickrid>NULL</flickrid>\n')
#         xml.write('\t\t<name>WH</name>\n')
#         xml.write('\t</owner>\n')
#         xml.write('\t<size>\n')
#         xml.write('\t\t<width>'+ str(width) + '</width>\n')
#         xml.write('\t\t<height>'+ str(height) + '</height>\n')
#         xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
#         xml.write('\t</size>\n')
#         xml.write('\t\t<segmented>0</segmented>\n')
#         for multi in json_file["objects"]:
#             points = np.array(multi["polygon"])
#             xmin = min(points[:,0])
#             xmax = max(points[:,0])
#             ymin = min(points[:,1])
#             ymax = max(points[:,1])
#             label = multi["label"]
#             if xmax <= xmin:
#                 pass
#             elif ymax <= ymin:
#                 pass
#             else:
#                 xml.write('\t<object>\n')
#                 xml.write('\t\t<name>'+label+'</name>\n')
#                 xml.write('\t\t<pose>Unspecified</pose>\n')
#                 xml.write('\t\t<truncated>1</truncated>\n')
#                 xml.write('\t\t<difficult>0</difficult>\n')
#                 xml.write('\t\t<bndbox>\n')
#                 xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
#                 xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
#                 xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
#                 xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
#                 xml.write('\t\t</bndbox>\n')
#                 xml.write('\t</object>\n')
#                 print(json_filename,xmin,ymin,xmax,ymax,label)
#         xml.write('</annotation>')
#
# #5.复制图片到 VOC2007/JPEGImages/下
# image_files = glob(labelme_path + "*.jpg")
# print("copy image files to VOC007/JPEGImages/")
# for image in image_files:
#     shutil.copy(image, saved_path +"JPEGImages/")
#
# #6.split files for txt
# txtsavepath = saved_path + "ImageSets/Main/"
# ftrainval = open(txtsavepath+'/trainval.txt', 'w')
# ftest = open(txtsavepath+'/test.txt', 'w')
# ftrain = open(txtsavepath+'/train.txt', 'w')
# fval = open(txtsavepath+'/val.txt', 'w')
# total_files = glob("./VOC2007/Annotations/*.xml")
# total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
# #test_filepath = ""
# for file in total_files:
#     ftrainval.write(file + "\n")
# #test
# #for file in os.listdir(test_filepath):
# #    ftest.write(file.split(".jpg")[0] + "\n")
# #split
# train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
# #train
# for file in train_files:
#     ftrain.write(file + "\n")
# #val
# for file in val_files:
#     fval.write(file + "\n")
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# #ftest.close()
