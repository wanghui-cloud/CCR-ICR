import os
from shutil import copy, rmtree
import random

train_dir = "/home/ubuntu/Desktop/datasets/BDD100K/JPEGImages/100k/train/"
val_dir = "/home/ubuntu/Desktop/datasets/BDD100K/JPEGImages/100k/val/"
test_dir = "/home/ubuntu/Desktop/datasets/BDD100K/JPEGImages/100k/test/"
out_dir = "/home/ubuntu/Documents/CR-DA-DET/DA_Faster_ICR_CCR/data/datasets/BDD_TODO/ImageSets/Main/"

dir = [train_dir, val_dir, test_dir]

for root in dir:
    file = os.path.join(out_dir, root.split("/")[-2] + '.txt')
    if os.path.exists(file):
        f = open(file, 'w')
    else:
        print("{} file not exists".format(file))

    for _, _, img_list in os.walk(root):
        num = len(img_list)

        for img in img_list:
            img = img.split(".")[0] + '\n'
            f.write(img)

    f.close()

dir_trainval = [train_dir, val_dir]

file = os.path.join(out_dir, 'trainval.txt')

if os.path.exists(file):
    f = open(file, 'w')
else:
    print("{} file not exists".format(file))

for root in dir_trainval:
    for _, _, img_list in os.walk(root):
        num = len(img_list)

        for img in img_list:
            img = img.split(".")[0] + '\n'
            f.write(img)

f.close()


# import os
# import random
#
# trainval_percent = 0.2
# train_percent = 0.8
# xmlfilepath = 'Annotations'
# txtsavepath = 'ImageSets\Main'
# total_xml = os.listdir(xmlfilepath)
#
# num = len(total_xml)
# list = range(num)
# tv = int(num * trainval_percent)
# tr = int(tv * train_percent)
# trainval = random.sample(list, tv)
# train = random.sample(trainval, tr)
#
# ftrainval = open('ImageSets/Main/trainval.txt', 'w')
# ftest = open('ImageSets/Main/test.txt', 'w')
# ftrain = open('ImageSets/Main/train.txt', 'w')
# fval = open('ImageSets/Main/val.txt', 'w')
#
# for i in list:
#     name = total_xml[i][:-4] + '\n'
#     if i in trainval:
#         ftrainval.write(name)
#         if i in train:
#             ftest.write(name)
#         else:
#             fval.write(name)
#     else:
#         ftrain.write(name)
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest.close()
