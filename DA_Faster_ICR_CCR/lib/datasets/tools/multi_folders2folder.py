import os
import shutil
import re

file_type = '.json'#指定文件类型
#原文件夹路径
old_path = '/home/ubuntu/Desktop/datasets/BDD100K/bdd100k (2)/labels'
#新文件夹路径
new_path = '/home/ubuntu/Desktop/datasets/BDD100K/Annotations_json'

name =[]
final_name_list = []
# root:当前文件夹的全局路径
# dirs:当前文件夹下的文件夹名称
# files:当前文件夹下的文件
# 不断循环过程中，进入每个文件夹
for root, dirs, files in os.walk(old_path):

    for i in files:
        if file_type in i:
            name.append(root + "/" + i)
print(" select {} {} files ".format(len(name), file_type))

print("start copy files to one folder !!")
for file_name in name:
    # 复制指定文件到另一个文件夹里，并删除原文件夹中的文件
    shutil.copyfile(os.path.join(old_path, file_name),
                    os.path.join(new_path, file_name.split("/")[-1]))
    # 路径拼接要用os.path.join，复制指定文件到另一个文件夹里
    # 删除原文件夹中的指定文件文件
    # os.remove(os.path.join(old_path, file_name))
print("multi folder to one folder done !!")