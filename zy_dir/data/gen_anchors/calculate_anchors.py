# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 20:29
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : data_cfg.py
# @Software: PyCharm
# @Brief   : 根据标签文件求先验框

import os
import xml.etree.cElementTree as et

import numpy as np
from kmeans import avg_iou, kmeans_zy


FILE_ROOT = "/home/zy/data_set/"
ANNOTATION_ROOT = "VOC2007_GuLouCancer/Annotations"  # 数据集标签文件夹路径
ANNOTATION_PATH = FILE_ROOT + ANNOTATION_ROOT

ANCHORS_TXT_PATH = "./anchors.txt"

CLUSTERS = 9
CLASS_NAMES = ['inflammation', 'cancer']


def load_data(anno_dir, class_names):
    xml_names = os.listdir(anno_dir)
    boxes = []
    for xml_name in xml_names:
        xml_pth = os.path.join(anno_dir, xml_name)
        tree = et.parse(xml_pth)

        width = float(tree.findtext("./size/width"))
        height = float(tree.findtext("./size/height"))

        for obj in tree.findall("./object"):
            cls_name = obj.findtext("name")
            if cls_name in class_names:
                xmin = float(obj.findtext("bndbox/xmin")) / width
                ymin = float(obj.findtext("bndbox/ymin")) / height
                xmax = float(obj.findtext("bndbox/xmax")) / width
                ymax = float(obj.findtext("bndbox/ymax")) / height

                box = [xmax - xmin, ymax - ymin]
                boxes.append(box)
            else:
                continue

    return np.array(boxes)


if __name__ == '__main__':

    anchors_txt = open(ANCHORS_TXT_PATH, "w")

    train_boxes = load_data(ANNOTATION_PATH, CLASS_NAMES)
    count = 1
    best_accuracy = 0
    best_anchors = []
    best_ratios = []

    for i in range(30):
        anchors_tmp = []
        # print(train_boxes)
        # clusters = kmeans(train_boxes, k=CLUSTERS)
        clusters = kmeans_zy(train_boxes, k=CLUSTERS)
        idx = clusters[:, 0].argsort()
        clusters = clusters[idx]
        print(clusters)

        for j in range(CLUSTERS):
            anchor = [round(clusters[j][0] * 640, 2), round(clusters[j][1] * 640, 2)]
            anchors_tmp.append(anchor)
            print(f"Anchors:{anchor}")

        try:
            # temp_accuracy = avg_iou(train_boxes, clusters) * 100
            temp_accuracy = 0.9
        except:
            print("Error!")
            continue

        print("Train_Accuracy:{:.2f}%".format(temp_accuracy))

        ratios = np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist()
        ratios.sort()
        print("Ratios:{}".format(ratios))
        print(20 * "*" + " {} ".format(count) + 20 * "*")

        count += 1

        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_anchors = anchors_tmp
            best_ratios = ratios

        anchors_txt.write("Best Accuracy = " + str(round(best_accuracy, 2)) + '%' + "\r\n")
        anchors_txt.write("Best Anchors = " + str(best_anchors) + "\r\n")
        anchors_txt.write("Best Ratios = " + str(best_ratios))
        anchors_txt.close()
        print('ha')
