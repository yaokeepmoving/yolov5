# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 20:29
# @Author  : PeterH
# @Email   : peterhuang0323@outlook.com
# @File    : data_cfg.py
# @Software: PyCharm
# @Brief   : 生成测试、验证、训练的图片和标签

import os
import shutil
from pathlib import Path
from shutil import copyfile
from xml.dom.minidom import parse

import numpy as np
from PIL import Image, ImageDraw

# 类别列表
# CLASS_NAME_LIST = ['inflammation', 'cancer']
CLASS_NAME_LIST = ['plexus']

# 类别名称映射到序号
NAME2INDEX_DICT = {name: i for i, name in enumerate(CLASS_NAME_LIST)}

# VOC 格式的数据集

DATA_STORE_ROOT = '/home/zy/data_set/'

IMAGE_SET_ROOT = DATA_STORE_ROOT + "VOC_HD_winsize_1024_stepsize_512_threshold_0.7/ImageSets/Main"  # 图片区分文件的路径
IMAGE_PATH = DATA_STORE_ROOT + "VOC_HD_winsize_1024_stepsize_512_threshold_0.7/JPEGImages"  # 图片的位置
ANNOTATIONS_PATH = DATA_STORE_ROOT + "VOC_HD_winsize_1024_stepsize_512_threshold_0.7/Annotations"  # 数据集标签文件的位置
LABELS_ROOT = DATA_STORE_ROOT + "VOC_HD_winsize_1024_stepsize_512_threshold_0.7/Labels"  # 进行归一化之后的标签位置

# yolov5 格式的数据集

DEST_IMAGES_PATH = "diy_dataset_yolov5/HD_winsize_1024_stepsize_512_threshold_0.7/images"  # 区分训练集、测试集、验证集的图片目标路径
DEST_LABELS_PATH = "diy_dataset_yolov5/HD_winsize_1024_stepsize_512_threshold_0.7/labels"  # 区分训练集、测试集、验证集的标签文件目标路径


def coordinate_converter(size, box):
    """
    将标注的 xml 文件标注转换为 darknet 形的坐标
    :param size: 图片的尺寸： [w,h]
    :param box: anchor box 的坐标 [左上角x,左上角y,右下角x,右下角y,]
    :return: 转换后的 [x,y,w,h]
    """

    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1. / int(size[0]))
    dh = np.float32(1. / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]


def save_file(img_jpg_file_name, size, img_box):
    outpath = LABELS_ROOT + '/' + img_jpg_file_name + '.txt'

    fo = open(outpath, "w")
    for box in img_box:
        name = box[0]
        if name not in NAME2INDEX_DICT:  # 没有在类别列表中
            print(f'[ warning ] Not recognized category: "{name}", valid category list is {CLASS_NAME_LIST}. Skip this annotation!' )
            continue

        cls_num = NAME2INDEX_DICT.get(name, None)

        assert cls_num is not None

        # 坐标转换
        new_box = coordinate_converter(size, box[1:])

        fo.write(
            f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n")

    fo.flush()
    fo.close()
    print(f'new annotation file: {outpath}')


def test_dataset_box_feature(file_name, point_array):
    """
    使用样本数据测试数据集的建议框
    :param image_name: 图片文件名
    :param point_array: 全部的点 [建议框sx1,sy1,sx2,sy2]
    :return: None
    """
    img_path = f"{IMAGE_PATH}/{file_name}"
    if not os.path.exists(img_path):
        return None

    im = Image.open(img_path)
    imDraw = ImageDraw.Draw(im)
    for box in point_array:
        x1 = box[1]
        y1 = box[2]
        x2 = box[3]
        y2 = box[4]
        imDraw.rectangle((x1, y1, x2, y2), outline='green', width=5)

    # outpath = f'{img_path}.debug.jpg'
    outpath = f'./vis_xml_annotations/{file_name}.debug.jpg'
    im.save(outpath)
    print(f'[ test_dataset_box_feature ] {img_path} => {outpath}')


def get_xml_data(file_path, img_xml_file):
    xml_path = file_path + '/' + img_xml_file + '.xml'
    print(f'xml_path: {xml_path}')

    dom = parse(xml_path)
    root = dom.documentElement
    try:
        img_name = root.getElementsByTagName("filename")[0].childNodes[0].data
    except:
        shutil.move(xml_path, './invalid_xml_filename/')
        return

    try:
        img_size = root.getElementsByTagName("size")[0]
    except:
        shutil.move(xml_path, './invalid_xml_size/')
        return

    objects = root.getElementsByTagName("object")
    img_w = img_size.getElementsByTagName("width")[0].childNodes[0].data
    if img_w == 'NULL':
        shutil.move(xml_path, './invalid_xml_size_null/')
        return

    img_h = img_size.getElementsByTagName("height")[0].childNodes[0].data
    img_c = img_size.getElementsByTagName("depth")[0].childNodes[0].data

    img_box = []
    try:
        for box in objects:
            cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
            x1 = int(box.getElementsByTagName("xmin")[0].childNodes[0].data)
            y1 = int(box.getElementsByTagName("ymin")[0].childNodes[0].data)
            x2 = int(box.getElementsByTagName("xmax")[0].childNodes[0].data)
            y2 = int(box.getElementsByTagName("ymax")[0].childNodes[0].data)
            # print("box:(c,xmin,ymin,xmax,ymax)", cls_name, x1, y1, x2, y2)
            img_jpg_file_name = img_xml_file + '.jpg'
            img_box.append([cls_name, x1, y1, x2, y2])
    except:
        shutil.move(xml_path, './invalid_xml/')
        return

    # 可视化方式检查标注的正确性
    # test_dataset_box_feature(img_jpg_file_name, img_box)

    save_file(img_xml_file, [img_w, img_h], img_box)


def copy_data(img_set_source, img_labels_root, imgs_source, subset_type):
    file_name = img_set_source + '/' + subset_type + ".txt"
    file_ = open(file_name)

    # 判断文件夹是否存在，不存在则创建
    root_file = Path(DATA_STORE_ROOT + DEST_IMAGES_PATH + '/' + subset_type)
    if not root_file.exists():
        os.makedirs(root_file)
        print(f'[ new dir ] => {root_file}')

    root_file = Path(DATA_STORE_ROOT + DEST_LABELS_PATH + '/' + subset_type)
    if not root_file.exists():
        os.makedirs(root_file)
        print(f'[ new dir ] => {root_file}')

    # 遍历文件夹
    for line in file_.readlines():
        img_name = line.strip('\n')
        img_sor_file = imgs_source + '/' + img_name + '.jpg'
        label_sor_file = img_labels_root + '/' + img_name + '.txt'

        # 复制图片
        DICT_DIR = DATA_STORE_ROOT + DEST_IMAGES_PATH + '/' + subset_type
        img_dict_file = DICT_DIR + '/' + img_name + '.jpg'
        copyfile(img_sor_file, img_dict_file)

        # 复制 label
        DICT_DIR = DATA_STORE_ROOT + DEST_LABELS_PATH + '/' + subset_type
        img_dict_file = DICT_DIR + '/' + img_name + '.txt'
        copyfile(label_sor_file, img_dict_file)


if __name__ == '__main__':
    # 生成标签
    # 将 VOC 的 xml 格式的标签，转换成 yolov5 的 txt 格式的标签
    # xml_file_list = os.listdir(ANNOTATIONS_PATH)

    # for file_ in xml_file_list:
    #     if file_.endswith('.xml'):
    #         get_xml_data(ANNOTATIONS_PATH, file_.replace('.xml', ''))

    # 将文件进行 train 和 val 的区分
    img_set_root = IMAGE_SET_ROOT
    imgs_root = IMAGE_PATH
    img_labels_root = LABELS_ROOT
    copy_data(img_set_root, img_labels_root, imgs_root, "train")
    copy_data(img_set_root, img_labels_root, imgs_root, "val")
    copy_data(img_set_root, img_labels_root, imgs_root, "test")
