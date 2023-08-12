# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:11:00 2022

@author: idm
"""

import random
from shutil import copy2
import os


def data_set_split(src_data_folder, target_data_folder, train_scale=0.6, valid_scale=0.2, test_scale=0.2):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹 D:\SeaFog\seafog_data\
    :param target_data_folder: 目标文件夹 D:\pycode\Pycharm\data
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    # 在目标目录下创建文件夹
    split_names = ['train', 'valid', 'test']
    #    split_names = ['train','valid']
    subfold_names = ['image', 'label']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        for subfold_name in subfold_names:
            subfold_path = os.path.join(split_path, subfold_name)
            if os.path.isdir(subfold_path):
                pass
            else:
                os.mkdir(subfold_path)

    # 按照比例划分数据集，并进行数据图片标签的复制
    train_folder = os.path.join(os.path.join(target_data_folder, 'train'), subfold_names[0])
    train_label_folder = os.path.join(os.path.join(target_data_folder, 'train'), subfold_names[1])
    valid_folder = os.path.join(os.path.join(target_data_folder, 'valid'), subfold_names[0])
    valid_label_folder = os.path.join(os.path.join(target_data_folder, 'valid'), subfold_names[1])
    test_folder = os.path.join(os.path.join(target_data_folder, 'test'), subfold_names[0])
    test_label_folder = os.path.join(os.path.join(target_data_folder, 'test'), subfold_names[1])
    image_srcpath = os.path.join(src_data_folder, subfold_names[0])
    label_srcpath = os.path.join(src_data_folder, subfold_names[1])
    current_all_data = os.listdir(image_srcpath)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.seed(507422)
    random.shuffle(current_data_index_list)
    train_stop_flag = current_data_length * train_scale
    valid_stop_flag = current_data_length * (train_scale + valid_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        src_img_path = os.path.join(image_srcpath, current_all_data[i])
        src_label_path = os.path.join(label_srcpath, current_all_data[i])
        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_folder)
            copy2(src_label_path, train_label_folder)
            # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1
        elif current_idx <= valid_stop_flag:
            copy2(src_img_path, valid_folder)
            copy2(src_label_path, valid_label_folder)
            #            print("{}复制到了{}".format(src_img_path, train_folder))
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_folder)
            copy2(src_label_path, test_label_folder)
            # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1
        current_idx = current_idx + 1

    print("训练集{}：{}张".format(train_folder, train_num))
    print("训练集{}：{}张".format(valid_folder, val_num))
    print("测试集{}：{}张".format(test_folder, test_num))


data_set_split(r"F:\PG_AO\31\data31\raw", r"F:\PG_AO\31\data31")
