# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 19:02:45 2022

@author: idm
"""

import random
import os


def file_rename_shuffle(path, index=True):
    """
    @param path: 文件夹路径
    """
    fileList = os.listdir(path)  # 获得所有文件名列表，可以print(fileList)查看
    Imgnum = len(fileList)
    print(Imgnum)
    i = 0
    L = random.sample(range(0, Imgnum), Imgnum)
    print(L)
    filetype = ".tif"  # 文件类型
    for filename in fileList:
        #        print(filename)
        portion = os.path.splitext(filename)  # 将文件名拆成名字和后缀
        if portion[1] == filetype:  # 检查文件的后缀
            if index:
                newname = 'f' + str(L[i]) + filetype
            else:
                newname = str(L[i]) + filetype
            #            print(newname)
            os.rename(path + "\\" + filename, path + "\\" + newname)  # 修改名称
            os.rename(path.replace("image", "label") + "\\" + filename, path.replace("image", "label") + "\\" + newname)
            i += 1


path = r'F:\PG_AO\data0.2\82\raw\image'
file_rename_shuffle(path,True)
file_rename_shuffle(path,False)
# file_rename_shuffle(path.replace("train","valid"),True)
# file_rename_shuffle(path.replace("train","valid"),False)
# file_rename_shuffle(path.replace("train","test"),True)
# file_rename_shuffle(path.replace("train","test"),False)