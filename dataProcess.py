import numpy as np
import os
import cv2

#  获取颜色字典
#  labelFolder 标签文件夹,之所以遍历文件夹是因为一张标签可能不包含所有类别颜色
#  classNum 类别总数(含背景)
def color_dict(labelFolder="rF:\PG_AO\pycharm\data\train\label", classNum=2):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        # img = cv2.imread(ImagePath,-1).astype(np.uint32)
        img = cv2.imread(ImagePath, -1)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        colorDict = sorted(set(colorDict))
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        if(len(colorDict) == classNum):
            break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        color_RGB = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_RGB.append(color_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1 ,colorDict_RGB.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_RGB, colorDict_GRAY


#  数据预处理：图像归一化+标签onehot编码
#  img 图像数据
#  label 标签数据
#  classNum 类别总数(含背景)
#  colorDict_GRAY 颜色字典
def datapreprocess(img, label, classNum, colorDict_GRAY):
    #  归一化
    # img = img / 255.0
    for i in range(colorDict_GRAY.shape[0]):
        label[label == colorDict_GRAY[i][0]] = i
    #  将数据厚度扩展到classNum层
    new_label = np.zeros(label.shape + (classNum,))
    #  将平面的label的每类，都单独变成一层
    for i in range(classNum):
        new_label[label == i,i] = 1
    label = new_label
    return img, label

def dataPreprocess(img,label,labelFolder=r"F:\PG_AO\pycharm\data\train\label", classNum=2):
    colorDict_RGB, colorDict_GRAY = color_dict(labelFolder, classNum)
    img, label = datapreprocess(img, label,classNum,colorDict_GRAY)
    return img, label

# def saveResult(i, img, test_image_path="data\\test\\predict"):
#         colorDict_RGB, colorDict_GRAY = color_dict()
#         channel_max = np.argmax(img, axis = -1)
#         img_out = np.uint8(colorDict_GRAY[channel_max.astype(np.uint8)])
#         #  保存为无损压缩png
#         cv2.imwrite(test_image_path + "\\" + str(i) + ".png", img_out)