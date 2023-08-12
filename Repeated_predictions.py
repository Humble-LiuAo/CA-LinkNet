from osgeo import gdal
import numpy as np
import math
import torch
import cv2
from torchvision import transforms as T
from resCBAMLinkNet import resCBAMLinkNetBase
import torch.nn.functional as F
from dataset import image_standardization
import cv2
from CBAMLinkNet import CBAMLinkNet1
from CALinkNet import CALinkNet
from linknet import LinkNet


# 读取tif数据集
def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)

    return data


# 保存tif文件函数
def writeTiff(fileName, data, im_geotrans=(0, 0, 0, 0, 0, 0), im_proj=""):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        im_bands, im_height, im_width = data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset


#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (256 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (256 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (256 - SideLength * 2): i * (256 - SideLength * 2) + 256,
                      j * (256 - SideLength * 2): j * (256 - SideLength * 2) + 256]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (256 - SideLength * 2): i * (256 - SideLength * 2) + 256,
                  (img.shape[1] - 256): img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 256): img.shape[0],
                  j * (256 - SideLength * 2): j * (256 - SideLength * 2) + 256]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 256): img.shape[0],
              (img.shape[1] - 256): img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver


#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0
    for i, img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if (i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 256 - RepetitiveLength, 0: 256 - RepetitiveLength] = img[0: 256 - RepetitiveLength,
                                                                               0: 256 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                #  原来错误的
                # result[shape[0] - ColumnOver : shape[0], 0 : 256 - RepetitiveLength] = img[0 : ColumnOver, 0 : 256 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0: 256 - RepetitiveLength] = img[
                                                                                                        256 - ColumnOver - RepetitiveLength: 256,
                                                                                                        0: 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            256 - 2 * RepetitiveLength) + RepetitiveLength,
                0:256 - RepetitiveLength] = img[RepetitiveLength: 256 - RepetitiveLength, 0: 256 - RepetitiveLength]
                #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif (i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 256 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0: 256 - RepetitiveLength,
                                                                                  256 - RowOver: 256]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0], shape[1] - RowOver: shape[1]] = img[256 - ColumnOver: 256,
                                                                                        256 - RowOver: 256]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            256 - 2 * RepetitiveLength) + RepetitiveLength,
                shape[1] - RowOver: shape[1]] = img[RepetitiveLength: 256 - RepetitiveLength, 256 - RowOver: 256]
                #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if (j == 0):
                result[0: 256 - RepetitiveLength,
                (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[0: 256 - RepetitiveLength, RepetitiveLength: 256 - RepetitiveLength]
                #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if (j == len(TifArray) - 1):
                result[shape[0] - ColumnOver: shape[0],
                (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                ] = img[256 - ColumnOver: 256, RepetitiveLength: 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength: (j + 1) * (
                            256 - 2 * RepetitiveLength) + RepetitiveLength,
                (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength: (i - j * len(
                    TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                ] = img[RepetitiveLength: 256 - RepetitiveLength, RepetitiveLength: 256 - RepetitiveLength]
    return result


area_perc = 0.02
TifPath = r"F:\PG_AO\pre\mask_rb.tif"
model_paths = [
    r'.\model\linknetca.pth'
]
ResultPath = r"F:\PG_AO\pre\dpca_tag.tif"
RepetitiveLength = int((1 - math.sqrt(area_perc)) * 256 / 2)

big_image = readTif(TifPath)
big_image = big_image.swapaxes(1, 0).swapaxes(1, 2)
big_image = image_standardization(big_image)

# big_image = cv2.imread(TifPath, cv2.IMREAD_UNCHANGED)

TifArray, RowOver, ColumnOver = TifCroppingArray(big_image, RepetitiveLength)

trfm = T.Compose([
    T.ToTensor(),
])

# 改成自己的model即可
model = CALinkNet()
# model = LinkNet()
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)

predicts = []
for i in range(len(TifArray)):
    for j in range(len(TifArray[0])):
        image = TifArray[i][j]
        image = trfm(image)
        image = image.to(DEVICE)
        image = image.unsqueeze(0)
        pred = np.zeros((1, 1, 256, 256))
        for model_path in model_paths:
            pth = torch.load(model_path, map_location='cpu')
            model.load_state_dict(pth)
            model.eval()

            with torch.no_grad():
                pred1 = model(image)
                pred1 = F.softmax(pred1,dim=1)[:,1,:,:].unsqueeze(1)
                pred1 = pred1.cpu().numpy()


                pred2 = model(torch.flip(image, [0, 3]))
                pred2 = F.softmax(pred2, dim=1)[:,1,:,:].unsqueeze(1)
                pred2 = torch.flip(pred2, [3, 0]).cpu().numpy()

                pred3 = model(torch.flip(image, [0, 2]))
                pred3 = F.softmax(pred3, dim=1)[:,1,:,:].unsqueeze(1)
                pred3 = torch.flip(pred3, [2, 0]).cpu().numpy()

                pred += pred1 + pred2 + pred3

        pred = pred / (len(model_paths) * 3)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred.astype(np.uint8)
        pred = pred.reshape((256, 256))
        predicts.append((pred))

# 保存结果predictspredicts
result_shape = (big_image.shape[0], big_image.shape[1])
result_data = Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)
# writeTiff(ResultPath, result_data*255)
cv2.imwrite(ResultPath[:-4]+".png",result_data*255)

def get_Tiff(path,data,ME_tif):
    im_geotrans,im_proj =  gdal.Open(ME_tif, gdal.GA_ReadOnly).GetGeoTransform(),gdal.Open(ME_tif, gdal.GA_ReadOnly).GetProjection()
    writeTiff(path,data,im_geotrans,im_proj)

get_Tiff(ResultPath,result_data,TifPath)