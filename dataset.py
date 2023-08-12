import torch
from torch.utils.data import Dataset, SubsetRandomSampler
import os
import torchvision.transforms as transforms
from osgeo import gdal
import numpy as np
import cv2
from dataProcess import dataPreprocess
import random
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# def per_image_standardization(image):
#     for i in range(image.shape[2]):
#         image[:, :, i] = (image[:, :, i]-image[:, :, i].mean())/image[:, :, i].std()
#     return image

def image_normalization(image,
                        max,
                        min):
    for i in range(image.shape[2]):
        image[:, :, i] = (image[:, :, i] - min[i]) / max[i] - min[i]
    return image

def image_standardization(image,
                          mean=None,
                          std=None
                          ):
    if std is None:
        std =[1.35797411e-01, 1.31497800e-01, 1.35461301e-01, 1.51400790e-01,
       1.80659276e-02, 1.10513717e-01, 8.07879418e-02, 1.06412984e-01,
       9.41269621e-02, 9.03227776e-02, 1.05267353e-01, 1.13237433e-01,
       9.98796076e-02, 1.00964159e-01, 1.08206309e-01, 1.23046890e-01,
       7.82336369e-02, 9.59569141e-02, 1.48895994e-01, 5.86562691e+01,
       5.52726402e+01, 4.91007690e+01, 5.37844200e+01, 5.42284355e+01,
       5.41764412e+01]
    if mean is None:
        mean = [3.4621206e-01, 3.0913228e-01, 2.9971826e-01, 3.1513056e-01,
       2.6682984e-02, 2.1294320e-01, 1.5268587e-01, 3.2787722e-01,
       2.8333017e-01, 2.5683394e-01, 2.6213261e-01, 2.6663908e-01,
       2.3302065e-01, 2.3397382e-01, 2.4636807e-01, 2.4829420e-01,
       1.5285505e-01, 1.8827960e-01, 3.0575281e-01, 2.4222867e+02,
       2.3157816e+02, 2.0937877e+02, 2.2441699e+02, 2.2539178e+02,
       2.2506474e+02]
    for i in range(image.shape[2]):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    return image

def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

class SFDataset(Dataset):
    def __init__(self, input_root, mode="train"):
        super().__init__()
        self.input_root = input_root
        self.mode = mode
        self.input_ids = sorted(img for img in os.listdir(self.input_root))
        # p1 = random.randint(0, 1)
        # p2 = random.randint(0, 1)
        self.train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomRotation((0,180)),
            # transforms.RandomHorizontalFlip(p1),
            # transforms.RandomVerticalFlip(p2),
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize([], [])
        ])
        self.test_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # at this point all transformations are applied and we expect to work with raw tensors
        imageName = os.path.join(self.input_root, self.input_ids[idx])
        # 3 band
        # image = np.array(np.transpose(readTif(imageName)[[2,15,24],:,:],[1,2,0]), dtype=np.float32)
        # 25 band
        image = np.array(np.transpose(readTif(imageName),[1,2,0]), dtype=np.float32)
        image = image_standardization(image)
        mask = np.array(cv2.imread(imageName.replace("image", "label"),-1),dtype=int)
        # image,mask = dataPreprocess(image,mask)
        # image = per_image_standardization(image)
        # image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        if self.mode == "train":
            # image, mask = self.transform(image, mask)
            # seed = np.random.randint(2147483647)
            # np.random.seed(seed)
            return self.train_transform(image), self.train_transform(mask)
        else:
            return self.test_transform(image), self.test_transform(mask)


# 划分训练集和验证集
def split_train_val(input_img_folder="./data/train/image"):
    # Get correct indices
    num_train = len(sorted(img for img in os.listdir(input_img_folder)))
    indices = list(range(num_train))
    random.seed(422)
    indices = random.sample(indices, len(indices))
    split = int(np.floor(0.2 * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    # set up datasets
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


# def save_test(path, model, device, prename="linknetpred"):
#     pred_path = path.replace("image", prename)
#     if not os.path.exists(pred_path):
#         os.mkdir(pred_path)
#     model.eval()
#     for _img in os.listdir(path):
#         img_path = os.path.join(path, _img)
#         img = torch.from_numpy(np.array(np.transpose(readTif(img_path),[1,2,0]), dtype=np.float32))
#         img = img.unsqueeze(dim=0).to(device)
#         with torch.no_grad():
#             pred = model(img)
#             pred = F.one_hot(pred.argmax(dim=1), 2).permute(0, 3, 1, 2)
#             pred = pred.detach().cpu().numpy().squeeze()[1, :, :]
#             pred_file = img_path.replace("image", prename)
#             cv2.imwrite(pred_file,pred*255)