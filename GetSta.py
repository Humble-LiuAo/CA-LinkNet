import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from osgeo import gdal
import numpy as np
import os
import cv2

def readTif(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
    return GdalImg_data

class SDataset(Dataset):
    def __init__(self, input_root, mode="train"):
        super().__init__()
        self.input_root = input_root
        self.mode = mode
        self.input_ids = sorted(img for img in os.listdir(self.input_root))
        self.train_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomRotation((0,180)),
            # transforms.RandomHorizontalFlip(p1),
            # transforms.RandomVerticalFlip(p2),
            # transforms.Resize((256, 256)),
            # transforms.ToTensor(),
            # transforms.Normalize([], [])
        ])
        self.test_transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((256, 256)),
            # transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # at this point all transformations are applied and we expect to work with raw tensors
        imageName = os.path.join(self.input_root, self.input_ids[idx])
        # 3 band
        image = np.array(readTif(imageName), dtype=np.float32)
        mask = np.array(cv2.imread(imageName.replace("image", "label"),-1),dtype=int)
        # image,mask = dataPreprocess(image,mask)
        # image, mask = torch.from_numpy(image), torch.from_numpy(mask)
        if self.mode == "train":
            # image, mask = self.transform(image, mask)
            # seed = np.random.randint(2147483647)
            # np.random.seed(seed)
            return self.train_transform(image), self.train_transform(mask)
        else:
            return self.test_transform(image), self.test_transform(mask)

def getStat(train_data):
    '''    Compute mean and variance for training data
    param train_data: 自定义类Dataset(或ImageFolder即可)
    return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    mean = torch.zeros(25)
    std = torch.zeros(25)
    for X, _ in train_loader:
        for d in range(25):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    with open("Satdata.txt", "w") as f:
        f.write("mean: ")
        f.write('\n')
        f.writelines(str(mean.numpy()))
        f.write('\n')
        f.write("std :")
        f.write('\n')
        f.writelines(str(std.numpy()))
    # return list(mean.numpy()), list(std.numpy())
    return mean.numpy(), std.numpy()
def get_stat():
    train_dataset = SDataset(r"F:\PG_AO\data0.2\82\raw\image")
    return getStat(train_dataset)

print(get_stat())