import torchvision.transforms as transforms
import numpy as np
import torch
import os
from dataset import readTif, image_standardization
import torch.nn.functional as F
import cv2

def predict(model,config,device,savepath,path=r"F:\PG_AO\data0.2\82\valid\image"):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    pth = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(pth)
    model.eval()
    imgs = os.listdir(path)
    for file in imgs:
        imageName = os.path.join(path,file)
        image = np.array(np.transpose(readTif(imageName), [1, 2, 0]), dtype=np.float32)
        image = image_standardization(image)
        #        model predict
        prep_data = transforms.Compose([
           transforms.ToTensor(),
           ])
        input_image = prep_data(image)
        input_image = input_image.unsqueeze(0).to(device)
        y = model(torch.autograd.Variable(input_image))
        pred = F.one_hot(y.argmax(dim=1), 2).permute(0, 3, 1, 2).detach().cpu().numpy()[:, 1, :, :]
        predict = np.squeeze(pred).astype(np.uint8)
        # save
        pred_file  = os.path.join(savepath,file[:-4]+'.png')
        cv2.imwrite(pred_file, predict * 255)
