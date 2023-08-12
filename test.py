import torch
from tqdm import tqdm
from evaluator import Evaluator
import numpy as np
import torch.nn.functional as F
import cv2
import os

def test(ts_set, model,criterion, device,config):
    pth = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(pth)
    model.eval()
    total_loss = 0
    acc = 0
    iou = 0
    miou = 0
    POD, FAR, CSI, HSS = 0,0,0,0
    evaluator = Evaluator()
    for i,(x, y) in tqdm(enumerate(ts_set)):
        x, y = x.to(device), y.to(device).squeeze(dim=1).long()
        with torch.no_grad():
            pred = model(x)
            loss = criterion(pred, y)
            pred = F.one_hot(pred.argmax(dim=1), 2).permute(0, 3, 1, 2).detach().cpu().numpy()[:, 1, :, :]
            y = y.detach().cpu().numpy()
        evaluator.add_batch(y,pred)
        total_loss += loss.detach().cpu().item()
        acc += evaluator.Pixel_Accuracy()
        iou += evaluator.Intersection_over_Union()
        miou += evaluator.Mean_Intersection_over_Union()
        pod,far,csi,hss= evaluator.POD()
        POD += pod
        FAR += far
        CSI += csi
        HSS += hss
        evaluator.reset()
    avg_loss = total_loss / len(ts_set)
    avg_acc = acc/len(ts_set)
    avg_iou = iou/len(ts_set)
    avg_miou = miou/len(ts_set)
    avg_POD, avg_FAR, avg_CSI, avg_HSS = POD/len(ts_set), FAR/len(ts_set), CSI/len(ts_set), HSS/len(ts_set)
    # return avg_loss, avg_acc, avg_iou, avg_miou
    return avg_miou, avg_POD, avg_FAR, avg_CSI, avg_HSS

