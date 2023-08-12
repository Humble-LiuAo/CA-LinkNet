import torch
from dataset import SFDataset, split_train_val
import numpy as np
from torch.utils.data import Dataset
from linknet import LinkNet,LinkNetBase
from CALinkNet import CALinkNet
from CBAMLinkNet import CBAMLinkNet,CBAMLinkNet1,CBAMlinknetbase1
from resCBAMLinkNet import resCBAMLinkNetBase, resCBAMLinkNetBaseCBAM
from train import train
from test import test
from plot import plot_one_pred
from plot import plot_learning_curve
from focalloss import FocalLoss_Ori
from predict import predict

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    ##确保输入固定时，输出不变
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def get_device():
    '''Get device'''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


config = {
    'n_epochs': 150,
    'batch_size': 16,
    'optimizer': 'Adam',
    # use default paras
    'optim_hparas':{
      'lr': 0.001,
       # 'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'early_stop': 10,
     'save_path': 'model/linknetca0.pth'
}

if __name__ == '__main__':
    same_seeds(422)
    device = get_device()
    # print(device)
    train_image_path = r"F:\PG_AO\data0.2\82\train\image"
    test_image_path = r"F:\PG_AO\data0.2\82\valid\image"
    valid_image_path = r"F:\PG_AO\data0.2\82\valid\image"
    # train_sampler, valid_sampler = split_train_val(train_image_path)
    train_dataset = SFDataset(train_image_path, mode="train")
    val_dataset = SFDataset(valid_image_path, mode="val")
    test_dataset = SFDataset(test_image_path, mode="test")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=2, pin_memory=False
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=2, pin_memory=False
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=config['batch_size'],num_workers=8, pin_memory=True,shuffle=True
    # )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=113, num_workers=2, pin_memory=True,shuffle=False)
    model = CALinkNet().to(device)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss_Ori()

    # start train
    # min_loss, max_acc, loss_record, acc_record = train(train_loader, valid_loader, model, criterion, config, device)
    # plot_learning_curve(loss_record, acc_record, title='LinkNetCA')


    # plot one
    # plot_one_pred(test_dataset, config, model, device)

    # test
    MIOU, POD, FAR, CSI, HSS=test(test_loader, model,criterion, device,config)
    print(MIOU, POD, FAR, CSI, HSS)

    # predict
    # savepath = r"F:\PG_AO\data0.2\82\LinkNet\data\predict"
    # predict(model, config, device, savepath)