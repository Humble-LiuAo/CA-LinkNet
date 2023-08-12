import torch
from dataset import SFDataset, split_train_val
import numpy as np
from torch.utils.data import Dataset
from linknet import LinkNet,LinkNetBase
from CALinkNet import CALinkNet
from CBAMLinkNet import CBAMLinkNet,CBAMLinkNet1,CBAMlinknetbase1
from resCBAMLinkNet import resCBAMLinkNetBase
from train import train
from test import test
from plot import plot_one_pred
from plot import plot_learning_curve

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
    'batch_size': 8,
    'optimizer': 'Adam',
    # use default paras
    'optim_hparas':{
      'lr': 0.001,
       # 'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'early_stop': 20,
     'save_path': 'model/resCBAMLinknetBase.pth'
}


same_seeds(422)
device = get_device()
# print(device)
train_image_path = r"data/train/image"
test_image_path = r"data/test/image"
train_sampler, valid_sampler = split_train_val(train_image_path)
train_dataset = SFDataset(train_image_path, mode="train")
val_dataset = SFDataset(train_image_path, mode="val")
test_dataset = SFDataset(test_image_path, mode="test")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
    num_workers=4, pin_memory=True
)
valid_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config['batch_size'], sampler=valid_sampler,
    num_workers=4, pin_memory=True
)

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=config['batch_size'],num_workers=8, pin_memory=True,shuffle=True
# )

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=4, pin_memory=True,shuffle=False)
# model = LinkNet().to(device)
# model = LinkNetBase().to(device)
# model = CALinkNet().to(device)
# model = CBAMLinkNet().to(device)
# model = CBAMLinkNet1().to(device)
# model = CBAMlinknetbase1().to(device)
model = resCBAMLinkNetBase().to(device)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.NLLLoss()

# start train
# pth = torch.load(config['save_path'], map_location='cpu')
# model.load_state_dict(pth)
min_loss, max_acc, loss_record = train(train_loader, valid_loader, model, criterion, config, device)
# plot_learning_curve(loss_record, title='LinkNet')
plot_learning_curve(loss_record, title='resCBAMLinkNetBase')

# plot_learning_curve(loss_record, title='CBAMLinkNet1')
# plot_learning_curve(loss_record, title='CBAMlinknetbase1')
# min_loss, max_acc, loss_record = train(train_loader, model, criterion, config, device)

# plot one
# plot_one_pred(test_dataset, config, model, device)

# test
# out_files = test_image_path.replace("image", "linknet_predict")
# MIOU, POD, FAR, CSI, HSS = test(test_loader, model,criterion, device,out_files)
# print(MIOU, POD, FAR, CSI, HSS)