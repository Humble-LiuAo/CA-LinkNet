import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

def plot_learning_curve(loss_record, acc_record,title=''):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[ : :len(loss_record['train']) // len(loss_record['dev'])]
    # plt.figure(figsize=(6, 4))
    # # Case-insensitive Tableau Colors from 'T10' categorical palette.
    # plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    # plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='valid')
    y_max = max(loss_record['train'])
    # plt.ylim(0.0, y_max+0.005)
    # plt.xlabel('Training epochs')
    # plt.ylabel('Loss')
    # plt.title('Learning curve of {}'.format(title))
    # # Place a legend on the Axes.
    # plt.legend()
    # plt.savefig(title + ".png")
    # plt.show()

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x_1, loss_record['train'], c='tab:red', label='train')
    ax1.plot(x_2, loss_record['dev'], c='tab:cyan', label='valid')
    ax1.set_xlabel('Training epochs')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0.0, y_max+0.005)
    # ax1.legend()

    ax2 = ax1.twinx()  # 创建共用x轴的第二个y轴
    ax2.plot(x_1, acc_record['train'], c='tab:red', label='train')
    ax2.plot(x_2, acc_record['dev'], c='tab:cyan', label='valid')
    ax2.set_xlabel('Training epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(top=1.0)
    # ax2.legend()

    fig.tight_layout()
    plt.savefig(title + ".png")
    plt.show()


def plot_img_and_mask(img, y, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 2)
    ax[0].set_title('Input image')
    # ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
        ax[2].set_title(f'Output True')
        ax[2].imshow(y)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_one_pred(ts_set,config,model,device):
    test_loader = torch.utils.data.DataLoader(ts_set, batch_size=1
                                              , num_workers=1,
                                              pin_memory=True)
    # model = model.to(device)
    # pth = torch.load(config['save_path'], map_location='cpu')
    # model.load_state_dict(pth)
    for i, (x,y) in enumerate(test_loader):
        if i == 32:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                pred = F.one_hot(pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
            pred = pred.detach().cpu().squeeze()[1,:,:]
            y = y.detach().cpu().squeeze()
            x = x.permute(0, 2, 3, 1).detach().cpu().squeeze()
            plot_img_and_mask(x, y, pred)