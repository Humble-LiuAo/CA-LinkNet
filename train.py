import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr=0.001, power=2):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



def train(tr_set, dv_set, model,criterion, config, device):
    n_epochs = config['n_epochs']

    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3)
    # scheduler = lr_scheduler.ExponentialLR(optimizer,gamma=0.96)
    min_loss = 1000.
    loss_record = {'train': [], 'dev': []}
    acc_record = {'train':[], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    max_acc = 0

    while epoch < n_epochs:
        tr_loss=0
        train_correct = 0
        model.train()
        for x, y in tqdm(tr_set):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device).squeeze(dim=1).long()
            pred = model(x)
            # print(pred.size(),y.size())
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.detach().cpu().item()
            predicted = F.one_hot(pred.argmax(dim=1), 2).permute(0, 3, 1, 2).detach().cpu().numpy()[:, 1, :, :]
            train_correct += (predicted == y.detach().cpu().numpy()).sum()
        train_loss = tr_loss/len(tr_set)
        loss_record['train'].append(train_loss)
        train_acc = train_correct/(len(tr_set.dataset)*256*256)
        acc_record['train'].append(train_acc)
        # scheduler.step(train_loss)
        # scheduler.step()
        # adjust_learning_rate_poly(optimizer,epoch,n_epochs)
        # train_loss = sum(loss_record['train']) / len(loss_record['train'])
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f} Acc = {train_acc:.5f}")

        val_correct = 0
        dev_loss = 0
        model.eval()
        # for x, y in tqdm(dv_set):
        for x, y in dv_set:
            x, y = x.to(device), y.to(device).squeeze(dim=1).long()
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            # acc
            predicted = F.one_hot(pred.argmax(dim=1), 2).permute(0, 3, 1, 2).detach().cpu().numpy()[:, 1, :, :]
            val_correct += (predicted == y.detach().cpu().numpy()).sum()
            dev_loss += loss.detach().cpu().item()
        dev_loss = dev_loss / len(dv_set)
        scheduler.step(dev_loss)
        val_acc = val_correct / (len(dv_set.dataset)*256*256)
        acc_record['dev'].append(val_acc)
        loss_record['dev'].append(dev_loss)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {dev_loss:.5f} Acc = {val_acc:.5f}")
        # if val_acc > max_acc:
        #     max_acc = val_acc
        #     print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {dev_loss:.5f} Acc = {val_acc:.5f}
        if dev_loss < min_loss:
            min_loss = dev_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), config['save_path'])
            print(f"[ Save Model | {epoch + 1:03d}/{n_epochs:03d} ] loss = {dev_loss:.5f}")
        else:
            early_stop_cnt += 1

        epoch += 1
        if early_stop_cnt > config['early_stop']:
            # Stop training if model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_loss, max_acc,loss_record,acc_record

# def dev(dv_set, model,criterion, device):
#     model.eval()
#     total_loss = 0
#     for x, y in dv_set:
#         x, y = x.to(device), y.to(device).squeeze().long()
#         with torch.no_grad():
#             pred = model(x)
#             loss = criterion(pred, y)
#         total_loss += loss.detach().cpu().item()
#     avg_loss = total_loss / len(dv_set)
#     return avg_loss


# def train(tr_set, model,criterion, config, device):
#     n_epochs = config['n_epochs']
#
#     optimizer = getattr(torch.optim, config['optimizer'])(
#         model.parameters(), **config['optim_hparas'])
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.1,patience=10)
#     min_loss = 1000.
#     loss_record = {'train': [], 'dev': []}
#     early_stop_cnt = 0
#     epoch = 0
#     max_acc = 0
#
#     while epoch < n_epochs:
#         train_correct = 0
#         model.train()
#         for x, y in tqdm(tr_set):
#             optimizer.zero_grad()
#             x, y = x.to(device), y.to(device).squeeze(dim=1).long()
#             pred = model(x)
#             # print(pred.size(),y.size())
#             loss = criterion(pred, y)
#             loss.backward()
#             optimizer.step()
#
#             predicted = F.one_hot(pred.argmax(dim=1), 2).permute(0, 3, 1, 2).detach().cpu().numpy()[:, 1, :, :]
#             train_correct += (predicted == y.detach().cpu().numpy()).sum()
#             loss_record['train'].append(loss.detach().cpu().item())
#
#         train_acc = train_correct/(len(tr_set.dataset)*256*256)
#         scheduler.step(loss)
#         train_loss = sum(loss_record['train']) / len(loss_record['train'])
#         print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f} Acc = {train_acc:.5f}")
#
#         if train_acc > max_acc:
#             max_acc = train_acc
#             print(f"[ SaveModel | {epoch + 1:03d}/{n_epochs:03d} ]")
#             torch.save(model.state_dict(), config['save_path'])
#             early_stop_cnt = 0
#         else:
#             early_stop_cnt += 1
#
#         epoch += 1
#         if early_stop_cnt > config['early_stop']:
#             # Stop training if model stops improving for "config['early_stop']" epochs.
#             break
#
#     print('Finished training after {} epochs'.format(epoch))
#     return min_loss, max_acc,loss_record
