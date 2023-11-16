import argparse as parser
import os
import time
from sklearn import utils
import torch
import torch.nn as nn
from os import cpu_count
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from image_dataset import imageDataset
import utils

use_gpu = torch.cuda.is_available()
parser = parser.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--img_path', default='./imgs', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--filelist_path', default='./filelist/filelist.csv', type=str)
parser.add_argument('--img_dir', default='./imgs', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoints', type=str)
parser.add_argument('--learning_rate', default=1e-4, type=float)
args = parser.parse_args()

data_transforms = transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((256, 256)),
        # # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
        # transforms.RandomResizedCrop(227),
        # 图像用于翻转
        transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
full_data = imageDataset(args.filelist_path, args.img_dir, transform=data_transforms)
val_ratio = 0.05
trn_size = int(len(full_data)*(1 - val_ratio))
val_size = len(full_data) - trn_size
trn_data, val_data = torch.utils.data.random_split(full_data, [trn_size, val_size])
while trn_data.dataset.getNumLabels() != val_data.dataset.getNumLabels():
    trn_data, val_data = torch.utils.data.random_split(full_data, [trn_size, val_size])
trn_dataLoader = DataLoader(trn_data, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count()//2)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, num_workers=cpu_count())
dataloaders = {'trn': trn_dataLoader, 'val': val_dataloader}
dataset_sizes = {'trn': trn_data.__len__(), 'val': val_data.__len__()}
# print(f'trn data set feature:\namount of data lines: {trn_data.dataset.__len__()}\namount of labels: {trn_data.dataset.getNumLabels()}')
# print(f'val data set feature:\namount of data lines: {val_data.dataset.__len__()}\namount of labels: {val_data.dataset.getNumLabels()}')

def print_runningloss(epoch, running_loss, running_precision, running_recall, batch_num, phase):

    epoch_loss = running_loss / dataset_sizes[phase]
    print('{} Loss: {:.4f} '.format(phase, epoch_loss))
    epoch_precision = running_precision / batch_num
    print('{} Precision: {:.4f} '.format(phase, epoch_precision))
    epoch_recall = running_recall / batch_num
    print('{} Recall: {:.4f} '.format(phase, epoch_recall))
    with open(os.path.join(args.checkpoint_path, 'losslog.txt'), 'a+', encoding='utf8') as f:
        f.write(f'epoch: {epoch}\tloss: {epoch_loss}\tprecision: {epoch_precision}\trecall: {epoch_recall}\tphase: {phase}\n')
    
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, starting_epoch = 0):
    sigmoid = nn.Sigmoid()
    since = time.time()
        
    for epoch in range(starting_epoch, num_epochs):        
        phase = 'trn'
        if epoch % 5 == 0:
            phase = 'val'
        
        if phase:
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0
            print(f'\n---- starting Epoch {epoch + 1}/{num_epochs} ----')
            model.train()
            for data in dataloaders['trn']:
                # print('\r', f'loading batch {batch_num + 1}', flush=True, end='')
                # print()
                inputs, labels = data
                
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(sigmoid(outputs), labels)
                precision, recall = calculate_acc(sigmoid(outputs), labels)
                running_precision += precision
                running_recall += recall
                batch_num += 1
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item() * inputs.size(0)
            print_runningloss(epoch, running_loss, running_precision, running_recall, batch_num, 'trn')
                
        if phase == 'val':
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0
            with torch.no_grad():
                model.eval()
                for data in dataloaders['val']:
                    # print('\r', f'loading batch {batch_num + 1}', flush=True, end='')
                    # print()
                    inputs, labels = data
                    
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    outputs = model(inputs)
                    loss = criterion(sigmoid(outputs), labels)
                    running_loss += loss.item() * inputs.size(0)
                    precision, recall = calculate_acc(sigmoid(outputs), labels)
                    running_precision += precision
                    running_recall += recall
                    batch_num += 1
            print_runningloss(epoch, running_loss, running_precision, running_recall, batch_num, 'val')
            # torch.save(model.state_dict(), f'{args.checkpoint_path} + Epoch_{epoch}.pth')
            utils.save_checkpoint(model, optimizer, epoch, args.checkpoint_path)
    time_elapsed = time.time() -since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
def calculate_acc(model_pred, labels):
    precision = 0
    recall = 0
    tops = torch.sum(labels, dim=1)
    batch_size = model_pred.shape[0]
    for bat in range(batch_size):
        top = tops[bat]
        pred_label_locate = torch.argsort(model_pred, descending=True)[:, :int(top)]
        tmp_label = torch.zeros(1, model_pred.shape[1]).cuda()
        tmp_label[0, pred_label_locate[bat]] = 1
        target_num = torch.sum(labels[bat])
        pred_num_true = torch.sum(tmp_label * labels[bat])
        precision += pred_num_true / top
        recall += pred_num_true / target_num
    return precision / batch_size, recall / batch_size

def main():
    num_labels = trn_data.dataset.getNumLabels()
    model = None
    starting_epoch = 0
    try:
        saved_model, iteration, _, _ = utils.load_checkpoint(utils.get_latest_checkpoint(args.checkpoint, 'cv_*.pth'))
        model = saved_model
        starting_epoch = iteration
        print(f'successfully loading checkpoint cv_{iteration}.pth')
    except:
        model = models.resnet50(weights='DEFAULT')
    in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features=in_features, out_features=num_labels)
    model.fc = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_labels),
        nn.LogSoftmax(dim=1)
    )
    if use_gpu:
        model = model.cuda()
        
    criterion = nn.BCELoss()
    
    # fc_params = list(map(id, model.fc.parameters()))
    # base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    # params = [
    #     {"params": base_params, "lr": 1e-4},
    #     {"params": model.fc.parameters(), "lr": 1e-3},
    # ]
    optimizer_ft = torch.optim.AdamW(
        model.parameters(),
        args.learning_rate,
        eps=1e-9,
        betas=[0.8, 0.99],
        )
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.num_epochs, starting_epoch=starting_epoch)
    
if __name__ == '__main__':
    main()