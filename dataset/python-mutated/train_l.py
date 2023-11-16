import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import sys
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model_compat import DSN
from data_loader import GetLoader
from functions import SIMSE, DiffLoss, MSE, DiffLoss_tfTrans
from test import test
from log.logger import Logger
import time

def run(net_str):
    if False:
        return 10
    net_str = os.path.join('D:\\study\\graduation_project\\grdaution_project\\instru_identify\\dataset18dataset2', net_str)
    source_image_root = os.path.join('D:\\', 'study', 'graduation_project', 'grdaution_project', 'instru_identify', 'dataset', 'dataset1')
    target_image_root = os.path.join('D:\\', 'study', 'graduation_project', 'grdaution_project', 'instru_identify', 'dataset', 'dataset2')
    target = 'dataset2'
    p = str(8)
    model_root = 'dataset1' + p + 'dataset2'
    if not os.path.exists(model_root):
        os.mkdir(model_root)
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    log_path = os.path.join(model_root, 'train.txt')
    sys.stdout = Logger(log_path)
    cuda = False
    cudnn.benchmark = True
    lr = 0.01
    batch_size = 16
    image_size = 28
    n_epoch = 1
    step_decay_weight = 0.95
    lr_decay_step = 20000
    active_domain_loss_step = 10000
    weight_decay = 1e-06
    alpha_weight = 0.01
    beta_weight = 0.075
    gamma_weight = 0.25
    momentum = 0.9
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    img_transform_source = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    img_transform_target = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    source_list = os.path.join(source_image_root, 'dataset1_train_labels.txt')
    dataset_source = GetLoader(data_root=os.path.join(source_image_root, 'dataset1_train'), data_list=source_list, transform=img_transform_target)
    dataloader_source = torch.utils.data.DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=0)
    target_list = os.path.join(target_image_root, 'dataset2_train_labels.txt')
    dataset_target = GetLoader(data_root=os.path.join(target_image_root, 'dataset2_train'), data_list=target_list, transform=img_transform_target)
    dataloader_target = torch.utils.data.DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=0)
    my_net = DSN()
    my_net.load_state_dict(torch.load(net_str))

    def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):
        if False:
            return 10
        current_lr = init_lr * step_decay_weight ** (step / lr_decay_step)
        if step % lr_decay_step == 0:
            print('learning rate is set to %f' % current_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        return optimizer
    optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_classfication = torch.nn.CrossEntropyLoss()
    loss_recon1 = MSE()
    loss_recon2 = SIMSE()
    loss_diff = DiffLoss_tfTrans()
    loss_similarity = torch.nn.CrossEntropyLoss()
    if cuda:
        my_net = my_net.cuda()
        loss_classification = loss_classification.cuda()
        loss_recon1 = loss_recon1.cuda()
        loss_recon2 = loss_recon2.cuda()
        loss_diff = loss_diff.cuda()
        loss_similarity = loss_similarity.cuda()
    for p in my_net.parameters():
        p.requires_grad = True
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)
    current_step = 0
    accu_total1 = 0
    accu_total2 = 0
    time_total1 = 0
    time_total2 = 0
    for epoch in range(n_epoch):
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
        i = 0
        while i < len_dataloader:
            data_target = data_target_iter.next()
            (t_img, t_label) = data_target
            my_net.zero_grad()
            loss = 0
            batch_size = len(t_label)
            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()
            if cuda:
                t_img = t_img.cuda()
                t_label = t_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()
            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)
            target_inputv_img = Variable(input_img)
            target_classv_label = Variable(class_label)
            target_domainv_label = Variable(domain_label)
            if current_step > active_domain_loss_step:
                p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
                p = 2.0 / (1.0 + np.exp(-10 * p)) - 1
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', p=p)
                (target_private_coda, target_share_coda, target_domain_label, target_rec_code) = result
                target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)
                loss += target_dann
            else:
                if cuda:
                    target_dann = Variable(torch.zeros(1).float().cuda())
                else:
                    target_dann = Variable(torch.zeros(1).float())
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all')
                (target_private_coda, target_share_coda, _, target_rec_code) = result
                target_diff = beta_weight * loss_diff(target_private_coda, target_share_coda, weight=0.05)
                loss += target_diff
                target_mse = alpha_weight * loss_recon1(target_rec_code, target_inputv_img)
                loss += target_mse
                target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
                loss += target_mse
                loss.backward()
                optimizer.step()
                data_source = data_source_iter.next()
                (s_img, s_label) = data_source
                my_net.zero_grad()
                batch_size = len(s_label)
                input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
                class_label = torch.LongTensor(batch_size)
                domain_label = torch.zeros(batch_size)
                damain_label = domain_label.long()
                loss = 0
                if cuda:
                    s_img = s_img.cuda()
                    s_label = s_label.cuda()
                    input_img = input_img.cuda()
                    class_label = class_label.cuda()
                    domain_label = domain_label.cuda()
                input_img.resize_as_(input_img).copy_(s_img)
                class_label.resize_as_(s_label).copy_(s_label)
                source_inputv_img = Variable(input_img)
                source_classv_label = Variable(class_label)
                source_domainv_label = Variable(domain_label)
                if current_step > active_domain_loss_step:
                    result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', p=p)
                    (source_private_code, source_share_code, source_domain_label, source_classv_label, source_rec_code) = result
                    source_dann = gamma_weight * loss_similarity(source_domain_label, source_classv_label)
                    loss += source_dann
                else:
                    if cuda:
                        source_dann = Variable(torch.zeros(1).float().cuda())
                    else:
                        if cuda:
                            source_dann = Variable(torch.zeros(1).float().cuda())
                        else:
                            source_dann = Variable(torch.zeros(1).float())
                        result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all')
                        (source_private_code, source_share_code, _, source_class_label, source_rec_code) = result
                    source_classification = loss_classfication(source_class_label, source_classv_label)
                    loss += source_classification
                    source_diff = beta_weight * loss_diff(source_private_code, source_share_code, weight=0.05)
                    loss += source_diff
                    source_mse = alpha_weight * loss_recon1(source_rec_code, source_inputv_img)
                    loss += source_mse
                    source_simse = gamma_weight * loss_recon2(source_rec_code, source_inputv_img)
                    loss += source_simse
                    loss.backward()
                    optimizer.step()
                    i += 1
                    current_step += 1
                    start1 = time.time()
                    accu1 = test(epoch=epoch, name='dataset1')
                    end1 = time.time()
                    curr1 = end1 - start1
                    time_total1 += curr1
                    accu_total1 += accu1
                    start2 = time.time()
                    accu2 = test(epoch=epoch, name='dataset2')
                    end2 = time.time()
                    curr2 = end2 - start2
                    time_total2 += curr2
                    accu_total2 += accu2
    model_index = epoch
    model_path = 'D:\\study\\graduation_project\\grdaution_project\\instru_identify\\dataset18dataset2' + '\\dsn_epoch_' + str(model_index) + '.pth'
    while os.path.exists(model_path):
        model_index = model_index + 1
        model_path = 'D:\\study\\graduation_project\\grdaution_project\\instru_identify\\dataset18dataset2' + '\\dsn_epoch_' + str(model_index) + '.pth'
    torch.save(my_net.state_dict(), model_path)
    average_accu1 = accu_total1 / (len_dataloader * n_epoch)
    average_accu2 = accu_total2 / (len_dataloader * n_epoch)
    print(round(float(average_accu1), 3))
    print(round(float(average_accu2), 3))
    print(round(float(time_total1), 3))
    print(round(float(time_total2), 3))
    return result
if __name__ == '__main__':
    nett_str = sys.argv[1]
    run(nett_str)