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
#from mobilenet_model_compat import DSN
#from vgg_compat import DSN
from log.logger import Logger
import time


######################
# params             #
######################
def run(net_str):
    # execute only if run as the entry point into the program
    # 定义源域和当前目标域
    net_str = os.path.join('D:\study\graduation_project\grdaution_project\instru_identify\dataset18dataset2', net_str)
    source_image_root = os.path.join('D:\\', 'study', 'graduation_project', 'grdaution_project', 'instru_identify',
                                     'dataset', 'dataset1')
    target_image_root = os.path.join('D:\\', 'study', 'graduation_project', 'grdaution_project', 'instru_identify',
                                     'dataset', 'dataset2')

    target = 'dataset2'

    # 选取历史数据的比例
    p = str(8)
    # 模型保存路径
    model_root = 'dataset1'+p+'dataset2'
    if not os.path.exists(model_root):
        os.mkdir(model_root)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # 训练日志保存
    log_path = os.path.join(model_root, 'train.txt')
    sys.stdout = Logger(log_path)

    # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    # 训练参数定义
    cuda = False
    cudnn.benchmark = True
    lr = 1e-2
    batch_size = 16
    image_size = 28
    n_epoch = 1
    step_decay_weight = 0.95
    lr_decay_step = 20000
    active_domain_loss_step = 10000
    weight_decay = 1e-6
    alpha_weight = 0.01
    beta_weight = 0.075
    gamma_weight = 0.25
    momentum = 0.9

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    #######################
    # load data           #
    #######################

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 源域数据加载
    source_list = os.path.join(source_image_root, 'dataset1_train_labels.txt')
    dataset_source = GetLoader(
        data_root=os.path.join(source_image_root, 'dataset1_train'),
        data_list=source_list,
        transform=img_transform_target,
    )

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dataset_source,
        batch_size=batch_size,
        shuffle=True,  # 随机数种子
        num_workers=0  # 进程数
    )

    # 目标域数据加载
    target_list = os.path.join(target_image_root, 'dataset2_train_labels.txt')
    dataset_target = GetLoader(
        data_root=os.path.join(target_image_root, 'dataset2_train'),
        data_list=target_list,
        transform=img_transform_target,
    )

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 单进程加载
    )


    #####################
    #  load model       #
    #####################

    my_net = DSN()
    my_net.load_state_dict(torch.load(net_str))

    #####################
    # setup optimizer   #
    #####################


    def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

        # Decay learning rate by a factor of step_decay_weight every lr_decay_step
        current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

        if step % lr_decay_step == 0:
            print('learning rate is set to %f' % current_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        return optimizer


    optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


    # 损失函数定义
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

    #############################
    # training network          #
    #############################

    # 获取最短数据长度
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    # 设置epoch
    dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)

    current_step = 0
    # 开始训练
    accu_total1 = 0  # 统计dataset1中的总准确率和
    accu_total2 = 0  # 统计dataset2中的总准确率和
    time_total1 = 0  # 统计dataset1训练的总时间
    time_total2 = 0  # 统计dataset2训练的总时间
    for epoch in range(n_epoch):

        # 1.加载数据
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0

        # 防止数据超过最短数据长度，否则可能由于缺失某些数据出现报错
        while i < len_dataloader:

            ########################
            # target data training #
            ########################

            # 加载target
            data_target = data_target_iter.next()
            t_img, t_label = data_target

            # 1.梯度清零
            my_net.zero_grad()
            loss = 0
            batch_size = len(t_label)

            # 2.初始化一些变量
            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()

            # 判断gpu是否可用，如果可用，就将数据传入cuda中
            if cuda:
                t_img = t_img.cuda()
                t_label = t_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            # 将一部分数据resize,并拷贝到上面设置的变量
            input_img.resize_as_(t_img).copy_(t_img)
            class_label.resize_as_(t_label).copy_(t_label)
            target_inputv_img = Variable(input_img)
            target_classv_label = Variable(class_label)
            target_domainv_label = Variable(domain_label)

            # 论文中涉及到的公式
            if current_step > active_domain_loss_step:
                p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
                p = 2. / (1. + np.exp(-10 * p)) - 1

                # active domain loss
                # 这一步就是将输入输入到模型中，然后得到模型的结果
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all', p=p)
                target_private_coda, target_share_coda, target_domain_label, target_rec_code = result  # 通过python拆包得到的几个变量
                target_dann = gamma_weight * loss_similarity(target_domain_label, target_domainv_label)  # 4.计算损失值
                loss += target_dann  # 计算累计损失值
            else:
                if cuda:
                    target_dann = Variable(torch.zeros(1).float().cuda())  # ?
                else:
                    target_dann = Variable(torch.zeros(1).float())
                # 将输入传到模型中，然后得到模型结果
                result = my_net(input_data=target_inputv_img, mode='target', rec_scheme='all')
                target_private_coda, target_share_coda, _, target_rec_code = result  # 通过python的拆包得到几个变量

                # 以下几步用于计算损失值
                target_diff = beta_weight * loss_diff(target_private_coda, target_share_coda, weight=0.05)
                loss += target_diff
                target_mse = alpha_weight * loss_recon1(target_rec_code, target_inputv_img)
                loss += target_mse
                target_simse = alpha_weight * loss_recon2(target_rec_code, target_inputv_img)
                loss += target_mse

                # 5.计算梯度
                loss.backward()
                # 6.利用梯度优化权重和偏置等网络参数
                # optimizer = exp_lr_scheduler(optimizer=optimizer,step = current_step)
                optimizer.step()

                #######################
                # source data training#
                #######################

                data_source = data_source_iter.next()
                s_img, s_label = data_source

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

                    # active domain loss

                    # 输入模型进行训练
                    result = my_net(input_data=source_inputv_img, mode='source', rec_scheme='all', p=p)
                    source_private_code, source_share_code, source_domain_label, source_classv_label, source_rec_code = result
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
                        source_private_code, source_share_code, _, source_class_label, source_rec_code = result

                    source_classification = loss_classfication(source_class_label, source_classv_label)
                    loss += source_classification

                    source_diff = beta_weight * loss_diff(source_private_code, source_share_code, weight=0.05)
                    loss += source_diff
                    source_mse = alpha_weight * loss_recon1(source_rec_code, source_inputv_img)
                    loss += source_mse
                    source_simse = gamma_weight * loss_recon2(source_rec_code, source_inputv_img)
                    loss += source_simse

                    loss.backward()
                    # optimizer = exp_lr_scheduler(optimizer=optimizer,step=current_step)
                    optimizer.step()

                    ##############
                    # 测试保存    #
                    ##############
                    i += 1
                    current_step += 1
                    # print('source_classification: %f, source_dann: %f, source_diff: %f, '\
                    # 'source_mse: %f, source_simse: %f, target_dann: %f, target_diff: %f, '\
                    # 'target_mse: %f, target_simse: %f' \
                    # % (source_classification.data.cpu().numpy(), source_dann.data.cpu().numpy(),
                    #   source_diff.data.cpu().numpy(),
                    #   source_mse.data.cpu().numpy(), source_simse.data.cpu().numpy(), target_dann.data.cpu().numpy(),
                    #   target_diff.data.cpu().numpy(), target_mse.data.cpu().numpy(), target_simse.data.cpu().numpy()))
                    # 训练数据集1并计算累积时间，和累积准确率
                    start1 = time.time()
                    accu1 = test(epoch=epoch, name='dataset1')
                    end1 = time.time()
                    curr1 = end1 - start1
                    time_total1 += curr1
                    accu_total1 += accu1
                    # 训练数据集2并计算累积时间，和累积准确率
                    start2 = time.time()
                    accu2 = test(epoch=epoch, name='dataset2')
                    end2 = time.time()
                    curr2 = end2 - start2
                    time_total2 += curr2
                    accu_total2 += accu2
                    # print(time.strftime('%Y-%m-%d %H:%M:%S'), time.localtime(time.time()))

                # 获取平均准确率做为训练性能的评价指标
    model_index = epoch
    # 获取模型保存路径
    model_path = 'D:\study\graduation_project\grdaution_project\instru_identify\dataset18dataset2' + '\dsn_epoch_' + str(
        model_index) + '.pth'
    while os.path.exists(model_path):
        model_index = model_index + 1
        model_path = 'D:\study\graduation_project\grdaution_project\instru_identify\dataset18dataset2' + '\dsn_epoch_' + str(
            model_index) + '.pth'
    torch.save(my_net.state_dict(), model_path)  # 保存模型
    average_accu1 = accu_total1 / (len_dataloader * n_epoch)
    average_accu2 = accu_total2 / (len_dataloader * n_epoch)
    # result = [float(average_accu1),float(average_accu2)]
    # 所有数据均保留三位小数进行存储
    print(round(float(average_accu1), 3))
    print(round(float(average_accu2), 3))
    print(round(float(time_total1), 3))
    print(round(float(time_total2), 3))
    # print('result:',result)
    return result


if __name__ == '__main__':
    nett_str = sys.argv[1]
    #print(nett_str)
    run(nett_str)