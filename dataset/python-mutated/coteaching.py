"""
Implements the co-teaching algorithm for training neural networks on noisily-labeled data (Han et al., 2018).
This module requires PyTorch (https://pytorch.org/get-started/locally/).
Example using this algorithm with cleanlab to achieve state of the art on CIFAR-10
for learning with noisy labels is provided within: https://github.com/cleanlab/examples/

``cifar_cnn.py`` provides an example model that can be trained via this algorithm.
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
MINIMUM_BATCH_SIZE = 16

def loss_coteaching(y_1, y_2, t, forget_rate, class_weights=None):
    if False:
        i = 10
        return i + 15
    'Co-Teaching Loss function.\n\n    Parameters\n    ----------\n    y_1 : Tensor array\n      Output logits from model 1\n\n    y_2 : Tensor array\n      Output logits from model 2\n\n    t : np.ndarray\n      List of Noisy Labels (t means targets)\n\n    forget_rate : float\n      Decimal between 0 and 1 for how quickly the models forget what they learn.\n      Just use rate_schedule[epoch] for this value\n\n    class_weights : Tensor array, shape (Number of classes x 1), Default: None\n      A np.torch.tensor list of length number of classes with weights\n    '
    loss_1 = F.cross_entropy(y_1, t, reduce=False, weight=class_weights)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]
    loss_2 = F.cross_entropy(y_2, t, reduce=False, weight=class_weights)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], weight=class_weights)
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], weight=class_weights)
    return (torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember)

def initialize_lr_scheduler(lr=0.001, epochs=250, epoch_decay_start=80):
    if False:
        while True:
            i = 10
    'Scheduler to adjust learning rate and betas for Adam Optimizer'
    mom1 = 0.9
    mom2 = 0.9
    alpha_plan = [lr] * epochs
    beta1_plan = [mom1] * epochs
    for i in range(epoch_decay_start, epochs):
        alpha_plan[i] = float(epochs - i) / (epochs - epoch_decay_start) * lr
        beta1_plan[i] = mom2
    return (alpha_plan, beta1_plan)

def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
    if False:
        return 10
    'Scheduler to adjust learning rate and betas for Adam Optimizer'
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)

def forget_rate_scheduler(epochs, forget_rate, num_gradual, exponent):
    if False:
        i = 10
        return i + 15
    'Tells Co-Teaching what fraction of examples to forget at each epoch.'
    forget_rate_schedule = np.ones(epochs) * forget_rate
    forget_rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
    return forget_rate_schedule

def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, args, forget_rate_schedule, class_weights, accuracy):
    if False:
        print('Hello World!')
    'PyTorch training function.\n\n    Parameters\n    ----------\n    train_loader : torch.utils.data.DataLoader\n    epoch : int\n    model1 : PyTorch class inheriting nn.Module\n        Must define __init__ and forward(self, x,)\n    optimizer1 : PyTorch torch.optim.Adam\n    model2 : PyTorch class inheriting nn.Module\n        Must define __init__ and forward(self, x,)\n    optimizer2 : PyTorch torch.optim.Adam\n    args : parser.parse_args() object\n        Must contain num_iter_per_epoch, print_freq, and epochs\n    forget_rate_schedule : np.ndarray of length number of epochs\n        Tells Co-Teaching loss what fraction of examples to forget about.\n    class_weights : Tensor array, shape (Number of classes x 1), Default: None\n      A np.torch.tensor list of length number of classes with weights\n    accuracy : function\n        A function of the form accuracy(output, target, topk=(1,)) for\n        computing top1 and top5 accuracy given output and true targets.'
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    model1.train()
    model2.train()
    for (i, (images, labels)) in enumerate(train_loader):
        if i == len(train_loader) - 1 and len(labels) < MINIMUM_BATCH_SIZE:
            continue
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        logits1 = model1(images)
        (prec1, _) = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1
        logits2 = model2(images)
        (prec2, _) = accuracy(logits2, labels, topk=(1, 5))
        train_total2 += 1
        train_correct2 += prec2
        (loss_1, loss_2) = loss_coteaching(logits1, logits2, labels, forget_rate=forget_rate_schedule[epoch], class_weights=class_weights)
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f ' % (epoch + 1, args.epochs, i + 1, len(train_loader.dataset) // args.batch_size, prec1, prec2, loss_1.data.item(), loss_2.data.item()))
    train_acc1 = float(train_correct) / float(train_total)
    train_acc2 = float(train_correct2) / float(train_total2)
    return (train_acc1, train_acc2)

def evaluate(test_loader, model1, model2):
    if False:
        while True:
            i = 10
    print('Evaluating Co-Teaching Model')
    model1.eval()
    correct1 = 0
    total1 = 0
    for (images, labels) in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        outputs1 = F.softmax(logits1, dim=1)
        (_, pred1) = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()
    model2.eval()
    correct2 = 0
    total2 = 0
    for (images, labels) in test_loader:
        images = Variable(images).cuda()
        logits2 = model2(images)
        outputs2 = F.softmax(logits2, dim=1)
        (_, pred2) = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
    acc1 = 100 * float(correct1) / float(total1)
    acc2 = 100 * float(correct2) / float(total2)
    return (acc1, acc2)