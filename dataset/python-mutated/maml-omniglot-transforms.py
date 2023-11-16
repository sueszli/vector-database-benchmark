"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""
import argparse
import functools
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from support.omniglot_loaders import OmniglotNShot
from torch import nn
from torch.func import functional_call, grad, vmap
mpl.use('Agg')
plt.style.use('bmh')

def main():
    if False:
        i = 10
        return i + 15
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n-way', '--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k-spt', '--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k-qry', '--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--device', type=str, help='device', default='cuda')
    argparser.add_argument('--task-num', '--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    args = argparser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = args.device
    db = OmniglotNShot('/tmp/omniglot-data', batchsz=args.task_num, n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry, imgsz=28, device=device)
    inplace_relu = True
    net = nn.Sequential(nn.Conv2d(1, 64, 3), nn.BatchNorm2d(64, affine=True, track_running_stats=False), nn.ReLU(inplace=inplace_relu), nn.MaxPool2d(2, 2), nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64, affine=True, track_running_stats=False), nn.ReLU(inplace=inplace_relu), nn.MaxPool2d(2, 2), nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64, affine=True, track_running_stats=False), nn.ReLU(inplace=inplace_relu), nn.MaxPool2d(2, 2), nn.Flatten(), nn.Linear(64, args.n_way)).to(device)
    net.train()
    meta_opt = optim.Adam(net.parameters(), lr=0.001)
    log = []
    for epoch in range(100):
        train(db, net, device, meta_opt, epoch, log)
        test(db, net, device, epoch, log)
        plot(log)

def loss_for_task(net, n_inner_iter, x_spt, y_spt, x_qry, y_qry):
    if False:
        print('Hello World!')
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    querysz = x_qry.size(0)

    def compute_loss(new_params, buffers, x, y):
        if False:
            print('Hello World!')
        logits = functional_call(net, (new_params, buffers), x)
        loss = F.cross_entropy(logits, y)
        return loss
    new_params = params
    for _ in range(n_inner_iter):
        grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
        new_params = {k: new_params[k] - g * 0.1 for (k, g) in grads.items()}
    qry_logits = functional_call(net, (new_params, buffers), x_qry)
    qry_loss = F.cross_entropy(qry_logits, y_qry)
    qry_acc = (qry_logits.argmax(dim=1) == y_qry).sum() / querysz
    return (qry_loss, qry_acc)

def train(db, net, device, meta_opt, epoch, log):
    if False:
        while True:
            i = 10
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    n_train_iter = db.x_train.shape[0] // db.batchsz
    for batch_idx in range(n_train_iter):
        start_time = time.time()
        (x_spt, y_spt, x_qry, y_qry) = db.next()
        (task_num, setsz, c_, h, w) = x_spt.size()
        n_inner_iter = 5
        meta_opt.zero_grad()
        compute_loss_for_task = functools.partial(loss_for_task, net, n_inner_iter)
        (qry_losses, qry_accs) = vmap(compute_loss_for_task)(x_spt, y_spt, x_qry, y_qry)
        qry_losses.sum().backward()
        meta_opt.step()
        qry_losses = qry_losses.detach().sum() / task_num
        qry_accs = 100.0 * qry_accs.sum() / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}')
        log.append({'epoch': i, 'loss': qry_losses, 'acc': qry_accs, 'mode': 'train', 'time': time.time()})

def test(db, net, device, epoch, log):
    if False:
        return 10
    params = dict(net.named_parameters())
    buffers = dict(net.named_buffers())
    n_test_iter = db.x_test.shape[0] // db.batchsz
    qry_losses = []
    qry_accs = []
    for batch_idx in range(n_test_iter):
        (x_spt, y_spt, x_qry, y_qry) = db.next('test')
        (task_num, setsz, c_, h, w) = x_spt.size()
        n_inner_iter = 5
        for i in range(task_num):
            new_params = params
            for _ in range(n_inner_iter):
                spt_logits = functional_call(net, (new_params, buffers), x_spt[i])
                spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                grads = torch.autograd.grad(spt_loss, new_params.values())
                new_params = {k: new_params[k] - g * 0.1 for (k, g) in zip(new_params, grads)}
            qry_logits = functional_call(net, (new_params, buffers), x_qry[i]).detach()
            qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction='none')
            qry_losses.append(qry_loss.detach())
            qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())
    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100.0 * torch.cat(qry_accs).float().mean().item()
    print(f'[Epoch {epoch + 1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}')
    log.append({'epoch': epoch + 1, 'loss': qry_losses, 'acc': qry_accs, 'mode': 'test', 'time': time.time()})

def plot(log):
    if False:
        print('Hello World!')
    df = pd.DataFrame(log)
    (fig, ax) = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(70, 100)
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'maml-accs.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)
if __name__ == '__main__':
    main()