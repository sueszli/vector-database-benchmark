"""
Runs CIFAR10 training with differential privacy.
"""
import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.func import functional_call, grad_and_value, vmap
from torchvision import models
from torchvision.datasets import CIFAR10
from tqdm import tqdm
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S', stream=sys.stdout)
logger = logging.getLogger('ddp')
logger.setLevel(level=logging.INFO)

def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    if False:
        for i in range(10):
            print('nop')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def accuracy(preds, labels):
    if False:
        return 10
    return (preds == labels).mean()

def compute_norms(sample_grads):
    if False:
        return 10
    batch_size = sample_grads[0].shape[0]
    norms = [sample_grad.view(batch_size, -1).norm(2, dim=-1) for sample_grad in sample_grads]
    norms = torch.stack(norms, dim=0).norm(2, dim=0)
    return (norms, batch_size)

def clip_and_accumulate_and_add_noise(model, max_per_sample_grad_norm=1.0, noise_multiplier=1.0):
    if False:
        return 10
    sample_grads = tuple((param.grad_sample for param in model.parameters()))
    (sample_norms, batch_size) = compute_norms(sample_grads)
    clip_factor = max_per_sample_grad_norm / (sample_norms + 1e-06)
    clip_factor = clip_factor.clamp(max=1.0)
    grads = tuple((torch.einsum('i,i...', clip_factor, sample_grad) for sample_grad in sample_grads))
    stddev = max_per_sample_grad_norm * noise_multiplier
    noises = tuple((torch.normal(0, stddev, grad_param.shape, device=grad_param.device) for grad_param in grads))
    grads = tuple((noise + grad_param for (noise, grad_param) in zip(noises, grads)))
    for (param, param_grad) in zip(model.parameters(), grads):
        param.grad = param_grad / batch_size
        del param.grad_sample

def train(args, model, train_loader, optimizer, epoch, device):
    if False:
        while True:
            i = 10
    start_time = datetime.now()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []
    for (i, (images, target)) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        def compute_loss_and_output(weights, image, target):
            if False:
                while True:
                    i = 10
            images = image.unsqueeze(0)
            targets = target.unsqueeze(0)
            output = functional_call(model, weights, images)
            loss = criterion(output, targets)
            return (loss, output.squeeze(0))
        grads_loss_output = grad_and_value(compute_loss_and_output, has_aux=True)
        weights = dict(model.named_parameters())
        detached_weights = {k: v.detach() for (k, v) in weights.items()}
        (sample_grads, (sample_loss, output)) = vmap(grads_loss_output, (None, 0, 0))(detached_weights, images, target)
        loss = sample_loss.mean()
        for (name, grad_sample) in sample_grads.items():
            weights[name].grad_sample = grad_sample.detach()
        clip_and_accumulate_and_add_noise(model, args.max_per_sample_grad_norm, args.sigma)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        losses.append(loss.item())
        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)
        optimizer.step()
        optimizer.zero_grad()
        if i % args.print_freq == 0:
            print(f'\tTrain Epoch: {epoch} \tLoss: {np.mean(losses):.6f} Acc@1: {np.mean(top1_acc):.6f} ')
    train_duration = datetime.now() - start_time
    return train_duration

def test(args, model, test_loader, device):
    if False:
        while True:
            i = 10
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []
    with torch.no_grad():
        for (images, target) in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)
            losses.append(loss.item())
            top1_acc.append(acc1)
    top1_avg = np.mean(top1_acc)
    print(f'\tTest set:Loss: {np.mean(losses):.6f} Acc@1: {top1_avg:.6f} ')
    return np.mean(top1_acc)

def main():
    if False:
        return 10
    args = parse_args()
    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)
    device = args.device
    if args.secure_rng:
        try:
            import torchcsprng as prng
        except ImportError as e:
            msg = 'To use secure RNG, you must install the torchcsprng package! Check out the instructions here: https://github.com/pytorch/csprng#installation'
            raise ImportError(msg) from e
        generator = prng.create_random_device_generator('/dev/urandom')
    else:
        generator = None
    augmentations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    normalize = [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201))]
    train_transform = transforms.Compose(normalize)
    test_transform = transforms.Compose(normalize)
    train_dataset = CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.sample_rate * len(train_dataset)), generator=generator, num_workers=args.workers, pin_memory=True)
    test_dataset = CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=args.workers)
    best_acc1 = 0
    model = models.__dict__[args.architecture](pretrained=False, norm_layer=lambda c: nn.GroupNorm(args.gn_groups, c))
    model = model.to(device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError('Optimizer not recognized. Please check spelling')
    accuracy_per_epoch = []
    time_per_epoch = []
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.lr_schedule == 'cos':
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        train_duration = train(args, model, train_loader, optimizer, epoch, device)
        top1_acc = test(args, model, test_loader, device)
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)
        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(top1_acc))
        save_checkpoint({'epoch': epoch + 1, 'arch': 'Convnet', 'state_dict': model.state_dict(), 'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, is_best, filename=args.checkpoint_file + '.tar')
    time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
    avg_time_per_epoch = sum(time_per_epoch_seconds) / len(time_per_epoch_seconds)
    metrics = {'accuracy': best_acc1, 'accuracy_per_epoch': accuracy_per_epoch, 'avg_time_per_epoch_str': str(timedelta(seconds=int(avg_time_per_epoch))), 'time_per_epoch': time_per_epoch_seconds}
    logger.info("\nNote:\n- 'total_time' includes the data loading time, training time and testing time.\n- 'time_per_epoch' measures the training time only.\n")
    logger.info(metrics)

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DP Training')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size-test', default=256, type=int, metavar='N', help='mini-batch size for test dataset (default: 256)')
    parser.add_argument('--sample-rate', default=0.005, type=float, metavar='SR', help='sample rate used for batch construction (default: 0.005)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, metavar='W', help='SGD weight decay', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--sigma', type=float, default=1.5, metavar='S', help='Noise multiplier (default 1.0)')
    parser.add_argument('-c', '--max-per-sample-grad_norm', type=float, default=10.0, metavar='C', help='Clip per-sample gradients to this norm (default 1.0)')
    parser.add_argument('--secure-rng', action='store_true', default=False, help="Enable Secure RNG to have trustworthy privacy guarantees.Comes at a performance cost. Opacus will emit a warning if secure rng is off,indicating that for production use it's recommender to turn it on.")
    parser.add_argument('--delta', type=float, default=1e-05, metavar='D', help='Target delta (default: 1e-5)')
    parser.add_argument('--checkpoint-file', type=str, default='checkpoint', help='path to save check points')
    parser.add_argument('--data-root', type=str, default='../cifar10', help='Where CIFAR10 is/will be stored')
    parser.add_argument('--log-dir', type=str, default='/tmp/stat/tensorboard', help='Where Tensorboard log will be stored')
    parser.add_argument('--optim', type=str, default='SGD', help='Optimizer to use (Adam, RMSprop, SGD)')
    parser.add_argument('--lr-schedule', type=str, choices=['constant', 'cos'], default='cos')
    parser.add_argument('--device', type=str, default='cpu', help='Device on which to run the code.')
    parser.add_argument('--architecture', type=str, default='resnet18', help='model from torchvision to run')
    parser.add_argument('--gn-groups', type=int, default=8, help='Number of groups in GroupNorm')
    parser.add_argument('--clip-per-layer', '--clip_per_layer', action='store_true', default=False, help='Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.')
    parser.add_argument('--debug', type=int, default=0, help='debug level (default: 0)')
    return parser.parse_args()
if __name__ == '__main__':
    main()