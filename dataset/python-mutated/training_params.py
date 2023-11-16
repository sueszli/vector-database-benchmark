import torch
from torch.optim import SGD
from torchvision.models import resnet50, inception_v3, mobilenet_v2, densenet121
model = resnet50(pretrained=False)

def resnet50_train_params():
    if False:
        while True:
            i = 10
    model = resnet50(pretrained=False)
    return {'model': model, 'optimizer': SGD, 'weight_decay': 0.0001, 'lr': 0.1, 'lr_decay_rate': None, 'lr_step_size': None}

def inception_v3_train_params():
    if False:
        return 10
    model = inception_v3(pretrained=False, init_weights=False)
    return {'model': model, 'optimizer': SGD, 'weight_decay': 0, 'lr': 0.045, 'lr_decay_rate': 0.94, 'lr_step_size': 2}

def mobilenet_v2_train_params():
    if False:
        while True:
            i = 10
    model = mobilenet_v2(pretrained=False)
    return {'model': model, 'optimizer': SGD, 'weight_decay': 4e-05, 'lr': 0.045, 'lr_decay_rate': 0.98, 'lr_step_size': 1}

def densenet121_train_params():
    if False:
        for i in range(10):
            print('nop')
    model = densenet121(pretrained=False)
    return {'model': model, 'optimizer': SGD, 'weight_decay': 0.0001, 'lr': 0.1, 'lr_decay_rate': None, 'lr_step_size': None}