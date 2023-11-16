"""
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:
Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Network"
"""
import mxnet as mx
import numpy as np

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, num_group=32, bn_mom=0.9, workspace=256, memonger=False):
    if False:
        while True:
            i = 10
    'Return ResNet Unit symbol for building ResNet\n    Parameters\n    ----------\n    data : str\n        Input data\n    num_filter : int\n        Number of output channels\n    bnf : int\n        Bottle neck channels factor with regard to num_filter\n    stride : tuple\n        Stride used in convolution\n    dim_match : Boolean\n        True means channel number between input and output is the same, otherwise means differ\n    name : str\n        Base name of the operators\n    workspace : int\n        Workspace used in convolution operator\n    '
    if bottle_neck:
        conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter * 0.5), kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-05, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.5), num_group=num_group, kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-05, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv3')
        bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-05, momentum=bn_mom, name=name + '_bn3')
        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True, workspace=workspace, name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-05, momentum=bn_mom, name=name + '_sc_bn')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn3 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')
    else:
        conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv1')
        bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-05, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
        bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-05, name=name + '_bn2')
        if dim_match:
            shortcut = data
        else:
            shortcut_conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True, workspace=workspace, name=name + '_sc')
            shortcut = mx.sym.BatchNorm(data=shortcut_conv, fix_gamma=False, eps=2e-05, momentum=bn_mom, name=name + '_sc_bn')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        eltwise = bn2 + shortcut
        return mx.sym.Activation(data=eltwise, act_type='relu', name=name + '_relu')

def resnext(units, num_stages, filter_list, num_classes, num_group, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, dtype='float32', memonger=False):
    if False:
        while True:
            i = 10
    'Return ResNeXt symbol of\n    Parameters\n    ----------\n    units : list\n        Number of units in each stage\n    num_stages : int\n        Number of stage\n    filter_list : list\n        Channel size of each stage\n    num_classes : int\n        Ouput size of symbol\n    num_groupes: int\n    Number of conv groups\n    dataset : str\n        Dataset type, only cifar10 and imagenet supports\n    workspace : int\n        Workspace used in convolution operator\n    dtype : str\n        Precision (float32 or float16)\n    '
    num_unit = len(units)
    assert num_unit == num_stages
    data = mx.sym.Variable(name='data')
    if dtype == 'float32':
        data = mx.sym.identity(data=data, name='id')
    elif dtype == 'float16':
        data = mx.sym.Cast(data=data, dtype=np.float16)
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-05, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, name='conv0', workspace=workspace)
    else:
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3), no_bias=True, name='conv0', workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-05, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')
    for i in range(num_stages):
        body = residual_unit(body, filter_list[i + 1], (1 if i == 0 else 2, 1 if i == 0 else 2), False, name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)
        for j in range(units[i] - 1):
            body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2), bottle_neck=bottle_neck, num_group=num_group, bn_mom=bn_mom, workspace=workspace, memonger=memonger)
    pool1 = mx.sym.Pooling(data=body, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    if dtype == 'float16':
        fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')

def get_symbol(num_classes, num_layers, image_shape, num_group=32, conv_workspace=256, dtype='float32', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py\n    Original author Wei Wu\n    '
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 32:
        num_stages = 3
        if (num_layers - 2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers - 2) // 9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers - 2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers - 2) // 6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError('no experiments done on num_layers {}, you can do it yourself'.format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError('no experiments done on num_layers {}, you can do it yourself'.format(num_layers))
    return resnext(units=units, num_stages=num_stages, filter_list=filter_list, num_classes=num_classes, num_group=num_group, image_shape=image_shape, bottle_neck=bottle_neck, workspace=conv_workspace, dtype=dtype)