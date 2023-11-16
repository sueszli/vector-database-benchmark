from paddle import base
base.core._set_eager_deletion_mode(-1, -1, False)
import os
from seresnext_test_base import DeviceType
from simple_nets import init_data
import paddle
os.environ['CPU_NUM'] = str(4)
os.environ['FLAGS_cudnn_deterministic'] = str(1)
remove_dropout = False
remove_bn = False
remove_cudnn_conv = True
remove_dropout = True
remove_bn = True

def squeeze_excitation(input, num_channels, reduction_ratio):
    if False:
        while True:
            i = 10
    conv = input
    shape = conv.shape
    reshape = paddle.reshape(x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = paddle.mean(x=reshape, axis=2)
    squeeze = paddle.static.nn.fc(x=pool, size=num_channels // reduction_ratio, activation='relu')
    excitation = paddle.static.nn.fc(x=squeeze, size=num_channels, activation='sigmoid')
    scale = paddle.tensor.math._multiply_with_axis(x=input, y=excitation, axis=0)
    return scale

def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1, act=None):
    if False:
        while True:
            i = 10
    conv = paddle.static.nn.conv2d(input=input, num_filters=num_filters, filter_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, groups=groups, act=None, use_cudnn=not remove_cudnn_conv, bias_attr=False)
    return conv if remove_bn else paddle.static.nn.batch_norm(input=conv, act=act, momentum=0.1)

def shortcut(input, ch_out, stride):
    if False:
        for i in range(10):
            print('nop')
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input

def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    if False:
        return 10
    conv0 = conv_bn_layer(input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(input=conv0, num_filters=num_filters * 2, filter_size=3, stride=stride, groups=cardinality, act='relu')
    conv2 = conv_bn_layer(input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(input=conv2, num_channels=num_filters * 2, reduction_ratio=reduction_ratio)
    short = shortcut(input, num_filters * 2, stride)
    return paddle.nn.functional.relu(paddle.add(x=short, y=scale))
img_shape = [3, 224, 224]

def SE_ResNeXt50Small(use_feed):
    if False:
        i = 10
        return i + 15
    img = paddle.static.data(name='image', shape=[-1] + img_shape, dtype='float32')
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    conv = conv_bn_layer(input=img, num_filters=16, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = paddle.nn.functional.max_pool2d(x=conv, kernel_size=3, stride=2, padding=1)
    cardinality = 32
    reduction_ratio = 16
    depth = [3, 4, 6, 3]
    num_filters = [128, 256, 512, 1024]
    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(input=conv, num_filters=num_filters[block], stride=2 if i == 0 and block != 0 else 1, cardinality=cardinality, reduction_ratio=reduction_ratio)
    shape = conv.shape
    reshape = paddle.reshape(x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = paddle.mean(x=reshape, axis=2)
    dropout = pool if remove_dropout else paddle.nn.functional.dropout(x=pool, p=0.2)
    prediction = paddle.static.nn.fc(x=dropout, size=1000, activation='softmax')
    loss = paddle.nn.functional.cross_entropy(input=prediction, label=label, reduction='none', use_softmax=True)
    loss = paddle.mean(loss)
    return loss

def optimizer(learning_rate=0.01):
    if False:
        print('Hello World!')
    optimizer = paddle.optimizer.Momentum(learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=learning_rate, T_max=1), momentum=0.9, weight_decay=paddle.regularizer.L2Decay(0.0001))
    return optimizer
model = SE_ResNeXt50Small

def batch_size(use_device):
    if False:
        print('Hello World!')
    if use_device == DeviceType.CUDA:
        return 4
    return 12

def iter(use_device):
    if False:
        i = 10
        return i + 15
    if use_device == DeviceType.CUDA:
        return 10
    return 1
(gpu_img, gpu_label) = init_data(batch_size=batch_size(use_device=DeviceType.CUDA), img_shape=img_shape, label_range=999)
(cpu_img, cpu_label) = init_data(batch_size=batch_size(use_device=DeviceType.CPU), img_shape=img_shape, label_range=999)
feed_dict_gpu = {'image': gpu_img, 'label': gpu_label}
feed_dict_cpu = {'image': cpu_img, 'label': cpu_label}

def feed_dict(use_device):
    if False:
        print('Hello World!')
    if use_device == DeviceType.CUDA:
        return feed_dict_gpu
    return feed_dict_cpu