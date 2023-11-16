from __future__ import division
import numpy as np
import pycuda.driver as drv
from neon import logger as neon_logger
from neon.backends.nervanagpu import NervanaGPU
from neon.backends.layer_gpu import Layer, DataLayer, ConvLayer, PoolLayer, FullLayer, Inception, BatchNorm
nets = ('Alexnet', 'AlexnetBN', 'GoogLeNet1BN')
dtypes = (np.float16,)
loops = 10
layer_bench = 0
print_stats = 0
zeros = 0
verbose = 0
ng = NervanaGPU(bench=layer_bench)
neon_logger.display(drv.Context.get_current().get_device().name())
conv11 = {'R': 11, 'S': 11, 'pad_h': 2, 'pad_w': 2, 'str_h': 4, 'str_w': 4}
conv11p0 = {'R': 11, 'S': 11, 'pad_h': 0, 'pad_w': 0, 'str_h': 4, 'str_w': 4}
conv7 = {'R': 7, 'S': 7, 'pad_h': 3, 'pad_w': 3, 'str_h': 2, 'str_w': 2}
conv5 = {'R': 5, 'S': 5, 'pad_h': 2, 'pad_w': 2}
conv5p0 = {'R': 5, 'S': 5, 'pad_h': 0, 'pad_w': 0}
conv3 = {'R': 3, 'S': 3, 'pad_h': 1, 'pad_w': 1}
conv2 = {'R': 2, 'S': 2, 'pad_h': 0, 'pad_w': 0, 'str_h': 2, 'str_w': 2}
conv1 = {'R': 1, 'S': 1, 'pad_h': 0, 'pad_w': 0}
pool2s2p0 = {'R': 2, 'S': 2}
pool3s2p0 = {'R': 3, 'S': 3, 'str_h': 2, 'str_w': 2}
pool3s2p1 = {'R': 3, 'S': 3, 'str_h': 2, 'str_w': 2, 'pad_h': 1, 'pad_w': 1}
pool3s1p1 = {'R': 3, 'S': 3, 'str_h': 1, 'str_w': 1, 'pad_h': 1, 'pad_w': 1}
pool7s1p0 = {'R': 7, 'S': 7, 'str_h': 1, 'str_w': 1}
pool1j2 = {'op': 'max', 'J': 2}
pool2j2 = {'op': 'max', 'J': 2, 'R': 2, 'S': 2}
pool3j2 = {'op': 'max', 'J': 2, 'R': 3, 'S': 3}

def inception1(conf):
    if False:
        print('Hello World!')
    return {'layer': Inception, 'partitions': (({'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[0][0]},), ({'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[1][0]}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': conf[1][1]}), ({'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[2][0]}, {'layer': ConvLayer, 'common': conv5, 'relu': True, 'K': conf[2][1]}), ({'layer': PoolLayer, 'common': pool3s1p1, 'op': 'max'}, {'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[3][0]}))}

def inception1BN(conf):
    if False:
        return 10
    return {'layer': Inception, 'partitions': (({'layer': ConvLayer, 'common': conv1, 'K': conf[0][0], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}), ({'layer': ConvLayer, 'common': conv1, 'K': conf[1][0], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': conf[1][1], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}), ({'layer': ConvLayer, 'common': conv1, 'K': conf[2][0], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv5, 'K': conf[2][1], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}), ({'layer': PoolLayer, 'common': pool3s1p1, 'op': 'max'}, {'layer': ConvLayer, 'common': conv1, 'K': conf[3][0], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}))}

def inception2(conf):
    if False:
        for i in range(10):
            print('nop')
    layer = {'layer': Inception, 'partitions': []}
    partitions = layer['partitions']
    if conf[0][0]:
        partitions.append(({'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[0][0]},))
    partitions.extend((({'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[1][0]}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': conf[1][1]}), ({'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[2][0]}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': conf[2][1]}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': conf[2][1]})))
    if conf[3][1]:
        partitions.append(({'layer': PoolLayer, 'common': pool3s1p1, 'op': conf[3][0]}, {'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': conf[3][1]}))
    else:
        partitions.append(({'layer': PoolLayer, 'common': pool3s1p1, 'op': conf[3][0]},))
    return layer

def inception2BN(conf):
    if False:
        print('Hello World!')
    layer = {'layer': Inception, 'partitions': []}
    partitions = layer['partitions']
    if conf[0][0]:
        partitions.append(({'layer': ConvLayer, 'common': conv1, 'K': conf[0][0], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}))
    partitions.extend((({'layer': ConvLayer, 'common': conv1, 'K': conf[1][0], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': conf[1][1], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}), ({'layer': ConvLayer, 'common': conv1, 'K': conf[2][0], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': conf[2][1], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': conf[2][1], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True})))
    if conf[3][1]:
        partitions.append(({'layer': PoolLayer, 'common': pool3s1p1, 'op': conf[3][0]}, {'layer': ConvLayer, 'common': conv1, 'K': conf[3][1], 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}))
    else:
        partitions.append(({'layer': PoolLayer, 'common': pool3s1p1, 'op': conf[3][0]},))
    return layer
networks = {'Alexnet': ({'warmup': 4}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv11, 'relu': True, 'K': 64}, {'layer': PoolLayer, 'common': pool3s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv5, 'relu': True, 'K': 192}, {'layer': PoolLayer, 'common': pool3s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 384}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': PoolLayer, 'common': pool3s2p0, 'op': 'max'}, {'layer': FullLayer, 'nOut': 4096, 'relu': True}, {'layer': FullLayer, 'nOut': 4096, 'relu': True}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'AlexnetBN': ({'warmup': 4}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv11, 'K': 64, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool3s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv5, 'K': 192, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool3s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'K': 384, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 256, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 256, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool3s2p0, 'op': 'max'}, {'layer': FullLayer, 'nOut': 4096}, {'layer': BatchNorm, 'relu': True}, {'layer': FullLayer, 'nOut': 4096}, {'layer': BatchNorm, 'relu': True}, {'layer': FullLayer, 'relu': True, 'nOut': 1000}), 'Overfeat': ({'warmup': 1}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 231, 'W': 231}, {'layer': ConvLayer, 'common': conv11p0, 'relu': True, 'K': 96}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv5p0, 'relu': True, 'K': 256}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 1024}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 1024}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': FullLayer, 'nOut': 3072, 'relu': True}, {'layer': FullLayer, 'nOut': 4096, 'relu': True}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'OverfeatBN': ({'warmup': 1}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 231, 'W': 231}, {'layer': ConvLayer, 'common': conv11p0, 'K': 96, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv5p0, 'K': 256, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'K': 512, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 1024, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 1024, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': FullLayer, 'nOut': 3072}, {'layer': BatchNorm, 'relu': True}, {'layer': FullLayer, 'nOut': 4096}, {'layer': BatchNorm, 'relu': True}, {'layer': FullLayer, 'relu': True, 'nOut': 1000}), 'VGG': ({'warmup': 1}, {'layer': DataLayer, 'N': 64, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 64}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 128}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': FullLayer, 'nOut': 4096, 'relu': True}, {'layer': FullLayer, 'nOut': 4096, 'relu': True}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'VGG_BN': ({'warmup': 1}, {'layer': DataLayer, 'N': 64, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv3, 'K': 64, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'K': 128, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'K': 256, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 256, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'K': 512, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 512, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'K': 512, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 512, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': FullLayer, 'nOut': 4096}, {'layer': BatchNorm, 'relu': True}, {'layer': FullLayer, 'nOut': 4096}, {'layer': BatchNorm, 'relu': True}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'VGG_E': ({'warmup': 1}, {'layer': DataLayer, 'N': 64, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 64}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 64}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 128}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 128}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 256}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 512}, {'layer': PoolLayer, 'common': pool2s2p0, 'op': 'max'}, {'layer': FullLayer, 'nOut': 4096, 'relu': True}, {'layer': FullLayer, 'nOut': 4096, 'relu': True}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'GoogLeNet1': ({'warmup': 1}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv7, 'relu': True, 'K': 64}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, {'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': 64}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 192}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception1([(64,), (96, 128), (16, 32), (32,)]), inception1([(128,), (128, 192), (32, 96), (64,)]), {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception1([(192,), (96, 208), (16, 48), (64,)]), inception1([(160,), (112, 224), (24, 64), (64,)]), inception1([(128,), (128, 256), (24, 64), (64,)]), inception1([(112,), (144, 288), (32, 64), (64,)]), inception1([(256,), (160, 320), (32, 128), (128,)]), {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception1([(256,), (160, 320), (32, 128), (128,)]), inception1([(384,), (192, 384), (48, 128), (128,)]), {'layer': PoolLayer, 'common': pool7s1p0, 'op': 'avg'}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'GoogLeNet1BN': ({'warmup': 1}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv7, 'K': 64, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, {'layer': ConvLayer, 'common': conv1, 'K': 64, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 192, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception1BN([(64,), (96, 128), (16, 32), (32,)]), inception1BN([(128,), (128, 192), (32, 96), (64,)]), {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception1BN([(192,), (96, 208), (16, 48), (64,)]), inception1BN([(160,), (112, 224), (24, 64), (64,)]), inception1BN([(128,), (128, 256), (24, 64), (64,)]), inception1BN([(112,), (144, 288), (32, 64), (64,)]), inception1BN([(256,), (160, 320), (32, 128), (128,)]), {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception1BN([(256,), (160, 320), (32, 128), (128,)]), inception1BN([(384,), (192, 384), (48, 128), (128,)]), {'layer': PoolLayer, 'common': pool7s1p0, 'op': 'avg'}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'GoogLeNet2': ({'warmup': 1}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv7, 'relu': True, 'K': 64}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, {'layer': ConvLayer, 'common': conv1, 'relu': True, 'K': 64}, {'layer': ConvLayer, 'common': conv3, 'relu': True, 'K': 192}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception2([(64,), (64, 64), (64, 96), ('avg', 32)]), inception2([(64,), (64, 96), (64, 96), ('avg', 64)]), inception2([(0,), (128, 160), (64, 96), ('max', 0)]), {'layer': ConvLayer, 'common': conv2, 'relu': True, 'K': 576}, inception2([(224,), (64, 96), (96, 128), ('avg', 128)]), inception2([(192,), (96, 128), (96, 128), ('avg', 128)]), inception2([(160,), (128, 160), (128, 160), ('avg', 96)]), inception2([(96,), (128, 192), (160, 192), ('avg', 96)]), inception2([(0,), (128, 192), (192, 256), ('max', 0)]), {'layer': ConvLayer, 'common': conv2, 'relu': True, 'K': 1024}, inception2([(352,), (192, 320), (160, 224), ('avg', 128)]), inception2([(352,), (192, 320), (192, 224), ('max', 128)]), {'layer': PoolLayer, 'common': pool7s1p0, 'op': 'avg'}, {'layer': FullLayer, 'nOut': 1000, 'relu': True}), 'GoogLeNet2BN': ({'warmup': 1}, {'layer': DataLayer, 'N': 128, 'C': 3, 'H': 224, 'W': 224}, {'layer': ConvLayer, 'common': conv7, 'K': 64, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, {'layer': ConvLayer, 'common': conv1, 'K': 64, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': ConvLayer, 'common': conv3, 'K': 192, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, {'layer': PoolLayer, 'common': pool3s2p1, 'op': 'max'}, inception2BN([(64,), (64, 64), (64, 96), ('avg', 32)]), inception2BN([(64,), (64, 96), (64, 96), ('avg', 64)]), inception2BN([(0,), (128, 160), (64, 96), ('max', 0)]), {'layer': ConvLayer, 'common': conv2, 'K': 576, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, inception2BN([(224,), (64, 96), (96, 128), ('avg', 128)]), inception2BN([(192,), (96, 128), (96, 128), ('avg', 128)]), inception2BN([(160,), (128, 160), (128, 160), ('avg', 96)]), inception2BN([(96,), (128, 192), (160, 192), ('avg', 96)]), inception2BN([(0,), (128, 192), (192, 256), ('max', 0)]), {'layer': ConvLayer, 'common': conv2, 'K': 1024, 'bsum': True}, {'layer': BatchNorm, 'relu': True, 'bsum': True}, inception2BN([(352,), (192, 320), (160, 224), ('avg', 128)]), inception2BN([(352,), (192, 320), (192, 224), ('max', 128)]), {'layer': PoolLayer, 'common': pool7s1p0, 'op': 'avg'}, {'layer': FullLayer, 'nOut': 1000, 'relu': True})}
for net in nets:
    for dtype in dtypes:
        warmup = networks[net][0]['warmup']
        network = networks[net][1:]
        name = '%s (dtype=%s, N=%d)' % (net, np.dtype(dtype).name, network[0]['N'])
        networks[net][0]['warmup'] = 1
        neon_logger.display('------------------------------------------------')
        neon_logger.display('Benchmarking: ' + name)
        neon_logger.display('------------------------------------------------')
        layers = []
        prev_layer = None
        max_deltas = 0
        max_weights = 0
        max_delta_layer = None
        max_weight_layer = None
        shared_weights = None
        shared_deltas = []
        inception = False
        for conf in network:
            layer = Layer.create(ng, conf, prev_layer, dtype)
            if type(layer) is Inception:
                inception = True
            if layer.sizeF > max_weights:
                max_weights = layer.sizeF
                max_weight_layer = layer
            if layer.sizeI > max_deltas and type(prev_layer) is not DataLayer:
                max_deltas = layer.sizeI
                max_delta_layer = layer
            prev_layer = layer
            layers.append(layer)
        shared_deltas.append(ng.empty(max_delta_layer.dimI, dtype=max_delta_layer.dtype))
        shared_deltas.append(ng.empty(max_delta_layer.dimI, dtype=max_delta_layer.dtype))
        if inception:
            shared_deltas.append(ng.empty(max_delta_layer.dimI, dtype=max_delta_layer.dtype))
            shared_deltas.append(ng.empty(max_delta_layer.dimI, dtype=max_delta_layer.dtype))
        shared_updates = ng.empty(max_weight_layer.dimF, dtype=np.float32)
        for (i, layer) in enumerate(layers):
            if verbose:
                neon_logger.display(layer)
            layer.init_activations()
            layer.init_weights(shared=shared_updates, zeros=zeros)
            if i > 1:
                layer.init_deltas(shared=shared_deltas)
        if verbose:
            (remain, total) = drv.mem_get_info()
            neon_logger.display('%.3fGB of %.3fGB Allocated (%.3fGB Remaining)' % ((total - remain) / 1024.0 ** 3, total / 1024.0 ** 3, remain / 1024.0 ** 3))
        if zeros:
            layers[0].init_data()
        else:
            layers[0].init_data(np.random.uniform(0.0, 1.0, layers[0].dimO))
        start = drv.Event()
        end = drv.Event()
        fprop_time = 0
        bprop_time = 0
        fprop_flops = 0
        bprop_flops = 0
        for loop in range(loops + warmup):
            loop = loop - warmup + 1
            if loop < 0:
                loop = 0
            start.record()
            flops = 0
            propagation = None
            for layer in layers:
                propagation = layer.fprop(propagation)
                flops += layer.flops
                if print_stats:
                    layer.fprop_stats()
            end.record()
            end.synchronize()
            msecs = end.time_since(start)
            neon_logger.display('fprop(%2d): %8.3f msecs %8.3f gflops' % (loop, msecs, flops / (msecs * 1000000.0)))
            if loop > 0:
                fprop_time += msecs
                fprop_flops += flops
            start.record()
            flops = 0
            for layer in layers[:0:-1]:
                propagation = layer.bprop(propagation)
                flops += layer.flops * 2
                if print_stats:
                    layer.bprop_stats()
            end.record()
            end.synchronize()
            msecs = end.time_since(start)
            neon_logger.display('bprop(%2d): %8.3f msecs %8.3f gflops' % (loop, msecs, flops / (msecs * 1000000.0)))
            if loop > 0:
                bprop_time += msecs
                bprop_flops += flops
        if loops > 0:
            neon_logger.display('---------------------------------------------')
            neon_logger.display(name + ' Results:')
            neon_logger.display('---------------------------------------------')
            neon_logger.display('Avg(%d) fprop: %8.3f msecs %.3f gflops' % (loops, fprop_time / loops, fprop_flops / (fprop_time * 1000000.0)))
            neon_logger.display('Avg(%d) bprop: %8.3f msecs %.3f gflops' % (loops, bprop_time / loops, bprop_flops / (bprop_time * 1000000.0)))
            fprop_time += bprop_time
            fprop_flops += bprop_flops
            neon_logger.display('Avg(%d) total: %8.3f msecs %.3f gflops\n\n' % (loops, fprop_time / loops, fprop_flops / (fprop_time * 1000000.0)))