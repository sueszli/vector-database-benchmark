from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu, splice

def conv_bn_relu_layer(input, num_filters, filter_size, strides=(1, 1), pad=True, bnTimeConst=4096, init=he_normal()):
    if False:
        i = 10
        return i + 15
    conv = Convolution(filter_size, num_filters, activation=None, init=init, pad=pad, strides=strides, bias=False)(input)
    bn = BatchNormalization(map_rank=1, normalization_time_constant=bnTimeConst, use_cntk_engine=False)(conv)
    return relu(bn)

def inception_block_with_avgpool(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):
    if False:
        return 10
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3, 3), (1, 1), True, bnTimeConst)
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3, 3), (1, 1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3, 3), (1, 1), True, bnTimeConst)
    branchPool_avgpool = AveragePooling((3, 3), strides=(1, 1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1, 1), (1, 1), True, bnTimeConst)
    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)
    return out

def inception_block_with_maxpool(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):
    if False:
        print('Hello World!')
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3, 3), (1, 1), True, bnTimeConst)
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3, 3), (1, 1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3, 3), (1, 1), True, bnTimeConst)
    branchPool_maxpool = MaxPooling((3, 3), strides=(1, 1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_maxpool, numPool, (1, 1), (1, 1), True, bnTimeConst)
    out = splice(branch1x1, branch3x3, branch3x3dbl, branchPool, axis=0)
    return out

def inception_block_pass_through(input, num1x1, num3x3r, num3x3, num3x3dblr, num3x3dbl, numPool, bnTimeConst):
    if False:
        i = 10
        return i + 15
    branch3x3_reduce = conv_bn_relu_layer(input, num3x3r, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_reduce, num3x3, (3, 3), (2, 2), True, bnTimeConst)
    branch3x3dbl_reduce = conv_bn_relu_layer(input, num3x3dblr, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3dbl_conv = conv_bn_relu_layer(branch3x3dbl_reduce, num3x3dbl, (3, 3), (1, 1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_conv, num3x3dbl, (3, 3), (2, 2), True, bnTimeConst)
    branchPool = MaxPooling((3, 3), strides=(2, 2), pad=True)(input)
    out = splice(branch3x3, branch3x3dbl, branchPool, axis=0)
    return out

def bn_inception_model(input, labelDim, bnTimeConst):
    if False:
        while True:
            i = 10
    conv1 = conv_bn_relu_layer(input, 64, (7, 7), (2, 2), True, bnTimeConst)
    pool1 = MaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=True)(conv1)
    conv2a = conv_bn_relu_layer(pool1, 64, (1, 1), (1, 1), True, bnTimeConst)
    conv2b = conv_bn_relu_layer(conv2a, 192, (3, 3), (1, 1), True, bnTimeConst)
    pool2 = MaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=True)(conv2b)
    inception3a = inception_block_with_avgpool(pool2, 64, 64, 64, 64, 96, 32, bnTimeConst)
    inception3b = inception_block_with_avgpool(inception3a, 64, 64, 96, 64, 96, 64, bnTimeConst)
    inception3c = inception_block_pass_through(inception3b, 0, 128, 160, 64, 96, 0, bnTimeConst)
    inception4a = inception_block_with_avgpool(inception3c, 224, 64, 96, 96, 128, 128, bnTimeConst)
    inception4b = inception_block_with_avgpool(inception4a, 192, 96, 128, 96, 128, 128, bnTimeConst)
    inception4c = inception_block_with_avgpool(inception4b, 160, 128, 160, 128, 160, 128, bnTimeConst)
    inception4d = inception_block_with_avgpool(inception4c, 96, 128, 192, 160, 192, 128, bnTimeConst)
    inception4e = inception_block_pass_through(inception4d, 0, 128, 192, 192, 256, 0, bnTimeConst)
    inception5a = inception_block_with_avgpool(inception4e, 352, 192, 320, 160, 224, 128, bnTimeConst)
    inception5b = inception_block_with_maxpool(inception5a, 352, 192, 320, 192, 224, 128, bnTimeConst)
    pool3 = AveragePooling(filter_shape=(7, 7))(inception5b)
    z = Dense(labelDim, init=he_normal())(pool3)
    return z

def bn_inception_cifar_model(input, labelDim, bnTimeConst):
    if False:
        for i in range(10):
            print('nop')
    conv1a = conv_bn_relu_layer(input, 32, (3, 3), (1, 1), True, bnTimeConst)
    conv1b = conv_bn_relu_layer(conv1a, 32, (3, 3), (1, 1), True, bnTimeConst)
    conv1c = conv_bn_relu_layer(conv1b, 32, (3, 3), (1, 1), True, bnTimeConst)
    conv2a = conv_bn_relu_layer(conv1c, 32, (1, 1), (1, 1), True, bnTimeConst)
    conv2b = conv_bn_relu_layer(conv2a, 64, (3, 3), (1, 1), True, bnTimeConst)
    inception3a = inception_block_with_avgpool(conv2b, 32, 32, 32, 32, 48, 16, bnTimeConst)
    inception3b = inception_block_pass_through(inception3a, 0, 64, 80, 32, 48, 0, bnTimeConst)
    inception4a = inception_block_with_avgpool(inception3b, 96, 48, 64, 48, 64, 64, bnTimeConst)
    inception4b = inception_block_with_avgpool(inception4a, 48, 64, 96, 80, 96, 64, bnTimeConst)
    inception4c = inception_block_pass_through(inception4b, 0, 128, 192, 192, 256, 0, bnTimeConst)
    inception5a = inception_block_with_maxpool(inception4c, 176, 96, 160, 96, 112, 64, bnTimeConst)
    pool1 = AveragePooling(filter_shape=(8, 8))(inception5a)
    z = Dense(labelDim, init=he_normal())(pool1)
    return z