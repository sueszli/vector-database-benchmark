from cntk.initializer import he_normal
from cntk.layers import AveragePooling, MaxPooling, BatchNormalization, Convolution, Dense, Dropout
from cntk.ops import minus, element_times, relu, splice

def conv_bn_relu_layer(input, num_filters, filter_size, strides=(1, 1), pad=True, bnTimeConst=4096, init=he_normal()):
    if False:
        for i in range(10):
            print('nop')
    conv = Convolution(filter_size, num_filters, activation=None, init=init, pad=pad, strides=strides, bias=False)(input)
    bn = BatchNormalization(map_rank=1, normalization_time_constant=bnTimeConst, use_cntk_engine=False)(conv)
    return relu(bn)

def inception_block_1(input, num1x1, num5x5, num3x3dbl, numPool, bnTimeConst):
    if False:
        i = 10
        return i + 15
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1, 1), (1, 1), True, bnTimeConst)
    branch5x5_1 = conv_bn_relu_layer(input, num5x5[0], (1, 1), (1, 1), True, bnTimeConst)
    branch5x5 = conv_bn_relu_layer(branch5x5_1, num5x5[1], (5, 5), (1, 1), True, bnTimeConst)
    branch3x3dbl_1 = conv_bn_relu_layer(input, num3x3dbl[0], (1, 1), (1, 1), True, bnTimeConst)
    branch3x3dbl_2 = conv_bn_relu_layer(branch3x3dbl_1, num3x3dbl[1], (3, 3), (1, 1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_2, num3x3dbl[2], (3, 3), (1, 1), True, bnTimeConst)
    branchPool_avgpool = AveragePooling((3, 3), strides=(1, 1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1, 1), (1, 1), True, bnTimeConst)
    out = splice(branch1x1, branch5x5, branch3x3dbl, branchPool, axis=0)
    return out

def inception_block_2(input, num3x3, num3x3dbl, bnTimeConst):
    if False:
        for i in range(10):
            print('nop')
    branch3x3 = conv_bn_relu_layer(input, num3x3, (3, 3), (2, 2), False, bnTimeConst)
    branch3x3dbl_1 = conv_bn_relu_layer(input, num3x3dbl[0], (1, 1), (1, 1), True, bnTimeConst)
    branch3x3dbl_2 = conv_bn_relu_layer(branch3x3dbl_1, num3x3dbl[1], (3, 3), (1, 1), True, bnTimeConst)
    branch3x3dbl = conv_bn_relu_layer(branch3x3dbl_2, num3x3dbl[2], (3, 3), (2, 2), False, bnTimeConst)
    branchPool = MaxPooling((3, 3), strides=(2, 2), pad=False)(input)
    out = splice(branch3x3, branch3x3dbl, branchPool, axis=0)
    return out

def inception_block_3(input, num1x1, num7x7, num7x7dbl, numPool, bnTimeConst):
    if False:
        for i in range(10):
            print('nop')
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1, 1), (1, 1), True, bnTimeConst)
    branch7x7_1 = conv_bn_relu_layer(input, num7x7[0], (1, 1), (1, 1), True, bnTimeConst)
    branch7x7_2 = conv_bn_relu_layer(branch7x7_1, num7x7[1], (1, 7), (1, 1), True, bnTimeConst)
    branch7x7 = conv_bn_relu_layer(branch7x7_2, num7x7[2], (7, 1), (1, 1), True, bnTimeConst)
    branch7x7dbl_1 = conv_bn_relu_layer(input, num7x7dbl[0], (1, 1), (1, 1), True, bnTimeConst)
    branch7x7dbl_2 = conv_bn_relu_layer(branch7x7dbl_1, num7x7dbl[1], (7, 1), (1, 1), True, bnTimeConst)
    branch7x7dbl_3 = conv_bn_relu_layer(branch7x7dbl_2, num7x7dbl[2], (1, 7), (1, 1), True, bnTimeConst)
    branch7x7dbl_4 = conv_bn_relu_layer(branch7x7dbl_3, num7x7dbl[3], (7, 1), (1, 1), True, bnTimeConst)
    branch7x7dbl = conv_bn_relu_layer(branch7x7dbl_4, num7x7dbl[4], (1, 7), (1, 1), True, bnTimeConst)
    branchPool_avgpool = AveragePooling((3, 3), strides=(1, 1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1, 1), (1, 1), True, bnTimeConst)
    out = splice(branch1x1, branch7x7, branch7x7dbl, branchPool, axis=0)
    return out

def inception_block_4(input, num3x3, num7x7_3x3, bnTimeConst):
    if False:
        i = 10
        return i + 15
    branch3x3_1 = conv_bn_relu_layer(input, num3x3[0], (1, 1), (1, 1), True, bnTimeConst)
    branch3x3 = conv_bn_relu_layer(branch3x3_1, num3x3[1], (3, 3), (2, 2), False, bnTimeConst)
    branch7x7_3x3_1 = conv_bn_relu_layer(input, num7x7_3x3[0], (1, 1), (1, 1), True, bnTimeConst)
    branch7x7_3x3_2 = conv_bn_relu_layer(branch7x7_3x3_1, num7x7_3x3[1], (1, 7), (1, 1), True, bnTimeConst)
    branch7x7_3x3_3 = conv_bn_relu_layer(branch7x7_3x3_2, num7x7_3x3[2], (7, 1), (1, 1), True, bnTimeConst)
    branch7x7_3x3 = conv_bn_relu_layer(branch7x7_3x3_3, num7x7_3x3[3], (3, 3), (2, 2), False, bnTimeConst)
    branchPool = MaxPooling((3, 3), strides=(2, 2), pad=False)(input)
    out = splice(branch3x3, branch7x7_3x3, branchPool, axis=0)
    return out

def inception_block_5(input, num1x1, num3x3, num3x3_3x3, numPool, bnTimeConst):
    if False:
        i = 10
        return i + 15
    branch1x1 = conv_bn_relu_layer(input, num1x1, (1, 1), (1, 1), True, bnTimeConst)
    branch3x3_1 = conv_bn_relu_layer(input, num3x3[0], (1, 1), (1, 1), True, bnTimeConst)
    branch3x3_2 = conv_bn_relu_layer(branch3x3_1, num3x3[1], (1, 3), (1, 1), True, bnTimeConst)
    branch3x3_3 = conv_bn_relu_layer(branch3x3_1, num3x3[2], (3, 1), (1, 1), True, bnTimeConst)
    branch3x3 = splice(branch3x3_2, branch3x3_3, axis=0)
    branch3x3_3x3_1 = conv_bn_relu_layer(input, num3x3_3x3[0], (1, 1), (1, 1), True, bnTimeConst)
    branch3x3_3x3_2 = conv_bn_relu_layer(branch3x3_3x3_1, num3x3_3x3[1], (3, 3), (1, 1), True, bnTimeConst)
    branch3x3_3x3_3 = conv_bn_relu_layer(branch3x3_3x3_2, num3x3_3x3[1], (1, 3), (1, 1), True, bnTimeConst)
    branch3x3_3x3_4 = conv_bn_relu_layer(branch3x3_3x3_2, num3x3_3x3[3], (3, 1), (1, 1), True, bnTimeConst)
    branch3x3_3x3 = splice(branch3x3_3x3_3, branch3x3_3x3_4, axis=0)
    branchPool_avgpool = AveragePooling((3, 3), strides=(1, 1), pad=True)(input)
    branchPool = conv_bn_relu_layer(branchPool_avgpool, numPool, (1, 1), (1, 1), True, bnTimeConst)
    out = splice(branch1x1, branch3x3, branch3x3_3x3, branchPool, axis=0)
    return out

def inception_v3_norm_model(input, labelDim, dropRate, bnTimeConst):
    if False:
        return 10
    featMean = 128
    featScale = 1 / 128
    input_subtracted = minus(input, featMean)
    input_scaled = element_times(input_subtracted, featScale)
    return inception_v3_model(input_scaled, labelDim, dropRate, bnTimeConst)

def inception_v3_model(input, labelDim, dropRate, bnTimeConst):
    if False:
        while True:
            i = 10
    conv1 = conv_bn_relu_layer(input, 32, (3, 3), (2, 2), False, bnTimeConst)
    conv2 = conv_bn_relu_layer(conv1, 32, (3, 3), (1, 1), False, bnTimeConst)
    conv3 = conv_bn_relu_layer(conv2, 64, (3, 3), (1, 1), True, bnTimeConst)
    pool1 = MaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=False)(conv3)
    conv4 = conv_bn_relu_layer(pool1, 80, (1, 1), (1, 1), False, bnTimeConst)
    conv5 = conv_bn_relu_layer(conv4, 192, (3, 3), (1, 1), False, bnTimeConst)
    pool2 = MaxPooling(filter_shape=(3, 3), strides=(2, 2), pad=False)(conv5)
    mixed1 = inception_block_1(pool2, 64, [48, 64], [64, 96, 96], 32, bnTimeConst)
    mixed2 = inception_block_1(mixed1, 64, [48, 64], [64, 96, 96], 64, bnTimeConst)
    mixed3 = inception_block_1(mixed2, 64, [48, 64], [64, 96, 96], 64, bnTimeConst)
    mixed4 = inception_block_2(mixed3, 384, [64, 96, 96], bnTimeConst)
    mixed5 = inception_block_3(mixed4, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192, bnTimeConst)
    mixed6 = inception_block_3(mixed5, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192, bnTimeConst)
    mixed7 = inception_block_3(mixed6, 192, [160, 160, 192], [160, 160, 160, 160, 192], 192, bnTimeConst)
    mixed8 = inception_block_3(mixed7, 192, [192, 192, 192], [192, 192, 192, 192, 192], 192, bnTimeConst)
    mixed9 = inception_block_4(mixed8, [192, 320], [192, 192, 192, 192], bnTimeConst)
    mixed10 = inception_block_5(mixed9, 320, [384, 384, 384], [448, 384, 384, 384], 192, bnTimeConst)
    mixed11 = inception_block_5(mixed10, 320, [384, 384, 384], [448, 384, 384, 384], 192, bnTimeConst)
    pool3 = AveragePooling(filter_shape=(8, 8), pad=False)(mixed11)
    drop = Dropout(dropout_rate=dropRate)(pool3)
    z = Dense(labelDim, init=he_normal())(drop)
    auxPool = AveragePooling(filter_shape=(5, 5), strides=(3, 3), pad=False)(mixed8)
    auxConv1 = conv_bn_relu_layer(auxPool, 128, (1, 1), (1, 1), True, bnTimeConst)
    auxConv2 = conv_bn_relu_layer(auxConv1, 768, (5, 5), (1, 1), False, bnTimeConst)
    aux = Dense(labelDim, init=he_normal())(auxConv2)
    return {'z': z, 'aux': aux}