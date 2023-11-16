"""ResNet56 model for Keras adapted from tf.keras.applications.ResNet50.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-05
L2_WEIGHT_DECAY = 0.0002

def identity_building_block(input_tensor, kernel_size, filters, stage, block, training=None):
    if False:
        while True:
            i = 10
    'The identity block is the block that has no conv layer at shortcut.\n\n  Arguments:\n    input_tensor: input tensor\n    kernel_size: default 3, the kernel size of\n        middle conv layer at main path\n    filters: list of integers, the filters of 3 conv layer at main path\n    stage: integer, current stage label, used for generating layer names\n    block: current block label, used for generating layer names\n    training: Only used if training keras model with Estimator.  In other\n      scenarios it is handled automatically.\n\n  Returns:\n    Output tensor for the block.\n  '
    (filters1, filters2) = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = layers.Conv2D(filters1, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=bn_name_base + '2a')(x, training=training)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters2, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=bn_name_base + '2b')(x, training=training)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_building_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), training=None):
    if False:
        print('Hello World!')
    'A block that has a conv layer at shortcut.\n\n  Arguments:\n    input_tensor: input tensor\n    kernel_size: default 3, the kernel size of\n        middle conv layer at main path\n    filters: list of integers, the filters of 3 conv layer at main path\n    stage: integer, current stage label, used for generating layer names\n    block: current block label, used for generating layer names\n    strides: Strides for the first conv layer in the block.\n    training: Only used if training keras model with Estimator.  In other\n      scenarios it is handled automatically.\n\n  Returns:\n    Output tensor for the block.\n\n  Note that from stage 3,\n  the first conv layer at main path is with strides=(2, 2)\n  And the shortcut should have strides=(2, 2) as well\n  '
    (filters1, filters2) = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = layers.Conv2D(filters1, kernel_size, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=bn_name_base + '2a')(x, training=training)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters2, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=bn_name_base + '2b')(x, training=training)
    shortcut = layers.Conv2D(filters2, (1, 1), strides=strides, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name=bn_name_base + '1')(shortcut, training=training)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def resnet_block(input_tensor, size, kernel_size, filters, stage, conv_strides=(2, 2), training=None):
    if False:
        return 10
    'A block which applies conv followed by multiple identity blocks.\n\n  Arguments:\n    input_tensor: input tensor\n    size: integer, number of constituent conv/identity building blocks.\n    A conv block is applied once, followed by (size - 1) identity blocks.\n    kernel_size: default 3, the kernel size of\n        middle conv layer at main path\n    filters: list of integers, the filters of 3 conv layer at main path\n    stage: integer, current stage label, used for generating layer names\n    conv_strides: Strides for the first conv layer in the block.\n    training: Only used if training keras model with Estimator.  In other\n      scenarios it is handled automatically.\n\n  Returns:\n    Output tensor after applying conv and identity blocks.\n  '
    x = conv_building_block(input_tensor, kernel_size, filters, stage=stage, strides=conv_strides, block='block_0', training=training)
    for i in range(size - 1):
        x = identity_building_block(x, kernel_size, filters, stage=stage, block='block_%d' % (i + 1), training=training)
    return x

def resnet(num_blocks, classes=10, training=None):
    if False:
        print('Hello World!')
    'Instantiates the ResNet architecture.\n\n  Arguments:\n    num_blocks: integer, the number of conv/identity blocks in each block.\n      The ResNet contains 3 blocks with each block containing one conv block\n      followed by (layers_per_block - 1) number of idenity blocks. Each\n      conv/idenity block has 2 convolutional layers. With the input\n      convolutional layer and the pooling layer towards the end, this brings\n      the total size of the network to (6*num_blocks + 2)\n    classes: optional number of classes to classify images into\n    training: Only used if training keras model with Estimator.  In other\n    scenarios it is handled automatically.\n\n  Returns:\n    A Keras model instance.\n  '
    input_shape = (32, 32, 3)
    img_input = layers.Input(shape=input_shape)
    if backend.image_data_format() == 'channels_first':
        x = layers.Lambda(lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)), name='transpose')(img_input)
        bn_axis = 1
    else:
        x = img_input
        bn_axis = 3
    x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    x = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, name='bn_conv1')(x, training=training)
    x = layers.Activation('relu')(x)
    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[16, 16], stage=2, conv_strides=(1, 1), training=training)
    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[32, 32], stage=3, conv_strides=(2, 2), training=training)
    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[64, 64], stage=4, conv_strides=(2, 2), training=training)
    rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
    x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)
    x = layers.Dense(classes, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=0.01), kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY), bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY), name='fc10')(x)
    inputs = img_input
    model = tf.keras.models.Model(inputs, x, name='resnet56')
    return model
resnet20 = functools.partial(resnet, num_blocks=3)
resnet32 = functools.partial(resnet, num_blocks=5)
resnet56 = functools.partial(resnet, num_blocks=9)
resnet10 = functools.partial(resnet, num_blocks=110)