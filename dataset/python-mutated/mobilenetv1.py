"""MobileNet for ImageNet."""
import os
import tensorflow as tf
from tensorlayer import logging
from tensorlayer.files import assign_weights, load_npz, maybe_download_and_extract
from tensorlayer.layers import BatchNorm, Conv2d, DepthwiseConv2d, Flatten, GlobalMeanPool2d, Input, Reshape
from tensorlayer.models import Model
__all__ = ['MobileNetV1']
layer_names = ['conv', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5', 'depth6', 'depth7', 'depth8', 'depth9', 'depth10', 'depth11', 'depth12', 'depth13', 'globalmeanpool', 'reshape', 'out']
n_filters = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]

def conv_block(n, n_filter, filter_size=(3, 3), strides=(1, 1), name='conv_block'):
    if False:
        while True:
            i = 10
    n = Conv2d(n_filter, filter_size, strides, b_init=None, name=name + '.conv')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm')(n)
    return n

def depthwise_conv_block(n, n_filter, strides=(1, 1), name='depth_block'):
    if False:
        for i in range(10):
            print('nop')
    n = DepthwiseConv2d((3, 3), strides, b_init=None, name=name + '.depthwise')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm1')(n)
    n = Conv2d(n_filter, (1, 1), (1, 1), b_init=None, name=name + '.conv')(n)
    n = BatchNorm(decay=0.99, act=tf.nn.relu6, name=name + '.batchnorm2')(n)
    return n

def restore_params(network, path='models'):
    if False:
        while True:
            i = 10
    logging.info('Restore pre-trained parameters')
    maybe_download_and_extract('mobilenet.npz', path, 'https://github.com/tensorlayer/pretrained-models/raw/master/models/', expected_bytes=25600116)
    params = load_npz(name=os.path.join(path, 'mobilenet.npz'))
    assign_weights(params[:len(network.all_weights)], network)
    del params

def MobileNetV1(pretrained=False, end_with='out', name=None):
    if False:
        while True:
            i = 10
    'Pre-trained MobileNetV1 model (static mode). Input shape [?, 224, 224, 3], value range [0, 1].\n\n    Parameters\n    ----------\n    pretrained : boolean\n        Whether to load pretrained weights. Default False.\n    end_with : str\n        The end point of the model [conv, depth1, depth2 ... depth13, globalmeanpool, out]. Default ``out`` i.e. the whole model.\n    name : None or str\n        Name for this model.\n\n    Examples\n    ---------\n    Classify ImageNet classes, see `tutorial_models_mobilenetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_mobilenetv1.py>`__\n\n    >>> # get the whole model with pretrained weights\n    >>> mobilenetv1 = tl.models.MobileNetV1(pretrained=True)\n    >>> # use for inferencing\n    >>> output = mobilenetv1(img1, is_train=False)\n    >>> prob = tf.nn.softmax(output)[0].numpy()\n\n    Extract features and Train a classifier with 100 classes\n\n    >>> # get model without the last layer\n    >>> cnn = tl.models.MobileNetV1(pretrained=True, end_with=\'reshape\').as_layer()\n    >>> # add one more layer and build new model\n    >>> ni = Input([None, 224, 224, 3], name="inputs")\n    >>> nn = cnn(ni)\n    >>> nn = Conv2d(100, (1, 1), (1, 1), name=\'out\')(nn)\n    >>> nn = Flatten(name=\'flatten\')(nn)\n    >>> model = tl.models.Model(inputs=ni, outputs=nn)\n    >>> # train your own classifier (only update the last layer)\n    >>> train_params = model.get_layer(\'out\').trainable_weights\n\n    Returns\n    -------\n        static MobileNetV1.\n    '
    ni = Input([None, 224, 224, 3], name='input')
    for i in range(len(layer_names)):
        if i == 0:
            n = conv_block(ni, n_filters[i], strides=(2, 2), name=layer_names[i])
        elif layer_names[i] in ['depth2', 'depth4', 'depth6', 'depth12']:
            n = depthwise_conv_block(n, n_filters[i], strides=(2, 2), name=layer_names[i])
        elif layer_names[i] == 'globalmeanpool':
            n = GlobalMeanPool2d(name='globalmeanpool')(n)
        elif layer_names[i] == 'reshape':
            n = Reshape([-1, 1, 1, 1024], name='reshape')(n)
        elif layer_names[i] == 'out':
            n = Conv2d(1000, (1, 1), (1, 1), name='out')(n)
            n = Flatten(name='flatten')(n)
        else:
            n = depthwise_conv_block(n, n_filters[i], name=layer_names[i])
        if layer_names[i] == end_with:
            break
    network = Model(inputs=ni, outputs=n, name=name)
    if pretrained:
        restore_params(network)
    return network