"""SqueezeNet for ImageNet."""
import os
import tensorflow as tf
from tensorlayer import logging
from tensorlayer.files import assign_weights, load_npz, maybe_download_and_extract
from tensorlayer.layers import Concat, Conv2d, Dropout, GlobalMeanPool2d, Input, Lambda, MaxPool2d
from tensorlayer.models import Model
__all__ = ['SqueezeNetV1']
layer_names = ['conv1', 'maxpool1', 'fire2', 'fire3', 'fire4', 'fire5', 'fire6', 'fire7', 'fire8', 'fire9', 'drop1', 'out']
n_filters = [16, 16, 32, 32, 48, 48, 64, 64]

def fire_block(n, n_filter, max_pool=False, name='fire_block'):
    if False:
        return 10
    n = Conv2d(n_filter, (1, 1), (1, 1), tf.nn.relu, 'SAME', name=name + '.squeeze1x1')(n)
    n1 = Conv2d(n_filter * 4, (1, 1), (1, 1), tf.nn.relu, 'SAME', name=name + '.expand1x1')(n)
    n2 = Conv2d(n_filter * 4, (3, 3), (1, 1), tf.nn.relu, 'SAME', name=name + '.expand3x3')(n)
    n = Concat(-1, name=name + '.concat')([n1, n2])
    if max_pool:
        n = MaxPool2d((3, 3), (2, 2), 'VALID', name=name + '.max')(n)
    return n

def restore_params(network, path='models'):
    if False:
        i = 10
        return i + 15
    logging.info('Restore pre-trained parameters')
    maybe_download_and_extract('squeezenet.npz', path, 'https://github.com/tensorlayer/pretrained-models/raw/master/models/', expected_bytes=7405613)
    params = load_npz(name=os.path.join(path, 'squeezenet.npz'))
    assign_weights(params[:len(network.all_weights)], network)
    del params

def SqueezeNetV1(pretrained=False, end_with='out', name=None):
    if False:
        for i in range(10):
            print('nop')
    'Pre-trained SqueezeNetV1 model (static mode). Input shape [?, 224, 224, 3], value range [0, 1].\n\n    Parameters\n    ------------\n    pretrained : boolean\n        Whether to load pretrained weights. Default False.\n    end_with : str\n        The end point of the model [conv1, maxpool1, fire2, fire3, fire4, ..., out]. Default ``out`` i.e. the whole model.\n    name : None or str\n        Name for this model.\n\n    Examples\n    ---------\n    Classify ImageNet classes, see `tutorial_models_squeezenetv1.py <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_models_squeezenetv1.py>`__\n\n    >>> # get the whole model\n    >>> squeezenet = tl.models.SqueezeNetV1(pretrained=True)\n    >>> # use for inferencing\n    >>> output = squeezenet(img1, is_train=False)\n    >>> prob = tf.nn.softmax(output)[0].numpy()\n\n    Extract features and Train a classifier with 100 classes\n\n    >>> # get model without the last layer\n    >>> cnn = tl.models.SqueezeNetV1(pretrained=True, end_with=\'drop1\').as_layer()\n    >>> # add one more layer and build new model\n    >>> ni = Input([None, 224, 224, 3], name="inputs")\n    >>> nn = cnn(ni)\n    >>> nn = Conv2d(100, (1, 1), (1, 1), padding=\'VALID\', name=\'conv10\')(nn)\n    >>> nn = GlobalMeanPool2d(name=\'globalmeanpool\')(nn)\n    >>> model = tl.models.Model(inputs=ni, outputs=nn)\n    >>> # train your own classifier (only update the last layer)\n    >>> train_params = model.get_layer(\'conv10\').trainable_weights\n\n    Returns\n    -------\n        static SqueezeNetV1.\n\n    '
    ni = Input([None, 224, 224, 3], name='input')
    n = Lambda(lambda x: x * 255, name='scale')(ni)
    for i in range(len(layer_names)):
        if layer_names[i] == 'conv1':
            n = Conv2d(64, (3, 3), (2, 2), tf.nn.relu, 'SAME', name='conv1')(n)
        elif layer_names[i] == 'maxpool1':
            n = MaxPool2d((3, 3), (2, 2), 'VALID', name='maxpool1')(n)
        elif layer_names[i] == 'drop1':
            n = Dropout(keep=0.5, name='drop1')(n)
        elif layer_names[i] == 'out':
            n = Conv2d(1000, (1, 1), (1, 1), padding='VALID', name='conv10')(n)
            n = GlobalMeanPool2d(name='globalmeanpool')(n)
        elif layer_names[i] in ['fire3', 'fire5']:
            n = fire_block(n, n_filters[i - 2], max_pool=True, name=layer_names[i])
        else:
            n = fire_block(n, n_filters[i - 2], max_pool=False, name=layer_names[i])
        if layer_names[i] == end_with:
            break
    network = Model(inputs=ni, outputs=n, name=name)
    if pretrained:
        restore_params(network)
    return network