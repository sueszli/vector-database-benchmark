import os
import numpy
import chainer
from chainer.dataset import download
from chainer.datasets._mnist_helper import make_npz
from chainer.datasets._mnist_helper import preprocess_mnist
_kuzushiji_mnist_labels = [('o', u'お'), ('ki', u'き'), ('su', u'す'), ('tsu', u'つ'), ('na', u'な'), ('ha', u'は'), ('ma', u'ま'), ('ya', u'や'), ('re', u'れ'), ('wo', u'を')]

def get_kuzushiji_mnist_labels():
    if False:
        for i in range(10):
            print('nop')
    'Provides a list of labels for the Kuzushiji-MNIST dataset.\n\n    Returns:\n        List of labels in the form of tuples. Each tuple contains the\n        character name in romaji as a string value and the unicode codepoint\n        for the character.\n\n    '
    return _kuzushiji_mnist_labels

def get_kuzushiji_mnist(withlabel=True, ndim=1, scale=1.0, dtype=None, label_dtype=numpy.int32, rgb_format=False):
    if False:
        for i in range(10):
            print('nop')
    'Gets the Kuzushiji-MNIST dataset.\n\n    `Kuzushiji-MNIST (KMNIST) <http://codh.rois.ac.jp/kmnist/>`_ is a set of\n    hand-written Japanese characters represented by grey-scale 28x28 images.\n    In the original images, each pixel is represented by one-byte unsigned\n    integer. This function scales the pixels to floating point values in the\n    interval ``[0, scale]``.\n\n    This function returns the training set and the test set of the official\n    KMNIST dataset. If ``withlabel`` is ``True``, each dataset consists of\n    tuples of images and labels, otherwise it only consists of images.\n\n    Args:\n        withlabel (bool): If ``True``, it returns datasets with labels. In this\n            case, each example is a tuple of an image and a label. Otherwise,\n            the datasets only contain images.\n        ndim (int): Number of dimensions of each image. The shape of each image\n            is determined depending on ``ndim`` as follows:\n\n            - ``ndim == 1``: the shape is ``(784,)``\n            - ``ndim == 2``: the shape is ``(28, 28)``\n            - ``ndim == 3``: the shape is ``(1, 28, 28)``\n\n        scale (float): Pixel value scale. If it is 1 (default), pixels are\n            scaled to the interval ``[0, 1]``.\n        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is\n            used by default (see :ref:`configuration`).\n        label_dtype: Data type of the labels.\n        rgb_format (bool): if ``ndim == 3`` and ``rgb_format`` is ``True``, the\n            image will be converted to rgb format by duplicating the channels\n            so the image shape is (3, 28, 28). Default is ``False``.\n\n    Returns:\n        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets\n        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both\n        datasets are arrays of images.\n\n    '
    dtype = chainer.get_dtype(dtype)
    train_raw = _retrieve_kuzushiji_mnist_training()
    train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype, label_dtype, rgb_format)
    test_raw = _retrieve_kuzushiji_mnist_test()
    test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype, label_dtype, rgb_format)
    return (train, test)

def _retrieve_kuzushiji_mnist_training():
    if False:
        print('Hello World!')
    base_url = 'http://codh.rois.ac.jp/'
    urls = [base_url + 'kmnist/dataset/kmnist/train-images-idx3-ubyte.gz', base_url + 'kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz']
    return _retrieve_kuzushiji_mnist('train.npz', urls)

def _retrieve_kuzushiji_mnist_test():
    if False:
        i = 10
        return i + 15
    base_url = 'http://codh.rois.ac.jp/'
    urls = [base_url + 'kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz', base_url + 'kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz']
    return _retrieve_kuzushiji_mnist('test.npz', urls)

def _retrieve_kuzushiji_mnist(name, urls):
    if False:
        print('Hello World!')
    root = download.get_dataset_directory('pfnet/chainer/kuzushiji_mnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(path, lambda path: make_npz(path, urls), numpy.load)