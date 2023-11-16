import os
import pickle
import sys
import numpy as np
from tensorlayer import logging
from tensorlayer.files.utils import maybe_download_and_extract
__all__ = ['load_cifar10_dataset']

def load_cifar10_dataset(shape=(-1, 32, 32, 3), path='data', plotable=False):
    if False:
        while True:
            i = 10
    'Load CIFAR-10 dataset.\n\n    It consists of 60000 32x32 colour images in 10 classes, with\n    6000 images per class. There are 50000 training images and 10000 test images.\n\n    The dataset is divided into five training batches and one test batch, each with\n    10000 images. The test batch contains exactly 1000 randomly-selected images from\n    each class. The training batches contain the remaining images in random order,\n    but some training batches may contain more images from one class than another.\n    Between them, the training batches contain exactly 5000 images from each class.\n\n    Parameters\n    ----------\n    shape : tupe\n        The shape of digit images e.g. (-1, 3, 32, 32) and (-1, 32, 32, 3).\n    path : str\n        The path that the data is downloaded to, defaults is ``data/cifar10/``.\n    plotable : boolean\n        Whether to plot some image examples, False as default.\n\n    Examples\n    --------\n    >>> X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))\n\n    References\n    ----------\n    - `CIFAR website <https://www.cs.toronto.edu/~kriz/cifar.html>`__\n    - `Data download link <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>`__\n    - `<https://teratail.com/questions/28932>`__\n\n    '
    path = os.path.join(path, 'cifar10')
    logging.info('Load or Download cifar10 > {}'.format(path))

    def unpickle(file):
        if False:
            i = 10
            return i + 15
        fp = open(file, 'rb')
        if sys.version_info.major == 2:
            data = pickle.load(fp)
        elif sys.version_info.major == 3:
            data = pickle.load(fp, encoding='latin-1')
        else:
            raise RuntimeError('Sys Version Unsupported')
        fp.close()
        return data
    filename = 'cifar-10-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/'
    maybe_download_and_extract(filename, path, url, extract=True)
    X_train = None
    y_train = []
    for i in range(1, 6):
        data_dic = unpickle(os.path.join(path, 'cifar-10-batches-py/', 'data_batch_{}'.format(i)))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        y_train += data_dic['labels']
    test_data_dic = unpickle(os.path.join(path, 'cifar-10-batches-py/', 'test_batch'))
    X_test = test_data_dic['data']
    y_test = np.array(test_data_dic['labels'])
    if shape == (-1, 3, 32, 32):
        X_test = X_test.reshape(shape)
        X_train = X_train.reshape(shape)
    elif shape == (-1, 32, 32, 3):
        X_test = X_test.reshape(shape, order='F')
        X_train = X_train.reshape(shape, order='F')
        X_test = np.transpose(X_test, (0, 2, 1, 3))
        X_train = np.transpose(X_train, (0, 2, 1, 3))
    else:
        X_test = X_test.reshape(shape)
        X_train = X_train.reshape(shape)
    y_train = np.array(y_train)
    if plotable:
        logging.info('\nCIFAR-10')
        import matplotlib.pyplot as plt
        fig = plt.figure(1)
        logging.info('Shape of a training image: X_train[0] %s' % X_train[0].shape)
        plt.ion()
        count = 1
        for _ in range(10):
            for _ in range(10):
                _ = fig.add_subplot(10, 10, count)
                if shape == (-1, 3, 32, 32):
                    plt.imshow(np.transpose(X_train[count - 1], (1, 2, 0)), interpolation='nearest')
                elif shape == (-1, 32, 32, 3):
                    plt.imshow(X_train[count - 1], interpolation='nearest')
                else:
                    raise Exception("Do not support the given 'shape' to plot the image examples")
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                count = count + 1
        plt.draw()
        plt.pause(3)
        logging.info('X_train: %s' % X_train.shape)
        logging.info('y_train: %s' % y_train.shape)
        logging.info('X_test:  %s' % X_test.shape)
        logging.info('y_test:  %s' % y_test.shape)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    return (X_train, y_train, X_test, y_test)