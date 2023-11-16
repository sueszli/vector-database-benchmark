"""CIFAR100 small images classification dataset."""
import os
import numpy as np
from keras import backend
from keras.api_export import keras_export
from keras.datasets.cifar import load_batch
from keras.utils.file_utils import get_file

@keras_export('keras.datasets.cifar100.load_data')
def load_data(label_mode='fine'):
    if False:
        i = 10
        return i + 15
    'Loads the CIFAR100 dataset.\n\n    This is a dataset of 50,000 32x32 color training images and\n    10,000 test images, labeled over 100 fine-grained classes that are\n    grouped into 20 coarse-grained classes. See more info at the\n    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).\n\n    Args:\n        label_mode: one of `"fine"`, `"coarse"`.\n            If it is `"fine"`, the category labels\n            are the fine-grained labels, and if it is `"coarse"`,\n            the output labels are the coarse-grained superclasses.\n\n    Returns:\n        Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.\n\n    **`x_train`**: `uint8` NumPy array of grayscale image data with shapes\n      `(50000, 32, 32, 3)`, containing the training data. Pixel values range\n      from 0 to 255.\n\n    **`y_train`**: `uint8` NumPy array of labels (integers in range 0-99)\n      with shape `(50000, 1)` for the training data.\n\n    **`x_test`**: `uint8` NumPy array of grayscale image data with shapes\n      `(10000, 32, 32, 3)`, containing the test data. Pixel values range\n      from 0 to 255.\n\n    **`y_test`**: `uint8` NumPy array of labels (integers in range 0-99)\n      with shape `(10000, 1)` for the test data.\n\n    Example:\n\n    ```python\n    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()\n    assert x_train.shape == (50000, 32, 32, 3)\n    assert x_test.shape == (10000, 32, 32, 3)\n    assert y_train.shape == (50000, 1)\n    assert y_test.shape == (10000, 1)\n    ```\n    '
    if label_mode not in ['fine', 'coarse']:
        raise ValueError(f'`label_mode` must be one of `"fine"`, `"coarse"`. Received: label_mode={label_mode}.')
    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(fname=dirname, origin=origin, untar=True, file_hash='85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7')
    fpath = os.path.join(path, 'train')
    (x_train, y_train) = load_batch(fpath, label_key=label_mode + '_labels')
    fpath = os.path.join(path, 'test')
    (x_test, y_test) = load_batch(fpath, label_key=label_mode + '_labels')
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    if backend.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    return ((x_train, y_train), (x_test, y_test))