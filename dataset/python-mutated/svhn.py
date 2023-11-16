import os
import numpy
try:
    from scipy import io
    _scipy_available = True
except Exception as e:
    _error = e
    _scipy_available = False
import chainer
from chainer.dataset import download
from chainer.datasets import tuple_dataset

def get_svhn(withlabel=True, scale=1.0, dtype=None, label_dtype=numpy.int32, add_extra=False):
    if False:
        i = 10
        return i + 15
    'Gets the SVHN dataset.\n\n    `The Street View House Numbers (SVHN) dataset\n    <http://ufldl.stanford.edu/housenumbers/>`_\n    is a dataset similar to MNIST but composed of cropped images of house\n    numbers.\n    The functionality of this function is identical to the counterpart for the\n    MNIST dataset (:func:`~chainer.datasets.get_mnist`),\n    with the exception that there is no ``ndim`` argument.\n\n    .. note::\n       `SciPy <https://www.scipy.org/>`_ is required to use this feature.\n\n    Args:\n        withlabel (bool): If ``True``, it returns datasets with labels. In this\n            case, each example is a tuple of an image and a label. Otherwise,\n            the datasets only contain images.\n        scale (float): Pixel value scale. If it is 1 (default), pixels are\n            scaled to the interval ``[0, 1]``.\n        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is\n            used by default (see :ref:`configuration`).\n        label_dtype: Data type of the labels.\n        add_extra: Use extra training set.\n\n    Returns:\n        If ``add_extra`` is ``False``, a tuple of two datasets (train and\n        test). Otherwise, a tuple of three datasets (train, test, and extra).\n        If ``withlabel`` is ``True``, all datasets are\n        :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both\n        datasets are arrays of images.\n\n    '
    if not _scipy_available:
        raise RuntimeError('SciPy is not available: %s' % _error)
    train_raw = _retrieve_svhn_training()
    dtype = chainer.get_dtype(dtype)
    train = _preprocess_svhn(train_raw, withlabel, scale, dtype, label_dtype)
    test_raw = _retrieve_svhn_test()
    test = _preprocess_svhn(test_raw, withlabel, scale, dtype, label_dtype)
    if add_extra:
        extra_raw = _retrieve_svhn_extra()
        extra = _preprocess_svhn(extra_raw, withlabel, scale, dtype, label_dtype)
        return (train, test, extra)
    else:
        return (train, test)

def _preprocess_svhn(raw, withlabel, scale, image_dtype, label_dtype):
    if False:
        for i in range(10):
            print('nop')
    images = raw['x'].transpose(3, 2, 0, 1)
    images = images.astype(image_dtype)
    images *= scale / 255.0
    labels = raw['y'].astype(label_dtype).flatten()
    labels[labels == 10] = 0
    if withlabel:
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images

def _retrieve_svhn_training():
    if False:
        while True:
            i = 10
    url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    return _retrieve_svhn('train.npz', url)

def _retrieve_svhn_test():
    if False:
        while True:
            i = 10
    url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    return _retrieve_svhn('test.npz', url)

def _retrieve_svhn_extra():
    if False:
        for i in range(10):
            print('nop')
    url = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
    return _retrieve_svhn('extra.npz', url)

def _retrieve_svhn(name, url):
    if False:
        print('Hello World!')
    root = download.get_dataset_directory('pfnet/chainer/svhn')
    path = os.path.join(root, name)
    return download.cache_or_load_file(path, lambda path: _make_npz(path, url), numpy.load)

def _make_npz(path, url):
    if False:
        while True:
            i = 10
    _path = download.cached_download(url)
    raw = io.loadmat(_path)
    images = raw['X'].astype(numpy.uint8)
    labels = raw['y'].astype(numpy.uint8)
    numpy.savez_compressed(path, x=images, y=labels)
    return {'x': images, 'y': labels}