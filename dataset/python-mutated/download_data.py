"""Download MNIST, Omniglot datasets for Rebar."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import urllib
import gzip
import os
import config
import struct
import numpy as np
import cPickle as pickle
import datasets
MNIST_URL = 'see README'
MNIST_BINARIZED_URL = 'see README'
OMNIGLOT_URL = 'see README'
MNIST_FLOAT_TRAIN = 'train-images-idx3-ubyte'

def load_mnist_float(local_filename):
    if False:
        print('Hello World!')
    with open(local_filename, 'rb') as f:
        f.seek(4)
        (nimages, rows, cols) = struct.unpack('>iii', f.read(12))
        dim = rows * cols
        images = np.fromfile(f, dtype=np.dtype(np.ubyte))
        images = (images / 255.0).astype('float32').reshape((nimages, dim))
    return images
if __name__ == '__main__':
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    local_filename = os.path.join(config.DATA_DIR, MNIST_FLOAT_TRAIN)
    if not os.path.exists(local_filename):
        urllib.urlretrieve('%s/%s.gz' % (MNIST_URL, MNIST_FLOAT_TRAIN), local_filename + '.gz')
        with gzip.open(local_filename + '.gz', 'rb') as f:
            file_content = f.read()
        with open(local_filename, 'wb') as f:
            f.write(file_content)
        os.remove(local_filename + '.gz')
    mnist_float_train = load_mnist_float(local_filename)[:-10000]
    np.save(os.path.join(config.DATA_DIR, config.MNIST_FLOAT), mnist_float_train)
    splits = ['train', 'valid', 'test']
    mnist_binarized = []
    for split in splits:
        filename = 'binarized_mnist_%s.amat' % split
        url = '%s/binarized_mnist_%s.amat' % (MNIST_BINARIZED_URL, split)
        local_filename = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(local_filename):
            urllib.urlretrieve(url, local_filename)
        with open(local_filename, 'rb') as f:
            mnist_binarized.append((np.array([map(int, line.split()) for line in f.readlines()]).astype('float32'), None))
    with open(os.path.join(config.DATA_DIR, config.MNIST_BINARIZED), 'w') as out:
        pickle.dump(mnist_binarized, out)
    local_filename = os.path.join(config.DATA_DIR, config.OMNIGLOT)
    if not os.path.exists(local_filename):
        urllib.urlretrieve(OMNIGLOT_URL, local_filename)