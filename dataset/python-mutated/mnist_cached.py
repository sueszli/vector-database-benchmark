import errno
import os
from functools import reduce
import numpy as np
import torch
from torch.utils.data import DataLoader
from pyro.contrib.examples.util import MNIST, get_data_directory

def fn_x_mnist(x, use_cuda):
    if False:
        for i in range(10):
            print('nop')
    xp = x * (1.0 / 255)
    xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])
    xp = xp.view(-1, xp_1d_size)
    if use_cuda:
        xp = xp.cuda()
    return xp

def fn_y_mnist(y, use_cuda):
    if False:
        for i in range(10):
            print('nop')
    yp = torch.zeros(y.size(0), 10)
    if use_cuda:
        yp = yp.cuda()
        y = y.cuda()
    yp = yp.scatter_(1, y.view(-1, 1), 1.0)
    return yp

def get_ss_indices_per_class(y, sup_per_class):
    if False:
        while True:
            i = 10
    n_idxs = y.size()[0]
    idxs_per_class = {j: [] for j in range(10)}
    for i in range(n_idxs):
        curr_y = y[i]
        for j in range(10):
            if curr_y[j] == 1:
                idxs_per_class[j].append(i)
                break
    idxs_sup = []
    idxs_unsup = []
    for j in range(10):
        np.random.shuffle(idxs_per_class[j])
        idxs_sup.extend(idxs_per_class[j][:sup_per_class])
        idxs_unsup.extend(idxs_per_class[j][sup_per_class:len(idxs_per_class[j])])
    return (idxs_sup, idxs_unsup)

def split_sup_unsup_valid(X, y, sup_num, validation_num=10000):
    if False:
        while True:
            i = 10
    '\n    helper function for splitting the data into supervised, un-supervised and validation parts\n    :param X: images\n    :param y: labels (digits)\n    :param sup_num: what number of examples is supervised\n    :param validation_num: what number of last examples to use for validation\n    :return: splits of data by sup_num number of supervised examples\n    '
    X_valid = X[-validation_num:]
    y_valid = y[-validation_num:]
    X = X[0:-validation_num]
    y = y[0:-validation_num]
    assert sup_num % 10 == 0, 'unable to have equal number of images per class'
    sup_per_class = int(sup_num / 10)
    (idxs_sup, idxs_unsup) = get_ss_indices_per_class(y, sup_per_class)
    X_sup = X[idxs_sup]
    y_sup = y[idxs_sup]
    X_unsup = X[idxs_unsup]
    y_unsup = y[idxs_unsup]
    return (X_sup, y_sup, X_unsup, y_unsup, X_valid, y_valid)

def print_distribution_labels(y):
    if False:
        return 10
    '\n    helper function for printing the distribution of class labels in a dataset\n    :param y: tensor of class labels given as one-hots\n    :return: a dictionary of counts for each label from y\n    '
    counts = {j: 0 for j in range(10)}
    for i in range(y.size()[0]):
        for j in range(10):
            if y[i][j] == 1:
                counts[j] += 1
                break
    print(counts)

class MNISTCached(MNIST):
    """
    a wrapper around MNIST to load and cache the transformed data
    once at the beginning of the inference
    """
    train_data_size = 50000
    (train_data_sup, train_labels_sup) = (None, None)
    (train_data_unsup, train_labels_unsup) = (None, None)
    validation_size = 10000
    (data_valid, labels_valid) = (None, None)
    test_size = 10000

    def __init__(self, mode, sup_num, use_cuda=True, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, train=mode in ['sup', 'unsup', 'valid'], **kwargs)

        def transform(x):
            if False:
                return 10
            return fn_x_mnist(x, use_cuda)

        def target_transform(y):
            if False:
                while True:
                    i = 10
            return fn_y_mnist(y, use_cuda)
        self.mode = mode
        assert mode in ['sup', 'unsup', 'test', 'valid'], 'invalid train/test option values'
        if mode in ['sup', 'unsup', 'valid']:
            if transform is not None:
                self.data = transform(self.data.float())
            if target_transform is not None:
                self.targets = target_transform(self.targets)
            if MNISTCached.train_data_sup is None:
                if sup_num is None:
                    assert mode == 'unsup'
                    (MNISTCached.train_data_unsup, MNISTCached.train_labels_unsup) = (self.data, self.targets)
                else:
                    (MNISTCached.train_data_sup, MNISTCached.train_labels_sup, MNISTCached.train_data_unsup, MNISTCached.train_labels_unsup, MNISTCached.data_valid, MNISTCached.labels_valid) = split_sup_unsup_valid(self.data, self.targets, sup_num)
            if mode == 'sup':
                (self.data, self.targets) = (MNISTCached.train_data_sup, MNISTCached.train_labels_sup)
            elif mode == 'unsup':
                self.data = MNISTCached.train_data_unsup
                self.targets = torch.Tensor(MNISTCached.train_labels_unsup.shape[0]).view(-1, 1) * np.nan
            else:
                (self.data, self.targets) = (MNISTCached.data_valid, MNISTCached.labels_valid)
        else:
            if transform is not None:
                self.data = transform(self.data.float())
            if target_transform is not None:
                self.targets = target_transform(self.targets)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param index: Index or slice object\n        :returns tuple: (image, target) where target is index of the target class.\n        '
        if self.mode in ['sup', 'unsup', 'valid']:
            (img, target) = (self.data[index], self.targets[index])
        elif self.mode == 'test':
            (img, target) = (self.data[index], self.targets[index])
        else:
            assert False, 'invalid mode: {}'.format(self.mode)
        return (img, target)

def setup_data_loaders(dataset, use_cuda, batch_size, sup_num=None, root=None, download=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n        helper function for setting up pytorch data loaders for a semi-supervised dataset\n    :param dataset: the data to use\n    :param use_cuda: use GPU(s) for training\n    :param batch_size: size of a batch of data to output when iterating over the data loaders\n    :param sup_num: number of supervised data examples\n    :param download: download the dataset (if it doesn't exist already)\n    :param kwargs: other params for the pytorch data loader\n    :return: three data loaders: (supervised data for training, un-supervised data for training,\n                                  supervised data for testing)\n    "
    if root is None:
        root = get_data_directory(__file__)
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 0, 'pin_memory': False}
    cached_data = {}
    loaders = {}
    for mode in ['unsup', 'test', 'sup', 'valid']:
        if sup_num is None and mode == 'sup':
            return (loaders['unsup'], loaders['test'])
        cached_data[mode] = dataset(root=root, mode=mode, download=download, sup_num=sup_num, use_cuda=use_cuda)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs)
    return loaders

def mkdir_p(path):
    if False:
        return 10
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
EXAMPLE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DATA_DIR = os.path.join(EXAMPLE_DIR, 'data')
RESULTS_DIR = os.path.join(EXAMPLE_DIR, 'results')