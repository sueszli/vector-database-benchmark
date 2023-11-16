import numpy as np
from bigdl.dllib.feature.dataset import base

def load_data(path='boston_housing.npz', dest_dir='/tmp/.zoo/dataset', test_split=0.2):
    if False:
        i = 10
        return i + 15
    'Loads the Boston Housing dataset, the source url of download\n       is copied from keras.datasets\n    # Arguments\n        dest_dir: where to cache the data (relative to `~/.zoo/dataset`).\n        nb_words: number of words to keep, the words are already indexed by frequency\n                  so that the less frequent words would be abandoned\n        oov_char: index to pad the abandoned words, if None, one abandoned word\n                  would be taken place with its next word and total length -= 1\n        test_split: the ratio to split part of dataset to test data,\n                    the remained data would be train data\n\n    # Returns\n        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.\n    '
    path = base.maybe_download(path, dest_dir, 'https://s3.amazonaws.com/keras-datasets/boston_housing.npz')
    with np.load(path) as f:
        x = f['x']
        y = f['y']
    shuffle_by_seed([x, y])
    split_index = int(len(x) * (1 - test_split))
    (x_train, y_train) = (x[:split_index], y[:split_index])
    (x_test, y_test) = (x[split_index:], y[split_index:])
    return ((x_train, y_train), (x_test, y_test))

def shuffle_by_seed(arr_list, seed=0):
    if False:
        while True:
            i = 10
    for arr in arr_list:
        np.random.seed(seed)
        np.random.shuffle(arr)