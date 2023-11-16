"""Utility functions for loading the automobile data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
COLUMN_TYPES = collections.OrderedDict([('symboling', int), ('normalized-losses', float), ('make', str), ('fuel-type', str), ('aspiration', str), ('num-of-doors', str), ('body-style', str), ('drive-wheels', str), ('engine-location', str), ('wheel-base', float), ('length', float), ('width', float), ('height', float), ('curb-weight', float), ('engine-type', str), ('num-of-cylinders', str), ('engine-size', float), ('fuel-system', str), ('bore', float), ('stroke', float), ('compression-ratio', float), ('horsepower', float), ('peak-rpm', float), ('city-mpg', float), ('highway-mpg', float), ('price', float)])

def raw_dataframe():
    if False:
        return 10
    'Load the automobile data set as a pd.DataFrame.'
    path = tf.keras.utils.get_file(URL.split('/')[-1], URL)
    df = pd.read_csv(path, names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES, na_values='?')
    return df

def load_data(y_name='price', train_fraction=0.7, seed=None):
    if False:
        i = 10
        return i + 15
    'Load the automobile data set and split it train/test and features/label.\n\n  A description of the data is available at:\n    https://archive.ics.uci.edu/ml/datasets/automobile\n\n  The data itself can be found at:\n    https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data\n\n  Args:\n    y_name: the column to return as the label.\n    train_fraction: the fraction of the data set to use for training.\n    seed: The random seed to use when shuffling the data. `None` generates a\n      unique shuffle every run.\n  Returns:\n    a pair of pairs where the first pair is the training data, and the second\n    is the test data:\n    `(x_train, y_train), (x_test, y_test) = load_data(...)`\n    `x` contains a pandas DataFrame of features, while `y` contains the label\n    array.\n  '
    data = raw_dataframe()
    data = data.dropna()
    np.random.seed(seed)
    x_train = data.sample(frac=train_fraction, random_state=seed)
    x_test = data.drop(x_train.index)
    y_train = x_train.pop(y_name)
    y_test = x_test.pop(y_name)
    return ((x_train, y_train), (x_test, y_test))

def make_dataset(batch_sz, x, y=None, shuffle=False, shuffle_buffer_size=1000):
    if False:
        print('Hello World!')
    'Create a slice Dataset from a pandas DataFrame and labels'

    def input_fn():
        if False:
            i = 10
            return i + 15
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dict(x))
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz).repeat()
        else:
            dataset = dataset.batch(batch_sz)
        return dataset
    return input_fn