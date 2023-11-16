"""
Utility functions to load Kaggle Otto Group Challenge Data.

Since these data/functions are used in many notebooks, it is better
to centralise functions to load and manipulate data so
to not replicate code.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def load_data(path, train=True):
    if False:
        while True:
            i = 10
    'Load data from a CSV File\n    \n    Parameters\n    ----------\n    path: str\n        The path to the CSV file\n        \n    train: bool (default True)\n        Decide whether or not data are *training data*.\n        If True, some random shuffling is applied.\n        \n    Return\n    ------\n    X: numpy.ndarray \n        The data as a multi dimensional array of floats\n    ids: numpy.ndarray\n        A vector of ids for each sample\n    '
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)
        (X, labels) = (X[:, 1:-1].astype(np.float32), X[:, -1])
        return (X, labels)
    else:
        (X, ids) = (X[:, 1:].astype(np.float32), X[:, 0].astype(str))
        return (X, ids)

def preprocess_data(X, scaler=None):
    if False:
        while True:
            i = 10
    'Preprocess input data by standardise features \n    by removing the mean and scaling to unit variance'
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return (X, scaler)

def preprocess_labels(labels, encoder=None, categorical=True):
    if False:
        return 10
    'Encode labels with values among 0 and `n-classes-1`'
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return (y, encoder)