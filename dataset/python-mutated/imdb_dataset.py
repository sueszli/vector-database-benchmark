import gzip
import os
import numpy as np
import six.moves.cPickle as pickle
from tensorlayer.files.utils import maybe_download_and_extract
__all__ = ['load_imdb_dataset']

def load_imdb_dataset(path='data', nb_words=None, skip_top=0, maxlen=None, test_split=0.2, seed=113, start_char=1, oov_char=2, index_from=3):
    if False:
        return 10
    "Load IMDB dataset.\n\n    Parameters\n    ----------\n    path : str\n        The path that the data is downloaded to, defaults is ``data/imdb/``.\n    nb_words : int\n        Number of words to get.\n    skip_top : int\n        Top most frequent words to ignore (they will appear as oov_char value in the sequence data).\n    maxlen : int\n        Maximum sequence length. Any longer sequence will be truncated.\n    seed : int\n        Seed for reproducible data shuffling.\n    start_char : int\n        The start of a sequence will be marked with this character. Set to 1 because 0 is usually the padding character.\n    oov_char : int\n        Words that were cut out because of the num_words or skip_top limit will be replaced with this character.\n    index_from : int\n        Index actual words with this index and higher.\n\n    Examples\n    --------\n    >>> X_train, y_train, X_test, y_test = tl.files.load_imdb_dataset(\n    ...                                 nb_words=20000, test_split=0.2)\n    >>> print('X_train.shape', X_train.shape)\n    (20000,)  [[1, 62, 74, ... 1033, 507, 27],[1, 60, 33, ... 13, 1053, 7]..]\n    >>> print('y_train.shape', y_train.shape)\n    (20000,)  [1 0 0 ..., 1 0 1]\n\n    References\n    -----------\n    - `Modified from keras. <https://github.com/fchollet/keras/blob/master/keras/datasets/imdb.py>`__\n\n    "
    path = os.path.join(path, 'imdb')
    filename = 'imdb.pkl'
    url = 'https://s3.amazonaws.com/text-datasets/'
    maybe_download_and_extract(filename, path, url)
    if filename.endswith('.gz'):
        f = gzip.open(os.path.join(path, filename), 'rb')
    else:
        f = open(os.path.join(path, filename), 'rb')
    (X, labels) = pickle.load(f)
    f.close()
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]
    if maxlen:
        new_X = []
        new_labels = []
        for (x, y) in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' + str(maxlen) + ', no sequence was kept. Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])
    if oov_char is not None:
        X = [[oov_char if w >= nb_words or w < skip_top else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if w >= nb_words or w < skip_top:
                    nx.append(w)
            nX.append(nx)
        X = nX
    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])
    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])
    return (X_train, y_train, X_test, y_test)