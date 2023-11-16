"""
Cross-validation iterators for GAM

Author: Luca Puggini

"""
from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np

class BaseCrossValidator(with_metaclass(ABCMeta)):
    """
    The BaseCrossValidator class is a base class for all the iterators that
    split the data in train and test as for example KFolds or LeavePOut
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def split(self):
        if False:
            i = 10
            return i + 15
        pass

class KFold(BaseCrossValidator):
    """
    K-Folds cross validation iterator:
    Provides train/test indexes to split data in train test sets

    Parameters
    ----------
    k: int
        number of folds
    shuffle : bool
        If true, then the index is shuffled before splitting into train and
        test indices.

    Notes
    -----
    All folds except for last fold have size trunc(n/k), the last fold has
    the remainder.
    """

    def __init__(self, k_folds, shuffle=False):
        if False:
            for i in range(10):
                print('nop')
        self.nobs = None
        self.k_folds = k_folds
        self.shuffle = shuffle

    def split(self, X, y=None, label=None):
        if False:
            i = 10
            return i + 15
        'yield index split into train and test sets\n        '
        nobs = X.shape[0]
        index = np.array(range(nobs))
        if self.shuffle:
            np.random.shuffle(index)
        folds = np.array_split(index, self.k_folds)
        for fold in folds:
            test_index = np.zeros(nobs, dtype=bool)
            test_index[fold] = True
            train_index = np.logical_not(test_index)
            yield (train_index, test_index)