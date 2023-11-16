"""
Utilities for cross validation.

taken from scikits.learn

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux    <gael.varoquaux@normalesup.org>
# License: BSD Style.
# $Id$

changes to code by josef-pktd:
 - docstring formatting: underlines of headers

"""
from statsmodels.compat.python import lrange
import numpy as np
from itertools import combinations

class LeaveOneOut:
    """
    Leave-One-Out cross validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, n):
        if False:
            print('Hello World!')
        '\n        Leave-One-Out cross validation iterator:\n        Provides train/test indexes to split data in train test sets\n\n        Parameters\n        ----------\n        n: int\n            Total number of elements\n\n        Examples\n        --------\n        >>> from scikits.learn import cross_val\n        >>> X = [[1, 2], [3, 4]]\n        >>> y = [1, 2]\n        >>> loo = cross_val.LeaveOneOut(2)\n        >>> for train_index, test_index in loo:\n        ...    print "TRAIN:", train_index, "TEST:", test_index\n        ...    X_train, X_test, y_train, y_test = cross_val.split(train_index, test_index, X, y)\n        ...    print X_train, X_test, y_train, y_test\n        TRAIN: [False  True] TEST: [ True False]\n        [[3 4]] [[1 2]] [2] [1]\n        TRAIN: [ True False] TEST: [False  True]\n        [[1 2]] [[3 4]] [1] [2]\n        '
        self.n = n

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        n = self.n
        for i in range(n):
            test_index = np.zeros(n, dtype=bool)
            test_index[i] = True
            train_index = np.logical_not(test_index)
            yield (train_index, test_index)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s.%s(n=%i)' % (self.__class__.__module__, self.__class__.__name__, self.n)

class LeavePOut:
    """
    Leave-P-Out cross validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, n, p):
        if False:
            while True:
                i = 10
        '\n        Leave-P-Out cross validation iterator:\n        Provides train/test indexes to split data in train test sets\n\n        Parameters\n        ----------\n        n: int\n            Total number of elements\n        p: int\n            Size test sets\n\n        Examples\n        --------\n        >>> from scikits.learn import cross_val\n        >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]\n        >>> y = [1, 2, 3, 4]\n        >>> lpo = cross_val.LeavePOut(4, 2)\n        >>> for train_index, test_index in lpo:\n        ...    print "TRAIN:", train_index, "TEST:", test_index\n        ...    X_train, X_test, y_train, y_test = cross_val.split(train_index, test_index, X, y)\n        TRAIN: [False False  True  True] TEST: [ True  True False False]\n        TRAIN: [False  True False  True] TEST: [ True False  True False]\n        TRAIN: [False  True  True False] TEST: [ True False False  True]\n        TRAIN: [ True False False  True] TEST: [False  True  True False]\n        TRAIN: [ True False  True False] TEST: [False  True False  True]\n        TRAIN: [ True  True False False] TEST: [False False  True  True]\n        '
        self.n = n
        self.p = p

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        n = self.n
        p = self.p
        comb = combinations(lrange(n), p)
        for idx in comb:
            test_index = np.zeros(n, dtype=bool)
            test_index[np.array(idx)] = True
            train_index = np.logical_not(test_index)
            yield (train_index, test_index)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '%s.%s(n=%i, p=%i)' % (self.__class__.__module__, self.__class__.__name__, self.n, self.p)

class KFold:
    """
    K-Folds cross validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, n, k):
        if False:
            i = 10
            return i + 15
        '\n        K-Folds cross validation iterator:\n        Provides train/test indexes to split data in train test sets\n\n        Parameters\n        ----------\n        n: int\n            Total number of elements\n        k: int\n            number of folds\n\n        Examples\n        --------\n        >>> from scikits.learn import cross_val\n        >>> X = [[1, 2], [3, 4], [1, 2], [3, 4]]\n        >>> y = [1, 2, 3, 4]\n        >>> kf = cross_val.KFold(4, k=2)\n        >>> for train_index, test_index in kf:\n        ...    print "TRAIN:", train_index, "TEST:", test_index\n        ...    X_train, X_test, y_train, y_test = cross_val.split(train_index, test_index, X, y)\n        TRAIN: [False False  True  True] TEST: [ True  True False False]\n        TRAIN: [ True  True False False] TEST: [False False  True  True]\n\n        Notes\n        -----\n        All the folds have size trunc(n/k), the last one has the complementary\n        '
        assert k > 0, ValueError('cannot have k below 1')
        assert k < n, ValueError('cannot have k=%d greater than %d' % (k, n))
        self.n = n
        self.k = k

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        n = self.n
        k = self.k
        j = int(np.ceil(n / k))
        for i in range(k):
            test_index = np.zeros(n, dtype=bool)
            if i < k - 1:
                test_index[i * j:(i + 1) * j] = True
            else:
                test_index[i * j:] = True
            train_index = np.logical_not(test_index)
            yield (train_index, test_index)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s.%s(n=%i, k=%i)' % (self.__class__.__module__, self.__class__.__name__, self.n, self.k)

class LeaveOneLabelOut:
    """
    Leave-One-Label_Out cross-validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, labels):
        if False:
            print('Hello World!')
        '\n        Leave-One-Label_Out cross validation:\n        Provides train/test indexes to split data in train test sets\n\n        Parameters\n        ----------\n        labels : list\n                List of labels\n\n        Examples\n        --------\n        >>> from scikits.learn import cross_val\n        >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]\n        >>> y = [1, 2, 1, 2]\n        >>> labels = [1, 1, 2, 2]\n        >>> lol = cross_val.LeaveOneLabelOut(labels)\n        >>> for train_index, test_index in lol:\n        ...    print "TRAIN:", train_index, "TEST:", test_index\n        ...    X_train, X_test, y_train, y_test = cross_val.split(train_index,             test_index, X, y)\n        ...    print X_train, X_test, y_train, y_test\n        TRAIN: [False False  True  True] TEST: [ True  True False False]\n        [[5 6]\n        [7 8]] [[1 2]\n        [3 4]] [1 2] [1 2]\n        TRAIN: [ True  True False False] TEST: [False False  True  True]\n        [[1 2]\n        [3 4]] [[5 6]\n        [7 8]] [1 2] [1 2]\n        '
        self.labels = labels

    def __iter__(self):
        if False:
            return 10
        labels = np.array(self.labels, copy=True)
        for i in np.unique(labels):
            test_index = np.zeros(len(labels), dtype=bool)
            test_index[labels == i] = True
            train_index = np.logical_not(test_index)
            yield (train_index, test_index)

    def __repr__(self):
        if False:
            return 10
        return '%s.%s(labels=%s)' % (self.__class__.__module__, self.__class__.__name__, self.labels)

def split(train_indexes, test_indexes, *args):
    if False:
        print('Hello World!')
    '\n    For each arg return a train and test subsets defined by indexes provided\n    in train_indexes and test_indexes\n    '
    ret = []
    for arg in args:
        arg = np.asanyarray(arg)
        arg_train = arg[train_indexes]
        arg_test = arg[test_indexes]
        ret.append(arg_train)
        ret.append(arg_test)
    return ret
'\n >>> cv = cross_val.LeaveOneLabelOut(X, y) # y making y optional and\npossible to add other arrays of the same shape[0] too\n >>> for X_train, y_train, X_test, y_test in cv:\n ...      print np.sqrt((model.fit(X_train, y_train).predict(X_test)\n- y_test) ** 2).mean())\n'

class KStepAhead:
    """
    KStepAhead cross validation iterator:
    Provides fit/test indexes to split data in sequential sets
    """

    def __init__(self, n, k=1, start=None, kall=True, return_slice=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        KStepAhead cross validation iterator:\n        Provides train/test indexes to split data in train test sets\n\n        Parameters\n        ----------\n        n: int\n            Total number of elements\n        k : int\n            number of steps ahead\n        start : int\n            initial size of data for fitting\n        kall : bool\n            if true. all values for up to k-step ahead are included in the test index.\n            If false, then only the k-th step ahead value is returnd\n\n\n        Notes\n        -----\n        I do not think this is really useful, because it can be done with\n        a very simple loop instead.\n        Useful as a plugin, but it could return slices instead for faster array access.\n\n        Examples\n        --------\n        >>> from scikits.learn import cross_val\n        >>> X = [[1, 2], [3, 4]]\n        >>> y = [1, 2]\n        >>> loo = cross_val.LeaveOneOut(2)\n        >>> for train_index, test_index in loo:\n        ...    print "TRAIN:", train_index, "TEST:", test_index\n        ...    X_train, X_test, y_train, y_test = cross_val.split(train_index, test_index, X, y)\n        ...    print X_train, X_test, y_train, y_test\n        TRAIN: [False  True] TEST: [ True False]\n        [[3 4]] [[1 2]] [2] [1]\n        TRAIN: [ True False] TEST: [False  True]\n        [[1 2]] [[3 4]] [1] [2]\n        '
        self.n = n
        self.k = k
        if start is None:
            start = int(np.trunc(n * 0.25))
        self.start = start
        self.kall = kall
        self.return_slice = return_slice

    def __iter__(self):
        if False:
            while True:
                i = 10
        n = self.n
        k = self.k
        start = self.start
        if self.return_slice:
            for i in range(start, n - k):
                train_slice = slice(None, i, None)
                if self.kall:
                    test_slice = slice(i, i + k)
                else:
                    test_slice = slice(i + k - 1, i + k)
                yield (train_slice, test_slice)
        else:
            for i in range(start, n - k):
                train_index = np.zeros(n, dtype=bool)
                train_index[:i] = True
                test_index = np.zeros(n, dtype=bool)
                if self.kall:
                    test_index[i:i + k] = True
                else:
                    test_index[i + k - 1:i + k] = True
                yield (train_index, test_index)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s.%s(n=%i)' % (self.__class__.__module__, self.__class__.__name__, self.n)