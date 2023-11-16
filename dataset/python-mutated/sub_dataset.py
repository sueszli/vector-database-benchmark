import numpy
import six
import warnings
from chainer.dataset import dataset_mixin

class SubDataset(dataset_mixin.DatasetMixin):
    """Subset of a base dataset.

    SubDataset defines a subset of a given base dataset. The subset is defined
    as an interval of indexes, optionally with a given permutation.

    If ``order`` is given, then the ``i``-th example of this dataset is the
    ``order[start + i]``-th example of the base dataset, where ``i`` is a
    non-negative integer. If ``order`` is not given, then the ``i``-th example
    of this dataset is the ``start + i``-th example of the base dataset.
    Negative indexing is also allowed: in this case, the term ``start + i`` is
    replaced by ``finish + i``.

    SubDataset is often used to split a dataset into training and validation
    subsets. The training set is used for training, while the validation set is
    used to track the generalization performance, i.e. how the learned model
    works well on unseen data. We can tune hyperparameters (e.g. number of
    hidden units, weight initializers, learning rate, etc.) by comparing the
    validation performance. Note that we often use another set called test set
    to measure the quality of the tuned hyperparameter, which can be made by
    nesting multiple SubDatasets.

    There are two ways to make training-validation splits. One is a single
    split, where the dataset is split just into two subsets. It can be done by
    :func:`split_dataset` or :func:`split_dataset_random`. The other one is a
    :math:`k`-fold cross validation, in which the dataset is divided into
    :math:`k` subsets, and :math:`k` different splits are generated using each
    of the :math:`k` subsets as a validation set and the rest as a training
    set. It can be done by :func:`get_cross_validation_datasets`.

    Args:
        dataset: Base dataset.
        start (int): The first index in the interval.
        finish (int): The next-to-the-last index in the interval.
        order (sequence of ints): Permutation of indexes in the base dataset.
            If this is ``None``, then the ascending order of indexes is used.

    """

    def __init__(self, dataset, start, finish, order=None):
        if False:
            print('Hello World!')
        if start < 0 or finish > len(dataset):
            raise ValueError('subset overruns the base dataset.')
        self._dataset = dataset
        self._start = start
        self._finish = finish
        self._size = finish - start
        if order is not None and len(order) != len(dataset):
            msg = 'order option must have the same length as the base dataset: len(order) = {} while len(dataset) = {}'.format(len(order), len(dataset))
            raise ValueError(msg)
        self._order = order

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return self._size

    def get_example(self, i):
        if False:
            for i in range(10):
                print('nop')
        if i >= 0:
            if i >= self._size:
                raise IndexError('dataset index out of range')
            index = self._start + i
        else:
            if i < -self._size:
                raise IndexError('dataset index out of range')
            index = self._finish + i
        if self._order is not None:
            index = self._order[index]
        return self._dataset[index]

def split_dataset(dataset, split_at, order=None):
    if False:
        return 10
    'Splits a dataset into two subsets.\n\n    This function creates two instances of :class:`SubDataset`. These instances\n    do not share any examples, and they together cover all examples of the\n    original dataset.\n\n    Args:\n        dataset: Dataset to split.\n        split_at (int): Position at which the base dataset is split.\n        order (sequence of ints): Permutation of indexes in the base dataset.\n            See the documentation of :class:`SubDataset` for details.\n\n    Returns:\n        tuple: Two :class:`SubDataset` objects. The first subset represents the\n        examples of indexes ``order[:split_at]`` while the second subset\n        represents the examples of indexes ``order[split_at:]``.\n\n    '
    n_examples = len(dataset)
    if not isinstance(split_at, (six.integer_types, numpy.integer)):
        raise TypeError('split_at must be int, got {} instead'.format(type(split_at)))
    if split_at < 0:
        raise ValueError('split_at must be non-negative')
    if split_at > n_examples:
        raise ValueError('split_at exceeds the dataset size')
    subset1 = SubDataset(dataset, 0, split_at, order)
    subset2 = SubDataset(dataset, split_at, n_examples, order)
    return (subset1, subset2)

def split_dataset_random(dataset, first_size, seed=None):
    if False:
        i = 10
        return i + 15
    'Splits a dataset into two subsets randomly.\n\n    This function creates two instances of :class:`SubDataset`. These instances\n    do not share any examples, and they together cover all examples of the\n    original dataset. The split is automatically done randomly.\n\n    Args:\n        dataset: Dataset to split.\n        first_size (int): Size of the first subset.\n        seed (int): Seed the generator used for the permutation of indexes.\n            If an integer being convertible to 32 bit unsigned integers is\n            specified, it is guaranteed that each sample\n            in the given dataset always belongs to a specific subset.\n            If ``None``, the permutation is changed randomly.\n\n    Returns:\n        tuple: Two :class:`SubDataset` objects. The first subset contains\n        ``first_size`` examples randomly chosen from the dataset without\n        replacement, and the second subset contains the rest of the\n        dataset.\n\n    '
    order = numpy.random.RandomState(seed).permutation(len(dataset))
    return split_dataset(dataset, first_size, order)

def split_dataset_n(dataset, n, order=None):
    if False:
        return 10
    'Splits a dataset into ``n`` subsets.\n\n    Args:\n        dataset: Dataset to split.\n        n(int): The number of subsets.\n        order (sequence of ints): Permutation of indexes in the base dataset.\n            See the documentation of :class:`SubDataset` for details.\n\n    Returns:\n        list: List of ``n`` :class:`SubDataset` objects.\n        Each subset contains the examples of indexes\n        ``order[i * (len(dataset) // n):(i + 1) * (len(dataset) // n)]``\n        .\n\n    '
    n_examples = len(dataset)
    sub_size = n_examples // n
    return [SubDataset(dataset, sub_size * i, sub_size * (i + 1), order) for i in six.moves.range(n)]

def split_dataset_n_random(dataset, n, seed=None):
    if False:
        print('Hello World!')
    'Splits a dataset into ``n`` subsets randomly.\n\n    Args:\n        dataset: Dataset to split.\n        n(int): The number of subsets.\n        seed (int): Seed the generator used for the permutation of indexes.\n            If an integer being convertible to 32 bit unsigned integers is\n            specified, it is guaranteed that each sample\n            in the given dataset always belongs to a specific subset.\n            If ``None``, the permutation is changed randomly.\n\n    Returns:\n        list: List of ``n`` :class:`SubDataset` objects.\n            Each subset contains ``len(dataset) // n`` examples randomly chosen\n            from the dataset without replacement.\n\n    '
    n_examples = len(dataset)
    sub_size = n_examples // n
    order = numpy.random.RandomState(seed).permutation(len(dataset))
    return [SubDataset(dataset, sub_size * i, sub_size * (i + 1), order) for i in six.moves.range(n)]

def get_cross_validation_datasets(dataset, n_folds=None, order=None, **kwargs):
    if False:
        while True:
            i = 10
    'Creates a set of training/test splits for cross validation.\n\n    This function generates ``n_folds`` splits of the given dataset. The first\n    part of each split corresponds to the training dataset, while the second\n    part to the test dataset. No pairs of test datasets share any examples, and\n    all test datasets together cover the whole base dataset. Each test dataset\n    contains almost same number of examples (the numbers may differ up to 1).\n\n    Args:\n        dataset: Dataset to split.\n        n_fold(int): *(deprecated)*\n            `n_fold` is now deprecated for consistency of naming choice.\n            Please use `n_folds` instead.\n        n_folds (int): Number of splits for cross validation.\n        order (sequence of ints): Order of indexes with which each split is\n            determined. If it is ``None``, then no permutation is used.\n\n    Returns:\n        list of tuples: List of dataset splits.\n\n    '
    if 'n_fold' in kwargs:
        warnings.warn('Argument `n_fold` is deprecated. Please use `n_folds` instead', DeprecationWarning)
        n_folds = kwargs['n_fold']
    if order is None:
        order = numpy.arange(len(dataset))
    else:
        order = numpy.array(order)
    whole_size = len(dataset)
    borders = [whole_size * i // n_folds for i in six.moves.range(n_folds + 1)]
    test_sizes = [borders[i + 1] - borders[i] for i in six.moves.range(n_folds)]
    splits = []
    for test_size in reversed(test_sizes):
        size = whole_size - test_size
        splits.append(split_dataset(dataset, size, order))
        new_order = numpy.empty_like(order)
        new_order[:test_size] = order[-test_size:]
        new_order[test_size:] = order[:-test_size]
        order = new_order
    return splits

def get_cross_validation_datasets_random(dataset, n_folds, seed=None, **kwargs):
    if False:
        return 10
    'Creates a set of training/test splits for cross validation randomly.\n\n    This function acts almost same as :func:`get_cross_validation_dataset`,\n    except automatically generating random permutation.\n\n    Args:\n        dataset: Dataset to split.\n        n_fold (int): *(deprecated)*\n            `n_fold` is now deprecated for consistency of naming choice.\n            Please use `n_folds` instead.\n        n_folds (int): Number of splits for cross validation.\n        seed (int): Seed the generator used for the permutation of indexes.\n            If an integer beging convertible to 32 bit unsigned integers is\n            specified, it is guaranteed that each sample\n            in the given dataset always belongs to a specific subset.\n            If ``None``, the permutation is changed randomly.\n\n    Returns:\n        list of tuples: List of dataset splits.\n\n    '
    if 'n_fold' in kwargs:
        warnings.warn('Argument `n_fold` is deprecated. Please use `n_folds` instead', DeprecationWarning)
        n_folds = kwargs['n_fold']
    order = numpy.random.RandomState(seed).permutation(len(dataset))
    return get_cross_validation_datasets(dataset, n_folds, order)