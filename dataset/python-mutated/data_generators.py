"""
Module defining an interface for data generators and providing concrete implementations for the supported frameworks.
Their purpose is to allow for data loading and batching on the fly, as well as dynamic data augmentation.
The generators can be used with the `fit_generator` function in the :class:`.Classifier` interface. Users can define
their own generators following the :class:`.DataGenerator` interface. For large, numpy array-based  datasets, the
:class:`.NumpyDataGenerator` class can be flexibly used with `fit_generator` on framework-specific classifiers.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import inspect
import logging
from typing import Any, Dict, Generator, Iterator, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    import keras
    import mxnet
    import tensorflow as tf
    import torch
logger = logging.getLogger(__name__)

class DataGenerator(abc.ABC):
    """
    Base class for data generators.
    """

    def __init__(self, size: Optional[int], batch_size: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Base initializer for data generators.\n\n        :param size: Total size of the dataset.\n        :param batch_size: Size of the minibatches.\n        '
        if size is not None and (not isinstance(size, int) or size < 1):
            raise ValueError('The total size of the dataset must be an integer greater than zero.')
        self._size = size
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('The batch size must be an integer greater than zero.')
        self._batch_size = batch_size
        if size is not None and batch_size > size:
            raise ValueError('The batch size must be smaller than the dataset size.')
        self._iterator: Optional[Any] = None

    @abc.abstractmethod
    def get_batch(self) -> tuple:
        if False:
            return 10
        '\n        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data\n        indefinitely.\n\n        :return: A tuple containing a batch of data `(x, y)`.\n        '
        raise NotImplementedError

    @property
    def iterator(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        :return: Return the framework's iterable data generator.\n        "
        return self._iterator

    @property
    def batch_size(self) -> int:
        if False:
            return 10
        '\n        :return: Return the batch size.\n        '
        return self._batch_size

    @property
    def size(self) -> Optional[int]:
        if False:
            while True:
                i = 10
        '\n        :return: Return the dataset size.\n        '
        return self._size

class NumpyDataGenerator(DataGenerator):
    """
    Simple numpy data generator backed by numpy arrays.

    Can be useful for applying numpy data to estimators in other frameworks
        e.g., when translating the entire numpy data to GPU tensors would cause OOM
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int=1, drop_remainder: bool=True, shuffle: bool=False):
        if False:
            print('Hello World!')
        '\n        Create a numpy data generator backed by numpy arrays\n\n        :param x: Numpy array of inputs\n        :param y: Numpy array of targets\n        :param batch_size: Size of the minibatches\n        :param drop_remainder: Whether to omit the last incomplete minibatch in an epoch\n        :param shuffle: Whether to shuffle the dataset for each epoch\n        '
        x = np.asanyarray(x)
        y = np.asanyarray(y)
        try:
            if len(x) != len(y):
                raise ValueError('inputs must be of equal length')
        except TypeError as err:
            raise ValueError(f'inputs x {x} and y {y} must be sized objects') from err
        size = len(x)
        self.x = x
        self.y = y
        super().__init__(size, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_remainder = bool(drop_remainder)
        batches_per_epoch = size / self.batch_size
        if not self.drop_remainder:
            batches_per_epoch = np.ceil(batches_per_epoch)
        self.batches_per_epoch = int(batches_per_epoch)
        self._iterator = self
        self.generator: Iterator[Any] = iter([])

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        if self.shuffle:
            index = np.arange(self.size)
            np.random.shuffle(index)
            for i in range(self.batches_per_epoch):
                batch_index = index[i * self.batch_size:(i + 1) * self.batch_size]
                yield (self.x[batch_index], self.y[batch_index])
        else:
            for i in range(self.batches_per_epoch):
                yield (self.x[i * self.batch_size:(i + 1) * self.batch_size], self.y[i * self.batch_size:(i + 1) * self.batch_size])

    def get_batch(self) -> tuple:
        if False:
            i = 10
            return i + 15
        '\n        Provide the next batch for training in the form of a tuple `(x, y)`.\n            The generator will loop over the data indefinitely.\n            If drop_remainder is True, then the last minibatch in each epoch may be a different size\n\n        :return: A tuple containing a batch of data `(x, y)`.\n        '
        try:
            return next(self.generator)
        except StopIteration:
            self.generator = iter(self)
            return next(self.generator)

class KerasDataGenerator(DataGenerator):
    """
    Wrapper class on top of the Keras-native data generators. These can either be generator functions,
    `keras.utils.Sequence` or Keras-specific data generators (`keras.preprocessing.image.ImageDataGenerator`).
    """

    def __init__(self, iterator: Union['keras.utils.Sequence', 'tf.keras.utils.Sequence', 'keras.preprocessing.image.ImageDataGenerator', 'tf.keras.preprocessing.image.ImageDataGenerator', Generator], size: Optional[int], batch_size: int) -> None:
        if False:
            print('Hello World!')
        '\n        Create a Keras data generator wrapper instance.\n\n        :param iterator: A generator as specified by Keras documentation. Its output must be a tuple of either\n                         `(inputs, targets)` or `(inputs, targets, sample_weights)`. All arrays in this tuple must have\n                         the same length. The generator is expected to loop over its data indefinitely.\n        :param size: Total size of the dataset.\n        :param batch_size: Size of the minibatches.\n        '
        super().__init__(size=size, batch_size=batch_size)
        self._iterator = iterator

    def get_batch(self) -> tuple:
        if False:
            i = 10
            return i + 15
        '\n        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data\n        indefinitely.\n\n        :return: A tuple containing a batch of data `(x, y)`.\n        '
        if inspect.isgeneratorfunction(self.iterator):
            return next(self.iterator)
        iter_ = iter(self.iterator)
        return next(iter_)

class PyTorchDataGenerator(DataGenerator):
    """
    Wrapper class on top of the PyTorch native data loader :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, iterator: 'torch.utils.data.DataLoader', size: int, batch_size: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a data generator wrapper on top of a PyTorch :class:`DataLoader`.\n\n        :param iterator: A PyTorch data generator.\n        :param size: Total size of the dataset.\n        :param batch_size: Size of the minibatches.\n        '
        from torch.utils.data import DataLoader
        super().__init__(size=size, batch_size=batch_size)
        if not isinstance(iterator, DataLoader):
            raise TypeError(f'Expected instance of PyTorch `DataLoader, received {type(iterator)} instead.`')
        self._iterator: DataLoader = iterator
        self._current = iter(self.iterator)

    def get_batch(self) -> tuple:
        if False:
            return 10
        '\n        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data\n        indefinitely.\n\n        :return: A tuple containing a batch of data `(x, y)`.\n        :rtype: `tuple`\n        '
        try:
            batch = list(next(self._current))
        except StopIteration:
            self._current = iter(self.iterator)
            batch = list(next(self._current))
        for (i, item) in enumerate(batch):
            batch[i] = item.data.cpu().numpy()
        return tuple(batch)

class MXDataGenerator(DataGenerator):
    """
    Wrapper class on top of the MXNet/Gluon native data loader :class:`mxnet.gluon.data.DataLoader`.
    """

    def __init__(self, iterator: 'mxnet.gluon.data.DataLoader', size: int, batch_size: int) -> None:
        if False:
            return 10
        '\n        Create a data generator wrapper on top of an MXNet :class:`DataLoader`.\n\n        :param iterator: A MXNet DataLoader instance.\n        :param size: Total size of the dataset.\n        :param batch_size: Size of the minibatches.\n        '
        import mxnet
        super().__init__(size=size, batch_size=batch_size)
        if not isinstance(iterator, mxnet.gluon.data.DataLoader):
            raise TypeError(f'Expected instance of Gluon `DataLoader, received {type(iterator)} instead.`')
        self._iterator = iterator
        self._current = iter(self.iterator)

    def get_batch(self) -> tuple:
        if False:
            i = 10
            return i + 15
        '\n        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data\n        indefinitely.\n\n        :return: A tuple containing a batch of data `(x, y)`.\n        '
        try:
            batch = list(next(self._current))
        except StopIteration:
            self._current = iter(self.iterator)
            batch = list(next(self._current))
        for (i, item) in enumerate(batch):
            batch[i] = item.asnumpy()
        return tuple(batch)

class TensorFlowDataGenerator(DataGenerator):
    """
    Wrapper class on top of the TensorFlow native iterators :class:`tf.data.Iterator`.
    """

    def __init__(self, sess: 'tf.Session', iterator: 'tf.data.Iterator', iterator_type: str, iterator_arg: Union[Dict, Tuple, 'tf.Operation'], size: int, batch_size: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a data generator wrapper for TensorFlow. Supported iterators: initializable, reinitializable, feedable.\n\n        :param sess: TensorFlow session.\n        :param iterator: Data iterator from TensorFlow.\n        :param iterator_type: Type of the iterator. Supported types: `initializable`, `reinitializable`, `feedable`.\n        :param iterator_arg: Argument to initialize the iterator. It is either a feed_dict used for the initializable\n        and feedable mode, or an init_op used for the reinitializable mode.\n        :param size: Total size of the dataset.\n        :param batch_size: Size of the minibatches.\n        :raises `TypeError`, `ValueError`: If input parameters are not valid.\n        '
        import tensorflow.compat.v1 as tf
        super().__init__(size=size, batch_size=batch_size)
        self.sess = sess
        self._iterator = iterator
        self.iterator_type = iterator_type
        self.iterator_arg = iterator_arg
        if not isinstance(iterator, tf.data.Iterator):
            raise TypeError('Only support object tf.data.Iterator')
        if iterator_type == 'initializable':
            if not isinstance(iterator_arg, dict):
                raise TypeError(f'Need to pass a dictionary for iterator type {iterator_type}')
        elif iterator_type == 'reinitializable':
            if not isinstance(iterator_arg, tf.Operation):
                raise TypeError(f'Need to pass a TensorFlow operation for iterator type {iterator_type}')
        elif iterator_type == 'feedable':
            if not isinstance(iterator_arg, tuple):
                raise TypeError(f'Need to pass a tuple for iterator type {iterator_type}')
        else:
            raise TypeError(f'Iterator type {iterator_type} not supported')

    def get_batch(self) -> tuple:
        if False:
            print('Hello World!')
        '\n        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data\n        indefinitely.\n\n        :return: A tuple containing a batch of data `(x, y)`.\n        :raises `ValueError`: If the iterator has reached the end.\n        '
        import tensorflow as tf
        next_batch = self.iterator.get_next()
        try:
            if self.iterator_type in ('initializable', 'reinitializable'):
                return self.sess.run(next_batch)
            return self.sess.run(next_batch, feed_dict=self.iterator_arg[1])
        except (tf.errors.FailedPreconditionError, tf.errors.OutOfRangeError):
            if self.iterator_type == 'initializable':
                self.sess.run(self.iterator.initializer, feed_dict=self.iterator_arg)
                return self.sess.run(next_batch)
            if self.iterator_type == 'reinitializable':
                self.sess.run(self.iterator_arg)
                return self.sess.run(next_batch)
            self.sess.run(self.iterator_arg[0].initializer)
            return self.sess.run(next_batch, feed_dict=self.iterator_arg[1])

class TensorFlowV2DataGenerator(DataGenerator):
    """
    Wrapper class on top of the TensorFlow v2 native iterators :class:`tf.data.Iterator`.
    """

    def __init__(self, iterator: 'tf.data.Dataset', size: int, batch_size: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a data generator wrapper for TensorFlow. Supported iterators: initializable, reinitializable, feedable.\n\n        :param iterator: TensorFlow Dataset.\n        :param size: Total size of the dataset.\n        :param batch_size: Size of the minibatches.\n        :raises `TypeError`, `ValueError`: If input parameters are not valid.\n        '
        import tensorflow as tf
        super().__init__(size=size, batch_size=batch_size)
        self._iterator = iterator
        self._iterator_iter = iter(iterator)
        if not isinstance(iterator, tf.data.Dataset):
            raise TypeError('Only support object tf.data.Dataset')

    def get_batch(self) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        '\n        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data\n        indefinitely.\n\n        :return: A tuple containing a batch of data `(x, y)`.\n        :raises `ValueError`: If the iterator has reached the end.\n        '
        (x, y) = next(self._iterator_iter)
        return (x.numpy(), y.numpy())