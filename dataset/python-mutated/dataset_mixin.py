import numpy
import six

class DatasetMixin(object):
    """Default implementation of dataset indexing.

    DatasetMixin provides the :meth:`__getitem__` operator. The default
    implementation uses :meth:`get_example` to extract each example, and
    combines the results into a list. This mixin makes it easy to implement a
    new dataset that does not support efficient slicing.

    Dataset implementation using DatasetMixin still has to provide the
    :meth:`__len__` operator explicitly.

    """

    def __getitem__(self, index):
        if False:
            return 10
        'Returns an example or a sequence of examples.\n\n        It implements the standard Python indexing and one-dimensional integer\n        array indexing. It uses the :meth:`get_example` method by default, but\n        it may be overridden by the implementation to, for example, improve the\n        slicing performance.\n\n        Args:\n            index (int, slice, list or numpy.ndarray): An index of an example\n                or indexes of examples.\n\n        Returns:\n            If index is int, returns an example created by `get_example`.\n            If index is either slice or one-dimensional list or numpy.ndarray,\n            returns a list of examples created by `get_example`.\n\n        .. admonition:: Example\n\n           >>> import numpy\n           >>> from chainer import dataset\n           >>> class SimpleDataset(dataset.DatasetMixin):\n           ...     def __init__(self, values):\n           ...         self.values = values\n           ...     def __len__(self):\n           ...         return len(self.values)\n           ...     def get_example(self, i):\n           ...         return self.values[i]\n           ...\n           >>> ds = SimpleDataset([0, 1, 2, 3, 4, 5])\n           >>> ds[1]   # Access by int\n           1\n           >>> ds[1:3]  # Access by slice\n           [1, 2]\n           >>> ds[[4, 0]]  # Access by one-dimensional integer list\n           [4, 0]\n           >>> index = numpy.arange(3)\n           >>> ds[index]  # Access by one-dimensional integer numpy.ndarray\n           [0, 1, 2]\n\n        '
        if isinstance(index, slice):
            (current, stop, step) = index.indices(len(self))
            return [self.get_example(i) for i in six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, numpy.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Returns the number of data points.'
        raise NotImplementedError

    def get_example(self, i):
        if False:
            return 10
        'Returns the i-th example.\n\n        Implementations should override it. It should raise :class:`IndexError`\n        if the index is invalid.\n\n        Args:\n            i (int): The index of the example.\n\n        Returns:\n            The i-th example.\n\n        '
        raise NotImplementedError