import six
import chainer
from chainer.dataset import dataset_mixin

class TabularDataset(dataset_mixin.DatasetMixin):
    """An abstract class that represents tabular dataset.

    This class represents a tabular dataset.
    In a tabular dataset, all examples have the same number of elements.
    For example, all examples of the dataset below have three elements
    (:obj:`a[i]`, :obj:`b[i]`, and :obj:`c[i]`).

    .. csv-table::
        :header: , a, b, c

        0, :obj:`a[0]`, :obj:`b[0]`, :obj:`c[0]`
        1, :obj:`a[1]`, :obj:`b[1]`, :obj:`c[1]`
        2, :obj:`a[2]`, :obj:`b[2]`, :obj:`c[2]`
        3, :obj:`a[3]`, :obj:`b[3]`, :obj:`c[3]`

    Since an example can be represented by both tuple and dict (
    :obj:`(a[i], b[i], c[i])` and :obj:`{'a': a[i], 'b': b[i], 'c': c[i]}`),
    this class uses :attr:`mode` to indicate which representation will be used.
    If there is only one column, an example also can be represented by a value
    (:obj:`a[i]`). In this case, :attr:`mode` is :obj:`None`.

    An inheritance should implement
    :meth:`__len__`, :attr:`keys`, :attr:`mode` and :meth:`get_examples`.

    >>> import numpy as np
    >>>
    >>> from chainer import dataset
    >>>
    >>> class MyDataset(dataset.TabularDataset):
    ...
    ...     def __len__(self):
    ...         return 4
    ...
    ...     @property
    ...     def keys(self):
    ...          return ('a', 'b', 'c')
    ...
    ...     @property
    ...     def mode(self):
    ...          return tuple
    ...
    ...     def get_examples(self, indices, key_indices):
    ...          data = np.arange(12).reshape((4, 3))
    ...          if indices is not None:
    ...              data = data[indices]
    ...          if key_indices is not None:
    ...              data = data[:, list(key_indices)]
    ...          return tuple(data.transpose())
    ...
    >>> dataset = MyDataset()
    >>> len(dataset)
    4
    >>> dataset.keys
    ('a', 'b', 'c')
    >>> dataset.astuple()[0]
    (0, 1, 2)
    >>> sorted(dataset.asdict()[0].items())
    [('a', 0), ('b', 1), ('c', 2)]
    >>>
    >>> view = dataset.slice[[3, 2], ('c', 0)]
    >>> len(view)
    2
    >>> view.keys
    ('c', 'a')
    >>> view.astuple()[1]
    (8, 6)
    >>> sorted(view.asdict()[1].items())
    [('a', 6), ('c', 8)]

    """

    def __len__(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @property
    def keys(self):
        if False:
            while True:
                i = 10
        'Names of columns.\n\n        A tuple of strings that indicate the names of columns.\n        '
        raise NotImplementedError

    @property
    def mode(self):
        if False:
            for i in range(10):
                print('nop')
        'Mode of representation.\n\n        This indicates the type of value returned\n        by :meth:`fetch` and :meth:`__getitem__`.\n        :class:`tuple`, :class:`dict`, and :obj:`None` are supported.\n        '
        raise NotImplementedError

    def get_examples(self, indices, key_indices):
        if False:
            print('Hello World!')
        'Return a part of data.\n\n        Args:\n            indices (list of ints or slice): Indices of requested rows.\n                If this argument is :obj:`None`, it indicates all rows.\n            key_indices (tuple of ints): Indices of requested columns.\n                If this argument is :obj:`None`, it indicates all columns.\n\n        Returns:\n            tuple of lists/arrays\n        '
        raise NotImplementedError

    @property
    def slice(self):
        if False:
            return 10
        'Get a slice of dataset.\n\n        Args:\n           indices (list/array of ints/bools or slice): Requested rows.\n           keys (tuple of ints/strs or int or str): Requested columns.\n\n        Returns:\n            A view of specified range.\n        '
        return chainer.dataset.tabular._slice._SliceHelper(self)

    def fetch(self):
        if False:
            i = 10
            return i + 15
        "Fetch data.\n\n        This method fetches all data of the dataset/view.\n        Note that this method returns a column-major data\n        (i.e. :obj:`([a[0], ..., a[3]], ..., [c[0], ... c[3]])`,\n        :obj:`{'a': [a[0], ..., a[3]], ..., 'c': [c[0], ..., c[3]]}`, or\n        :obj:`[a[0], ..., a[3]]`).\n\n        Returns:\n            If :attr:`mode` is :class:`tuple`,\n            this method returns a tuple of lists/arrays.\n            If :attr:`mode` is :class:`dict`,\n            this method returns a dict of lists/arrays.\n        "
        examples = self.get_examples(None, None)
        if self.mode is tuple:
            return examples
        elif self.mode is dict:
            return dict(six.moves.zip(self.keys, examples))
        elif self.mode is None:
            return examples[0]

    def convert(self, data):
        if False:
            i = 10
            return i + 15
        'Convert fetched data.\n\n        This method takes data fetched by :meth:`fetch` and\n        pre-process them before passing them to models.\n        The default behaviour is converting each column into an ndarray.\n        This behaviour can be overridden by :meth:`with_converter`.\n        If the dataset is constructed by :meth:`concat` or :meth:`join`,\n        the converter of the first dataset is used.\n\n        Args:\n            data (tuple or dict): Data from :meth:`fetch`.\n\n        Returns:\n            A tuple or dict.\n            Each value is an ndarray.\n        '
        if isinstance(data, tuple):
            return tuple((_as_array(d) for d in data))
        elif isinstance(data, dict):
            return {k: _as_array(v) for (k, v) in data.items()}
        else:
            return _as_array(data)

    def astuple(self):
        if False:
            return 10
        'Return a view with tuple mode.\n\n        Returns:\n            A view whose :attr:`mode` is :class:`tuple`.\n        '
        return chainer.dataset.tabular._asmode._Astuple(self)

    def asdict(self):
        if False:
            print('Hello World!')
        'Return a view with dict mode.\n\n        Returns:\n            A view whose :attr:`mode` is :class:`dict`.\n        '
        return chainer.dataset.tabular._asmode._Asdict(self)

    def concat(self, *datasets):
        if False:
            print('Hello World!')
        'Stack datasets along rows.\n\n        Args:\n            datasets (iterable of :class:`TabularDataset`):\n                Datasets to be concatenated.\n                All datasets must have the same :attr:`keys`.\n\n        Returns:\n            A concatenated dataset.\n        '
        return chainer.dataset.tabular._concat._Concat(self, *datasets)

    def join(self, *datasets):
        if False:
            print('Hello World!')
        'Stack datasets along columns.\n\n        Args:\n            datasets (iterable of :class:`TabularDataset`):\n                Datasets to be concatenated.\n                All datasets must have the same length\n\n        Returns:\n            A joined dataset.\n        '
        return chainer.dataset.tabular._join._Join(self, *datasets)

    def transform(self, keys, transform):
        if False:
            for i in range(10):
                print('nop')
        'Apply a transform to each example.\n\n        Args:\n            keys (tuple of strs): The keys of transformed examples.\n            transform (callable): A callable that takes an example\n                and returns transformed example. :attr:`mode` of\n                transformed dataset is determined by the transformed\n                examples.\n\n        Returns:\n            A transfromed dataset.\n        '
        return chainer.dataset.tabular._transform._Transform(self, keys, transform)

    def transform_batch(self, keys, transform_batch):
        if False:
            while True:
                i = 10
        'Apply a transform to examples.\n\n        Args:\n            keys (tuple of strs): The keys of transformed examples.\n            transform_batch (callable): A callable that takes examples\n                and returns transformed examples. :attr:`mode` of\n                transformed dataset is determined by the transformed\n                examples.\n\n        Returns:\n            A transfromed dataset.\n        '
        return chainer.dataset.tabular._transform._TransformBatch(self, keys, transform_batch)

    def with_converter(self, converter):
        if False:
            return 10
        'Override the behaviour of :meth:`convert`.\n\n        This method overrides :meth:`convert`.\n\n        Args:\n            converter (callable): A new converter.\n\n        Returns:\n            A dataset with the new converter.\n        '
        return chainer.dataset.tabular._with_converter._WithConverter(self, converter)

    def get_example(self, i):
        if False:
            print('Hello World!')
        example = self.get_examples([i], None)
        example = tuple((col[0] for col in example))
        if self.mode is tuple:
            return example
        elif self.mode is dict:
            return dict(six.moves.zip(self.keys, example))
        elif self.mode is None:
            return example[0]

    def __iter__(self):
        if False:
            while True:
                i = 10
        return (self.get_example(i) for i in six.moves.range(len(self)))

def _as_array(data):
    if False:
        return 10
    if isinstance(data, chainer.get_array_types()):
        return data
    else:
        device = chainer.backend.get_device_from_array(data[0])
        with chainer.using_device(device):
            return device.xp.asarray(data)