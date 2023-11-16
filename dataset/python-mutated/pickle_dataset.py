import io
import multiprocessing.util
import threading
import six
import six.moves.cPickle as pickle
from chainer.dataset import dataset_mixin

class PickleDatasetWriter(object):
    """Writer class that makes PickleDataset.

    To make :class:`PickleDataset`, a user needs to prepare data using
    :class:`PickleDatasetWriter`.

    Args:
        writer: File like object that supports ``write`` and ``tell`` methods.
        protocol (int): Valid protocol for :mod:`pickle`.

    .. seealso: chainer.datasets.PickleDataset

    """

    def __init__(self, writer, protocol=pickle.HIGHEST_PROTOCOL):
        if False:
            i = 10
            return i + 15
        self._positions = []
        self._writer = writer
        self._protocol = protocol

    def close(self):
        if False:
            print('Hello World!')
        self._writer.close()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        self.close()

    def write(self, x):
        if False:
            while True:
                i = 10
        position = self._writer.tell()
        pickle.dump(x, self._writer, protocol=self._protocol)
        self._positions.append(position)

    def flush(self):
        if False:
            while True:
                i = 10
        if hasattr(self._writer, 'flush'):
            self._writer.flush()

class PickleDataset(dataset_mixin.DatasetMixin):
    """Dataset stored in a storage using pickle.

    :mod:`pickle` is the default serialization library of Python.
    This dataset stores any objects in a storage using :mod:`pickle`.
    Even when a user wants to use a large dataset, this dataset can stores all
    data in a large storage like HDD and each data can be randomly accessible.

    .. testsetup::

        import tempfile
        fs, path_to_data = tempfile.mkstemp()

    >>> with chainer.datasets.open_pickle_dataset_writer(path_to_data) as w:
    ...     w.write((1, 2.0, 'hello'))
    ...     w.write((2, 3.0, 'good-bye'))
    ...
    >>> with chainer.datasets.open_pickle_dataset(path_to_data) as dataset:
    ...     print(dataset[1])
    ...
    (2, 3.0, 'good-bye')

    .. testcleanup::

        import os
        os.close(fs)

    Args:
        reader: File like object. `reader` must support random access.

    """

    def __init__(self, reader):
        if False:
            print('Hello World!')
        if six.PY3 and (not reader.seekable()):
            raise ValueError('reader must support random access')
        self._reader = reader
        self._positions = []
        reader.seek(0)
        while True:
            position = reader.tell()
            try:
                pickle.load(reader)
            except EOFError:
                break
            self._positions.append(position)
        self._lock = threading.RLock()
        self._register_hook()

    def _register_hook(self):
        if False:
            print('Hello World!')
        multiprocessing.util.register_after_fork(self, PickleDataset._after_fork)

    def _after_fork(self):
        if False:
            i = 10
            return i + 15
        if callable(getattr(self._reader, 'after_fork', None)):
            self._reader.after_fork()

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        if False:
            return 10
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._register_hook()

    def close(self):
        if False:
            print('Hello World!')
        'Closes a file reader.\n\n        After a user calls this method, the dataset will no longer be\n        accessible..\n        '
        with self._lock:
            self._reader.close()

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        self.close()

    def __len__(self):
        if False:
            return 10
        return len(self._positions)

    def get_example(self, index):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self._reader.seek(self._positions[index])
            return pickle.load(self._reader)

class _FileReader(io.RawIOBase):
    """A file-like class implemented `after_fork()` hook

    The method :meth:`after_fork` is called in the child process after forking,
    and it closes and reopens the file object to avoid race condition caused by
    open file description.
    See: https://www.securecoding.cert.org/confluence/x/ZQG7AQ
    """

    def __init__(self, path):
        if False:
            return 10
        super(_FileReader, self).__init__()
        self._path = path
        self._fp = None
        self._open()

    def _open(self):
        if False:
            while True:
                i = 10
        self._fp = open(self._path, 'rb')

    def after_fork(self):
        if False:
            return 10
        'Reopens the file to avoid race condition.'
        self.close()
        self._open()

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        state = self.__dict__.copy()
        del state['_fp']
        return state

    def __setstate__(self, state):
        if False:
            return 10
        self.__dict__.update(state)
        self._open()

    def flush(self):
        if False:
            return 10
        self._fp.flush()

    def close(self):
        if False:
            return 10
        self._fp.close()

    def fileno(self):
        if False:
            while True:
                i = 10
        return self._fp.fileno()

    def seekable(self):
        if False:
            while True:
                i = 10
        return self._fp.seekable()

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            return 10
        return self._fp.seek(offset, whence)

    def tell(self):
        if False:
            return 10
        return self._fp.tell()

    def readinto(self, b):
        if False:
            for i in range(10):
                print('nop')
        return self._fp.readinto(b)

def open_pickle_dataset(path):
    if False:
        for i in range(10):
            print('nop')
    "Opens a dataset stored in a given path.\n\n    This is a helper function to open :class:`PickleDataset`. It opens a given\n    file in binary mode, and creates a :class:`PickleDataset` instance.\n\n    This method does not close the opened file. A user needs to call\n    :func:`PickleDataset.close` or use `with`:\n\n    .. code-block:: python\n\n        with chainer.datasets.open_pickle_dataset('path') as dataset:\n            pass  # use dataset\n\n    Args:\n        path (str): Path to a dataset.\n\n    Returns:\n        chainer.datasets.PickleDataset: Opened dataset.\n\n    .. seealso: chainer.datasets.PickleDataset\n\n    "
    reader = _FileReader(path)
    return PickleDataset(reader)

def open_pickle_dataset_writer(path, protocol=pickle.HIGHEST_PROTOCOL):
    if False:
        while True:
            i = 10
    "Opens a writer to make a PickleDataset.\n\n    This is a helper function to open :class:`PickleDatasetWriter`. It opens a\n    given file in binary mode and creates a :class:`PickleDatasetWriter`\n    instance.\n\n    This method does not close the opened file. A user needs to call\n    :func:`PickleDatasetWriter.close` or use `with`:\n\n    .. code-block:: python\n\n        with chainer.datasets.open_pickle_dataset_writer('path') as writer:\n            pass  # use writer\n\n    Args:\n        path (str): Path to a dataset.\n        protocol (int): Valid protocol for :mod:`pickle`.\n\n    Returns:\n        chainer.datasets.PickleDatasetWriter: Opened writer.\n\n    .. seealso: chainer.datasets.PickleDataset\n\n    "
    writer = open(path, 'wb')
    return PickleDatasetWriter(writer, protocol=protocol)