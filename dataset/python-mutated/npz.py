import warnings
import numpy
import cupy
_support_allow_pickle = numpy.lib.NumpyVersion(numpy.__version__) >= '1.10.0'

class NpzFile(object):

    def __init__(self, npz_file):
        if False:
            return 10
        self.npz_file = npz_file

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.npz_file.__enter__()
        return self

    def __exit__(self, typ, val, traceback):
        if False:
            print('Hello World!')
        self.npz_file.__exit__(typ, val, traceback)

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        arr = self.npz_file[key]
        return cupy.array(arr)

    def close(self):
        if False:
            i = 10
            return i + 15
        self.npz_file.close()

def load(file, mmap_mode=None, allow_pickle=None):
    if False:
        i = 10
        return i + 15
    "Loads arrays or pickled objects from ``.npy``, ``.npz`` or pickled file.\n\n    This function just calls ``numpy.load`` and then sends the arrays to the\n    current device. NPZ file is converted to NpzFile object, which defers the\n    transfer to the time of accessing the items.\n\n    Args:\n        file (file-like object or string): The file to read.\n        mmap_mode (None, 'r+', 'r', 'w+', 'c'): If not ``None``, memory-map the\n            file to construct an intermediate :class:`numpy.ndarray` object and\n            transfer it to the current device.\n        allow_pickle (bool): Allow loading pickled object arrays stored in npy\n            files. Reasons for disallowing pickles include security, as\n            loading pickled data can execute arbitrary code. If pickles are\n            disallowed, loading object arrays will fail.\n            Please be aware that CuPy does not support arrays with dtype of\n            `object`.\n            The default is False.\n            This option is available only for NumPy 1.10 or later.\n            In NumPy 1.9, this option cannot be specified (loading pickled\n            objects is always allowed).\n\n    Returns:\n        CuPy array or NpzFile object depending on the type of the file. NpzFile\n        object is a dictionary-like object with the context manager protocol\n        (which enables us to use *with* statement on it).\n\n    .. seealso:: :func:`numpy.load`\n\n    "
    if _support_allow_pickle:
        allow_pickle = False if allow_pickle is None else allow_pickle
        obj = numpy.load(file, mmap_mode, allow_pickle)
    else:
        if allow_pickle is not None:
            warnings.warn('allow_pickle option is not supported in NumPy 1.9')
        obj = numpy.load(file, mmap_mode)
    if isinstance(obj, numpy.ndarray):
        return cupy.array(obj)
    elif isinstance(obj, numpy.lib.npyio.NpzFile):
        return NpzFile(obj)
    else:
        return obj

def save(file, arr, allow_pickle=None):
    if False:
        while True:
            i = 10
    'Saves an array to a binary file in ``.npy`` format.\n\n    Args:\n        file (file or str): File or filename to save.\n        arr (array_like): Array to save. It should be able to feed to\n            :func:`cupy.asnumpy`.\n        allow_pickle (bool): Allow saving object arrays using Python pickles.\n            Reasons for disallowing pickles include security (loading pickled\n            data can execute arbitrary code) and portability (pickled objects\n            may not be loadable on different Python installations, for example\n            if the stored objects require libraries that are not available,\n            and not all pickled data is compatible between Python 2 and Python\n            3).\n            The default is True.\n            This option is available only for NumPy 1.10 or later.\n            In NumPy 1.9, this option cannot be specified (saving objects\n            using pickles is always allowed).\n\n    .. seealso:: :func:`numpy.save`\n\n    '
    if _support_allow_pickle:
        allow_pickle = True if allow_pickle is None else allow_pickle
        numpy.save(file, cupy.asnumpy(arr), allow_pickle)
    else:
        if allow_pickle is not None:
            warnings.warn('allow_pickle option is not supported in NumPy 1.9')
        numpy.save(file, cupy.asnumpy(arr))

def savez(file, *args, **kwds):
    if False:
        print('Hello World!')
    'Saves one or more arrays into a file in uncompressed ``.npz`` format.\n\n    Arguments without keys are treated as arguments with automatic keys named\n    ``arr_0``, ``arr_1``, etc. corresponding to the positions in the argument\n    list. The keys of arguments are used as keys in the ``.npz`` file, which\n    are used for accessing NpzFile object when the file is read by\n    :func:`cupy.load` function.\n\n    Args:\n        file (file or str): File or filename to save.\n        *args: Arrays with implicit keys.\n        **kwds: Arrays with explicit keys.\n\n    .. seealso:: :func:`numpy.savez`\n\n    '
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez(file, *args, **kwds)

def savez_compressed(file, *args, **kwds):
    if False:
        i = 10
        return i + 15
    'Saves one or more arrays into a file in compressed ``.npz`` format.\n\n    It is equivalent to :func:`cupy.savez` function except the output file is\n    compressed.\n\n    .. seealso::\n       :func:`cupy.savez` for more detail,\n       :func:`numpy.savez_compressed`\n\n    '
    args = map(cupy.asnumpy, args)
    for key in kwds:
        kwds[key] = cupy.asnumpy(kwds[key])
    numpy.savez_compressed(file, *args, **kwds)