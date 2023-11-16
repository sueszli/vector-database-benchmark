"""Numpy pickle compatibility functions."""
import pickle
import os
import zlib
import inspect
from io import BytesIO
from .numpy_pickle_utils import _ZFILE_PREFIX
from .numpy_pickle_utils import Unpickler
from .numpy_pickle_utils import _ensure_native_byte_order

def hex_str(an_int):
    if False:
        return 10
    'Convert an int to an hexadecimal string.'
    return '{:#x}'.format(an_int)

def asbytes(s):
    if False:
        return 10
    if isinstance(s, bytes):
        return s
    return s.encode('latin1')
_MAX_LEN = len(hex_str(2 ** 64))
_CHUNK_SIZE = 64 * 1024

def read_zfile(file_handle):
    if False:
        i = 10
        return i + 15
    'Read the z-file and return the content as a string.\n\n    Z-files are raw data compressed with zlib used internally by joblib\n    for persistence. Backward compatibility is not guaranteed. Do not\n    use for external purposes.\n    '
    file_handle.seek(0)
    header_length = len(_ZFILE_PREFIX) + _MAX_LEN
    length = file_handle.read(header_length)
    length = length[len(_ZFILE_PREFIX):]
    length = int(length, 16)
    next_byte = file_handle.read(1)
    if next_byte != b' ':
        file_handle.seek(header_length)
    data = zlib.decompress(file_handle.read(), 15, length)
    assert len(data) == length, 'Incorrect data length while decompressing %s.The file could be corrupted.' % file_handle
    return data

def write_zfile(file_handle, data, compress=1):
    if False:
        while True:
            i = 10
    'Write the data in the given file as a Z-file.\n\n    Z-files are raw data compressed with zlib used internally by joblib\n    for persistence. Backward compatibility is not guaranteed. Do not\n    use for external purposes.\n    '
    file_handle.write(_ZFILE_PREFIX)
    length = hex_str(len(data))
    file_handle.write(asbytes(length.ljust(_MAX_LEN)))
    file_handle.write(zlib.compress(asbytes(data), compress))

class NDArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    The only thing this object does, is to carry the filename in which
    the array has been persisted, and the array subclass.
    """

    def __init__(self, filename, subclass, allow_mmap=True):
        if False:
            print('Hello World!')
        'Constructor. Store the useful information for later.'
        self.filename = filename
        self.subclass = subclass
        self.allow_mmap = allow_mmap

    def read(self, unpickler):
        if False:
            print('Hello World!')
        'Reconstruct the array.'
        filename = os.path.join(unpickler._dirname, self.filename)
        allow_mmap = getattr(self, 'allow_mmap', True)
        kwargs = {}
        if allow_mmap:
            kwargs['mmap_mode'] = unpickler.mmap_mode
        if 'allow_pickle' in inspect.signature(unpickler.np.load).parameters:
            kwargs['allow_pickle'] = True
        array = unpickler.np.load(filename, **kwargs)
        array = _ensure_native_byte_order(array)
        if hasattr(array, '__array_prepare__') and self.subclass not in (unpickler.np.ndarray, unpickler.np.memmap):
            new_array = unpickler.np.core.multiarray._reconstruct(self.subclass, (0,), 'b')
            return new_array.__array_prepare__(array)
        else:
            return array

class ZNDArrayWrapper(NDArrayWrapper):
    """An object to be persisted instead of numpy arrays.

    This object store the Zfile filename in which
    the data array has been persisted, and the meta information to
    retrieve it.
    The reason that we store the raw buffer data of the array and
    the meta information, rather than array representation routine
    (tobytes) is that it enables us to use completely the strided
    model to avoid memory copies (a and a.T store as fast). In
    addition saving the heavy information separately can avoid
    creating large temporary buffers when unpickling data with
    large arrays.
    """

    def __init__(self, filename, init_args, state):
        if False:
            for i in range(10):
                print('nop')
        'Constructor. Store the useful information for later.'
        self.filename = filename
        self.state = state
        self.init_args = init_args

    def read(self, unpickler):
        if False:
            while True:
                i = 10
        'Reconstruct the array from the meta-information and the z-file.'
        filename = os.path.join(unpickler._dirname, self.filename)
        array = unpickler.np.core.multiarray._reconstruct(*self.init_args)
        with open(filename, 'rb') as f:
            data = read_zfile(f)
        state = self.state + (data,)
        array.__setstate__(state)
        return array

class ZipNumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles."""
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.'
        self._filename = os.path.basename(filename)
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = self._open_pickle(file_handle)
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _open_pickle(self, file_handle):
        if False:
            print('Hello World!')
        return BytesIO(read_zfile(file_handle))

    def load_build(self):
        if False:
            while True:
                i = 10
        'Set the state of a newly created object.\n\n        We capture it to replace our place-holder objects,\n        NDArrayWrapper, by the array we are interested in. We\n        replace them directly in the stack of pickler.\n        '
        Unpickler.load_build(self)
        if isinstance(self.stack[-1], NDArrayWrapper):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, but numpy didn't import correctly")
            nd_array_wrapper = self.stack.pop()
            array = nd_array_wrapper.read(self)
            self.stack.append(array)
    dispatch[pickle.BUILD[0]] = load_build

def load_compatibility(filename):
    if False:
        print('Hello World!')
    'Reconstruct a Python object from a file persisted with joblib.dump.\n\n    This function ensures the compatibility with joblib old persistence format\n    (<= 0.9.3).\n\n    Parameters\n    ----------\n    filename: string\n        The name of the file from which to load the object\n\n    Returns\n    -------\n    result: any Python object\n        The object stored in the file.\n\n    See Also\n    --------\n    joblib.dump : function to save an object\n\n    Notes\n    -----\n\n    This function can load numpy array files saved separately during the\n    dump.\n    '
    with open(filename, 'rb') as file_handle:
        unpickler = ZipNumpyUnpickler(filename, file_handle=file_handle)
        try:
            obj = unpickler.load()
        except UnicodeDecodeError as exc:
            new_exc = ValueError('You may be trying to read with python 3 a joblib pickle generated with python 2. This feature is not supported by joblib.')
            new_exc.__cause__ = exc
            raise new_exc
        finally:
            if hasattr(unpickler, 'file_handle'):
                unpickler.file_handle.close()
        return obj