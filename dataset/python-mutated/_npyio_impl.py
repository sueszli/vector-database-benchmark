import os
import re
import functools
import itertools
import warnings
import weakref
import contextlib
import operator
from operator import itemgetter, index as opindex, methodcaller
from collections.abc import Mapping
import pickle
import numpy as np
from . import format
from ._datasource import DataSource
from numpy._core import overrides
from numpy._core.multiarray import packbits, unpackbits
from numpy._core._multiarray_umath import _load_from_filelike
from numpy._core.overrides import set_array_function_like_doc, set_module
from ._iotools import LineSplitter, NameValidator, StringConverter, ConverterError, ConverterLockError, ConversionWarning, _is_string_like, has_nested_fields, flatten_dtype, easy_dtype, _decode_line
from numpy._utils import asunicode, asbytes
__all__ = ['savetxt', 'loadtxt', 'genfromtxt', 'load', 'save', 'savez', 'savez_compressed', 'packbits', 'unpackbits', 'fromregex']
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')

class BagObj:
    """
    BagObj(obj)

    Convert attribute look-ups to getitems on the object passed in.

    Parameters
    ----------
    obj : class instance
        Object on which attribute look-up is performed.

    Examples
    --------
    >>> from numpy.lib._npyio_impl import BagObj as BO
    >>> class BagDemo:
    ...     def __getitem__(self, key): # An instance of BagObj(BagDemo)
    ...                                 # will call this method when any
    ...                                 # attribute look-up is required
    ...         result = "Doesn't matter what you want, "
    ...         return result + "you're gonna get this"
    ...
    >>> demo_obj = BagDemo()
    >>> bagobj = BO(demo_obj)
    >>> bagobj.hello_there
    "Doesn't matter what you want, you're gonna get this"
    >>> bagobj.I_can_be_anything
    "Doesn't matter what you want, you're gonna get this"

    """

    def __init__(self, obj):
        if False:
            while True:
                i = 10
        self._obj = weakref.proxy(obj)

    def __getattribute__(self, key):
        if False:
            return 10
        try:
            return object.__getattribute__(self, '_obj')[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self):
        if False:
            return 10
        '\n        Enables dir(bagobj) to list the files in an NpzFile.\n\n        This also enables tab-completion in an interpreter or IPython.\n        '
        return list(object.__getattribute__(self, '_obj').keys())

def zipfile_factory(file, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Create a ZipFile.\n\n    Allows for Zip64, and the `file` argument can accept file, str, or\n    pathlib.Path objects. `args` and `kwargs` are passed to the zipfile.ZipFile\n    constructor.\n    '
    if not hasattr(file, 'read'):
        file = os.fspath(file)
    import zipfile
    kwargs['allowZip64'] = True
    return zipfile.ZipFile(file, *args, **kwargs)

@set_module('numpy.lib.npyio')
class NpzFile(Mapping):
    """
    NpzFile(fid)

    A dictionary-like object with lazy-loading of files in the zipped
    archive provided on construction.

    `NpzFile` is used to load files in the NumPy ``.npz`` data archive
    format. It assumes that files in the archive have a ``.npy`` extension,
    other files are ignored.

    The arrays and file strings are lazily loaded on either
    getitem access using ``obj['key']`` or attribute lookup using
    ``obj.f.key``. A list of all files (without ``.npy`` extensions) can
    be obtained with ``obj.files`` and the ZipFile object itself using
    ``obj.zip``.

    Attributes
    ----------
    files : list of str
        List of all files in the archive with a ``.npy`` extension.
    zip : ZipFile instance
        The ZipFile object initialized with the zipped archive.
    f : BagObj instance
        An object on which attribute can be performed as an alternative
        to getitem access on the `NpzFile` instance itself.
    allow_pickle : bool, optional
        Allow loading pickled data. Default: False

        .. versionchanged:: 1.16.3
            Made default False in response to CVE-2019-6446.

    pickle_kwargs : dict, optional
        Additional keyword arguments to pass on to pickle.load.
        These are only useful when loading object arrays saved on
        Python 2 when using Python 3.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.
        This option is ignored when `allow_pickle` is passed.  In that case
        the file is by definition trusted and the limit is unnecessary.

    Parameters
    ----------
    fid : file, str, or pathlib.Path
        The zipped archive to open. This is either a file-like object
        or a string containing the path to the archive.
    own_fid : bool, optional
        Whether NpzFile should close the file handle.
        Requires that `fid` is a file-like object.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)
    >>> np.savez(outfile, x=x, y=y)
    >>> _ = outfile.seek(0)

    >>> npz = np.load(outfile)
    >>> isinstance(npz, np.lib.npyio.NpzFile)
    True
    >>> npz
    NpzFile 'object' with keys x, y
    >>> sorted(npz.files)
    ['x', 'y']
    >>> npz['x']  # getitem access
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> npz.f.x  # attribute lookup
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    zip = None
    fid = None
    _MAX_REPR_ARRAY_COUNT = 5

    def __init__(self, fid, own_fid=False, allow_pickle=False, pickle_kwargs=None, *, max_header_size=format._MAX_HEADER_SIZE):
        if False:
            while True:
                i = 10
        _zip = zipfile_factory(fid)
        self._files = _zip.namelist()
        self.files = []
        self.allow_pickle = allow_pickle
        self.max_header_size = max_header_size
        self.pickle_kwargs = pickle_kwargs
        for x in self._files:
            if x.endswith('.npy'):
                self.files.append(x[:-4])
            else:
                self.files.append(x)
        self.zip = _zip
        self.f = BagObj(self)
        if own_fid:
            self.fid = fid

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

    def close(self):
        if False:
            return 10
        '\n        Close the file.\n\n        '
        if self.zip is not None:
            self.zip.close()
            self.zip = None
        if self.fid is not None:
            self.fid.close()
            self.fid = None
        self.f = None

    def __del__(self):
        if False:
            return 10
        self.close()

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.files)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.files)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        member = False
        if key in self._files:
            member = True
        elif key in self.files:
            member = True
            key += '.npy'
        if member:
            bytes = self.zip.open(key)
            magic = bytes.read(len(format.MAGIC_PREFIX))
            bytes.close()
            if magic == format.MAGIC_PREFIX:
                bytes = self.zip.open(key)
                return format.read_array(bytes, allow_pickle=self.allow_pickle, pickle_kwargs=self.pickle_kwargs, max_header_size=self.max_header_size)
            else:
                return self.zip.read(key)
        else:
            raise KeyError(f'{key} is not a file in the archive')

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        return key in self._files or key in self.files

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.fid, str):
            filename = self.fid
        else:
            filename = getattr(self.fid, 'name', 'object')
        array_names = ', '.join(self.files[:self._MAX_REPR_ARRAY_COUNT])
        if len(self.files) > self._MAX_REPR_ARRAY_COUNT:
            array_names += '...'
        return f'NpzFile {filename!r} with keys: {array_names}'

@set_module('numpy')
def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=format._MAX_HEADER_SIZE):
    if False:
        return 10
    "\n    Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.\n\n    .. warning:: Loading files that contain object arrays uses the ``pickle``\n                 module, which is not secure against erroneous or maliciously\n                 constructed data. Consider passing ``allow_pickle=False`` to\n                 load data that is known not to contain object arrays for the\n                 safer handling of untrusted sources.\n\n    Parameters\n    ----------\n    file : file-like object, string, or pathlib.Path\n        The file to read. File-like objects must support the\n        ``seek()`` and ``read()`` methods and must always\n        be opened in binary mode.  Pickled files require that the\n        file-like object support the ``readline()`` method as well.\n    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional\n        If not None, then memory-map the file, using the given mode (see\n        `numpy.memmap` for a detailed description of the modes).  A\n        memory-mapped array is kept on disk. However, it can be accessed\n        and sliced like any ndarray.  Memory mapping is especially useful\n        for accessing small fragments of large files without reading the\n        entire file into memory.\n    allow_pickle : bool, optional\n        Allow loading pickled object arrays stored in npy files. Reasons for\n        disallowing pickles include security, as loading pickled data can\n        execute arbitrary code. If pickles are disallowed, loading object\n        arrays will fail. Default: False\n\n        .. versionchanged:: 1.16.3\n            Made default False in response to CVE-2019-6446.\n\n    fix_imports : bool, optional\n        Only useful when loading Python 2 generated pickled files on Python 3,\n        which includes npy/npz files containing object arrays. If `fix_imports`\n        is True, pickle will try to map the old Python 2 names to the new names\n        used in Python 3.\n    encoding : str, optional\n        What encoding to use when reading Python 2 strings. Only useful when\n        loading Python 2 generated pickled files in Python 3, which includes\n        npy/npz files containing object arrays. Values other than 'latin1',\n        'ASCII', and 'bytes' are not allowed, as they can corrupt numerical\n        data. Default: 'ASCII'\n    max_header_size : int, optional\n        Maximum allowed size of the header.  Large headers may not be safe\n        to load securely and thus require explicitly passing a larger value.\n        See :py:func:`ast.literal_eval()` for details.\n        This option is ignored when `allow_pickle` is passed.  In that case\n        the file is by definition trusted and the limit is unnecessary.\n\n    Returns\n    -------\n    result : array, tuple, dict, etc.\n        Data stored in the file. For ``.npz`` files, the returned instance\n        of NpzFile class must be closed to avoid leaking file descriptors.\n\n    Raises\n    ------\n    OSError\n        If the input file does not exist or cannot be read.\n    UnpicklingError\n        If ``allow_pickle=True``, but the file cannot be loaded as a pickle.\n    ValueError\n        The file contains an object array, but ``allow_pickle=False`` given.\n    EOFError\n        When calling ``np.load`` multiple times on the same file handle,\n        if all data has already been read\n\n    See Also\n    --------\n    save, savez, savez_compressed, loadtxt\n    memmap : Create a memory-map to an array stored in a file on disk.\n    lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.\n\n    Notes\n    -----\n    - If the file contains pickle data, then whatever object is stored\n      in the pickle is returned.\n    - If the file is a ``.npy`` file, then a single array is returned.\n    - If the file is a ``.npz`` file, then a dictionary-like object is\n      returned, containing ``{filename: array}`` key-value pairs, one for\n      each file in the archive.\n    - If the file is a ``.npz`` file, the returned value supports the\n      context manager protocol in a similar fashion to the open function::\n\n        with load('foo.npz') as data:\n            a = data['a']\n\n      The underlying file descriptor is closed when exiting the 'with'\n      block.\n\n    Examples\n    --------\n    Store data to disk, and load it again:\n\n    >>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))\n    >>> np.load('/tmp/123.npy')\n    array([[1, 2, 3],\n           [4, 5, 6]])\n\n    Store compressed data to disk, and load it again:\n\n    >>> a=np.array([[1, 2, 3], [4, 5, 6]])\n    >>> b=np.array([1, 2])\n    >>> np.savez('/tmp/123.npz', a=a, b=b)\n    >>> data = np.load('/tmp/123.npz')\n    >>> data['a']\n    array([[1, 2, 3],\n           [4, 5, 6]])\n    >>> data['b']\n    array([1, 2])\n    >>> data.close()\n\n    Mem-map the stored array, and then access the second row\n    directly from disk:\n\n    >>> X = np.load('/tmp/123.npy', mmap_mode='r')\n    >>> X[1, :]\n    memmap([4, 5, 6])\n\n    "
    if encoding not in ('ASCII', 'latin1', 'bytes'):
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")
    pickle_kwargs = dict(encoding=encoding, fix_imports=fix_imports)
    with contextlib.ExitStack() as stack:
        if hasattr(file, 'read'):
            fid = file
            own_fid = False
        else:
            fid = stack.enter_context(open(os.fspath(file), 'rb'))
            own_fid = True
        _ZIP_PREFIX = b'PK\x03\x04'
        _ZIP_SUFFIX = b'PK\x05\x06'
        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)
        if not magic:
            raise EOFError('No data left in file')
        fid.seek(-min(N, len(magic)), 1)
        if magic.startswith(_ZIP_PREFIX) or magic.startswith(_ZIP_SUFFIX):
            stack.pop_all()
            ret = NpzFile(fid, own_fid=own_fid, allow_pickle=allow_pickle, pickle_kwargs=pickle_kwargs, max_header_size=max_header_size)
            return ret
        elif magic == format.MAGIC_PREFIX:
            if mmap_mode:
                if allow_pickle:
                    max_header_size = 2 ** 64
                return format.open_memmap(file, mode=mmap_mode, max_header_size=max_header_size)
            else:
                return format.read_array(fid, allow_pickle=allow_pickle, pickle_kwargs=pickle_kwargs, max_header_size=max_header_size)
        else:
            if not allow_pickle:
                raise ValueError('Cannot load file containing pickled data when allow_pickle=False')
            try:
                return pickle.load(fid, **pickle_kwargs)
            except Exception as e:
                raise pickle.UnpicklingError(f'Failed to interpret file {file!r} as a pickle') from e

def _save_dispatcher(file, arr, allow_pickle=None, fix_imports=None):
    if False:
        while True:
            i = 10
    return (arr,)

@array_function_dispatch(_save_dispatcher)
def save(file, arr, allow_pickle=True, fix_imports=True):
    if False:
        while True:
            i = 10
    "\n    Save an array to a binary file in NumPy ``.npy`` format.\n\n    Parameters\n    ----------\n    file : file, str, or pathlib.Path\n        File or filename to which the data is saved. If file is a file-object,\n        then the filename is unchanged.  If file is a string or Path,\n        a ``.npy`` extension will be appended to the filename if it does not\n        already have one.\n    arr : array_like\n        Array data to be saved.\n    allow_pickle : bool, optional\n        Allow saving object arrays using Python pickles. Reasons for \n        disallowing pickles include security (loading pickled data can execute\n        arbitrary code) and portability (pickled objects may not be loadable \n        on different Python installations, for example if the stored objects\n        require libraries that are not available, and not all pickled data is\n        compatible between Python 2 and Python 3).\n        Default: True\n    fix_imports : bool, optional\n        Only useful in forcing objects in object arrays on Python 3 to be\n        pickled in a Python 2 compatible way. If `fix_imports` is True, pickle\n        will try to map the new Python 3 names to the old module names used in\n        Python 2, so that the pickle data stream is readable with Python 2.\n\n    See Also\n    --------\n    savez : Save several arrays into a ``.npz`` archive\n    savetxt, load\n\n    Notes\n    -----\n    For a description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.\n\n    Any data saved to the file is appended to the end of the file.\n\n    Examples\n    --------\n    >>> from tempfile import TemporaryFile\n    >>> outfile = TemporaryFile()\n\n    >>> x = np.arange(10)\n    >>> np.save(outfile, x)\n\n    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file\n    >>> np.load(outfile)\n    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n\n\n    >>> with open('test.npy', 'wb') as f:\n    ...     np.save(f, np.array([1, 2]))\n    ...     np.save(f, np.array([1, 3]))\n    >>> with open('test.npy', 'rb') as f:\n    ...     a = np.load(f)\n    ...     b = np.load(f)\n    >>> print(a, b)\n    # [1 2] [1 3]\n    "
    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os.fspath(file)
        if not file.endswith('.npy'):
            file = file + '.npy'
        file_ctx = open(file, 'wb')
    with file_ctx as fid:
        arr = np.asanyarray(arr)
        format.write_array(fid, arr, allow_pickle=allow_pickle, pickle_kwargs=dict(fix_imports=fix_imports))

def _savez_dispatcher(file, *args, **kwds):
    if False:
        i = 10
        return i + 15
    yield from args
    yield from kwds.values()

@array_function_dispatch(_savez_dispatcher)
def savez(file, *args, **kwds):
    if False:
        for i in range(10):
            print('nop')
    'Save several arrays into a single file in uncompressed ``.npz`` format.\n\n    Provide arrays as keyword arguments to store them under the\n    corresponding name in the output file: ``savez(fn, x=x, y=y)``.\n\n    If arrays are specified as positional arguments, i.e., ``savez(fn,\n    x, y)``, their names will be `arr_0`, `arr_1`, etc.\n\n    Parameters\n    ----------\n    file : file, str, or pathlib.Path\n        Either the filename (string) or an open file (file-like object)\n        where the data will be saved. If file is a string or a Path, the\n        ``.npz`` extension will be appended to the filename if it is not\n        already there.\n    args : Arguments, optional\n        Arrays to save to the file. Please use keyword arguments (see\n        `kwds` below) to assign names to arrays.  Arrays specified as\n        args will be named "arr_0", "arr_1", and so on.\n    kwds : Keyword arguments, optional\n        Arrays to save to the file. Each array will be saved to the\n        output file with its corresponding keyword name.\n\n    Returns\n    -------\n    None\n\n    See Also\n    --------\n    save : Save a single array to a binary file in NumPy format.\n    savetxt : Save an array to a file as plain text.\n    savez_compressed : Save several arrays into a compressed ``.npz`` archive\n\n    Notes\n    -----\n    The ``.npz`` file format is a zipped archive of files named after the\n    variables they contain.  The archive is not compressed and each file\n    in the archive contains one variable in ``.npy`` format. For a\n    description of the ``.npy`` format, see :py:mod:`numpy.lib.format`.\n\n    When opening the saved ``.npz`` file with `load` a `NpzFile` object is\n    returned. This is a dictionary-like object which can be queried for\n    its list of arrays (with the ``.files`` attribute), and for the arrays\n    themselves.\n\n    Keys passed in `kwds` are used as filenames inside the ZIP archive.\n    Therefore, keys should be valid filenames; e.g., avoid keys that begin with\n    ``/`` or contain ``.``.\n\n    When naming variables with keyword arguments, it is not possible to name a\n    variable ``file``, as this would cause the ``file`` argument to be defined\n    twice in the call to ``savez``.\n\n    Examples\n    --------\n    >>> from tempfile import TemporaryFile\n    >>> outfile = TemporaryFile()\n    >>> x = np.arange(10)\n    >>> y = np.sin(x)\n\n    Using `savez` with \\*args, the arrays are saved with default names.\n\n    >>> np.savez(outfile, x, y)\n    >>> _ = outfile.seek(0) # Only needed to simulate closing & reopening file\n    >>> npzfile = np.load(outfile)\n    >>> npzfile.files\n    [\'arr_0\', \'arr_1\']\n    >>> npzfile[\'arr_0\']\n    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n\n    Using `savez` with \\**kwds, the arrays are saved with the keyword names.\n\n    >>> outfile = TemporaryFile()\n    >>> np.savez(outfile, x=x, y=y)\n    >>> _ = outfile.seek(0)\n    >>> npzfile = np.load(outfile)\n    >>> sorted(npzfile.files)\n    [\'x\', \'y\']\n    >>> npzfile[\'x\']\n    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n\n    '
    _savez(file, args, kwds, False)

def _savez_compressed_dispatcher(file, *args, **kwds):
    if False:
        print('Hello World!')
    yield from args
    yield from kwds.values()

@array_function_dispatch(_savez_compressed_dispatcher)
def savez_compressed(file, *args, **kwds):
    if False:
        return 10
    '\n    Save several arrays into a single file in compressed ``.npz`` format.\n\n    Provide arrays as keyword arguments to store them under the\n    corresponding name in the output file: ``savez_compressed(fn, x=x, y=y)``.\n\n    If arrays are specified as positional arguments, i.e.,\n    ``savez_compressed(fn, x, y)``, their names will be `arr_0`, `arr_1`, etc.\n\n    Parameters\n    ----------\n    file : file, str, or pathlib.Path\n        Either the filename (string) or an open file (file-like object)\n        where the data will be saved. If file is a string or a Path, the\n        ``.npz`` extension will be appended to the filename if it is not\n        already there.\n    args : Arguments, optional\n        Arrays to save to the file. Please use keyword arguments (see\n        `kwds` below) to assign names to arrays.  Arrays specified as\n        args will be named "arr_0", "arr_1", and so on.\n    kwds : Keyword arguments, optional\n        Arrays to save to the file. Each array will be saved to the\n        output file with its corresponding keyword name.\n\n    Returns\n    -------\n    None\n\n    See Also\n    --------\n    numpy.save : Save a single array to a binary file in NumPy format.\n    numpy.savetxt : Save an array to a file as plain text.\n    numpy.savez : Save several arrays into an uncompressed ``.npz`` file format\n    numpy.load : Load the files created by savez_compressed.\n\n    Notes\n    -----\n    The ``.npz`` file format is a zipped archive of files named after the\n    variables they contain.  The archive is compressed with\n    ``zipfile.ZIP_DEFLATED`` and each file in the archive contains one variable\n    in ``.npy`` format. For a description of the ``.npy`` format, see\n    :py:mod:`numpy.lib.format`.\n\n\n    When opening the saved ``.npz`` file with `load` a `NpzFile` object is\n    returned. This is a dictionary-like object which can be queried for\n    its list of arrays (with the ``.files`` attribute), and for the arrays\n    themselves.\n\n    Examples\n    --------\n    >>> test_array = np.random.rand(3, 2)\n    >>> test_vector = np.random.rand(4)\n    >>> np.savez_compressed(\'/tmp/123\', a=test_array, b=test_vector)\n    >>> loaded = np.load(\'/tmp/123.npz\')\n    >>> print(np.array_equal(test_array, loaded[\'a\']))\n    True\n    >>> print(np.array_equal(test_vector, loaded[\'b\']))\n    True\n\n    '
    _savez(file, args, kwds, True)

def _savez(file, args, kwds, compress, allow_pickle=True, pickle_kwargs=None):
    if False:
        while True:
            i = 10
    import zipfile
    if not hasattr(file, 'write'):
        file = os.fspath(file)
        if not file.endswith('.npz'):
            file = file + '.npz'
    namedict = kwds
    for (i, val) in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError('Cannot use un-named variables and keyword %s' % key)
        namedict[key] = val
    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED
    zipf = zipfile_factory(file, mode='w', compression=compression)
    for (key, val) in namedict.items():
        fname = key + '.npy'
        val = np.asanyarray(val)
        with zipf.open(fname, 'w', force_zip64=True) as fid:
            format.write_array(fid, val, allow_pickle=allow_pickle, pickle_kwargs=pickle_kwargs)
    zipf.close()

def _ensure_ndmin_ndarray_check_param(ndmin):
    if False:
        return 10
    'Just checks if the param ndmin is supported on\n        _ensure_ndmin_ndarray. It is intended to be used as\n        verification before running anything expensive.\n        e.g. loadtxt, genfromtxt\n    '
    if ndmin not in [0, 1, 2]:
        raise ValueError(f'Illegal value of ndmin keyword: {ndmin}')

def _ensure_ndmin_ndarray(a, *, ndmin: int):
    if False:
        i = 10
        return i + 15
    'This is a helper function of loadtxt and genfromtxt to ensure\n        proper minimum dimension as requested\n\n        ndim : int. Supported values 1, 2, 3\n                    ^^ whenever this changes, keep in sync with\n                       _ensure_ndmin_ndarray_check_param\n    '
    if a.ndim > ndmin:
        a = np.squeeze(a)
    if a.ndim < ndmin:
        if ndmin == 1:
            a = np.atleast_1d(a)
        elif ndmin == 2:
            a = np.atleast_2d(a).T
    return a
_loadtxt_chunksize = 50000

def _check_nonneg_int(value, name='argument'):
    if False:
        for i in range(10):
            print('nop')
    try:
        operator.index(value)
    except TypeError:
        raise TypeError(f'{name} must be an integer') from None
    if value < 0:
        raise ValueError(f'{name} must be nonnegative')

def _preprocess_comments(iterable, comments, encoding):
    if False:
        return 10
    '\n    Generator that consumes a line iterated iterable and strips out the\n    multiple (or multi-character) comments from lines.\n    This is a pre-processing step to achieve feature parity with loadtxt\n    (we assume that this feature is a nieche feature).\n    '
    for line in iterable:
        if isinstance(line, bytes):
            line = line.decode(encoding)
        for c in comments:
            line = line.split(c, 1)[0]
        yield line
_loadtxt_chunksize = 50000

def _read(fname, *, delimiter=',', comment='#', quote='"', imaginary_unit='j', usecols=None, skiplines=0, max_rows=None, converters=None, ndmin=None, unpack=False, dtype=np.float64, encoding='bytes'):
    if False:
        i = 10
        return i + 15
    '\n    Read a NumPy array from a text file.\n    This is a helper function for loadtxt.\n\n    Parameters\n    ----------\n    fname : file, str, or pathlib.Path\n        The filename or the file to be read.\n    delimiter : str, optional\n        Field delimiter of the fields in line of the file.\n        Default is a comma, \',\'.  If None any sequence of whitespace is\n        considered a delimiter.\n    comment : str or sequence of str or None, optional\n        Character that begins a comment.  All text from the comment\n        character to the end of the line is ignored.\n        Multiple comments or multiple-character comment strings are supported,\n        but may be slower and `quote` must be empty if used.\n        Use None to disable all use of comments.\n    quote : str or None, optional\n        Character that is used to quote string fields. Default is \'"\'\n        (a double quote). Use None to disable quote support.\n    imaginary_unit : str, optional\n        Character that represent the imaginary unit `sqrt(-1)`.\n        Default is \'j\'.\n    usecols : array_like, optional\n        A one-dimensional array of integer column numbers.  These are the\n        columns from the file to be included in the array.  If this value\n        is not given, all the columns are used.\n    skiplines : int, optional\n        Number of lines to skip before interpreting the data in the file.\n    max_rows : int, optional\n        Maximum number of rows of data to read.  Default is to read the\n        entire file.\n    converters : dict or callable, optional\n        A function to parse all columns strings into the desired value, or\n        a dictionary mapping column number to a parser function.\n        E.g. if column 0 is a date string: ``converters = {0: datestr2num}``.\n        Converters can also be used to provide a default value for missing\n        data, e.g. ``converters = lambda s: float(s.strip() or 0)`` will\n        convert empty fields to 0.\n        Default: None\n    ndmin : int, optional\n        Minimum dimension of the array returned.\n        Allowed values are 0, 1 or 2.  Default is 0.\n    unpack : bool, optional\n        If True, the returned array is transposed, so that arguments may be\n        unpacked using ``x, y, z = read(...)``.  When used with a structured\n        data-type, arrays are returned for each field.  Default is False.\n    dtype : numpy data type\n        A NumPy dtype instance, can be a structured dtype to map to the\n        columns of the file.\n    encoding : str, optional\n        Encoding used to decode the inputfile. The special value \'bytes\'\n        (the default) enables backwards-compatible behavior for `converters`,\n        ensuring that inputs to the converter functions are encoded\n        bytes objects. The special value \'bytes\' has no additional effect if\n        ``converters=None``. If encoding is ``\'bytes\'`` or ``None``, the\n        default system encoding is used.\n\n    Returns\n    -------\n    ndarray\n        NumPy array.\n    '
    byte_converters = False
    if encoding == 'bytes':
        encoding = None
        byte_converters = True
    if dtype is None:
        raise TypeError('a dtype must be provided.')
    dtype = np.dtype(dtype)
    read_dtype_via_object_chunks = None
    if dtype.kind in 'SUM' and (dtype == 'S0' or dtype == 'U0' or dtype == 'M8' or (dtype == 'm8')):
        read_dtype_via_object_chunks = dtype
        dtype = np.dtype(object)
    if usecols is not None:
        try:
            usecols = list(usecols)
        except TypeError:
            usecols = [usecols]
    _ensure_ndmin_ndarray_check_param(ndmin)
    if comment is None:
        comments = None
    else:
        if '' in comment:
            raise ValueError('comments cannot be an empty string. Use comments=None to disable comments.')
        comments = tuple(comment)
        comment = None
        if len(comments) == 0:
            comments = None
        elif len(comments) == 1:
            if isinstance(comments[0], str) and len(comments[0]) == 1:
                comment = comments[0]
                comments = None
        elif delimiter in comments:
            raise TypeError(f"Comment characters '{comments}' cannot include the delimiter '{delimiter}'")
    if comments is not None:
        if quote is not None:
            raise ValueError('when multiple comments or a multi-character comment is given, quotes are not supported.  In this case quotechar must be set to None.')
    if len(imaginary_unit) != 1:
        raise ValueError('len(imaginary_unit) must be 1.')
    _check_nonneg_int(skiplines)
    if max_rows is not None:
        _check_nonneg_int(max_rows)
    else:
        max_rows = -1
    fh_closing_ctx = contextlib.nullcontext()
    filelike = False
    try:
        if isinstance(fname, os.PathLike):
            fname = os.fspath(fname)
        if isinstance(fname, str):
            fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
            if encoding is None:
                encoding = getattr(fh, 'encoding', 'latin1')
            fh_closing_ctx = contextlib.closing(fh)
            data = fh
            filelike = True
        else:
            if encoding is None:
                encoding = getattr(fname, 'encoding', 'latin1')
            data = iter(fname)
    except TypeError as e:
        raise ValueError(f'fname must be a string, filehandle, list of strings,\nor generator. Got {type(fname)} instead.') from e
    with fh_closing_ctx:
        if comments is not None:
            if filelike:
                data = iter(data)
                filelike = False
            data = _preprocess_comments(data, comments, encoding)
        if read_dtype_via_object_chunks is None:
            arr = _load_from_filelike(data, delimiter=delimiter, comment=comment, quote=quote, imaginary_unit=imaginary_unit, usecols=usecols, skiplines=skiplines, max_rows=max_rows, converters=converters, dtype=dtype, encoding=encoding, filelike=filelike, byte_converters=byte_converters)
        else:
            if filelike:
                data = iter(data)
            c_byte_converters = False
            if read_dtype_via_object_chunks == 'S':
                c_byte_converters = True
            chunks = []
            while max_rows != 0:
                if max_rows < 0:
                    chunk_size = _loadtxt_chunksize
                else:
                    chunk_size = min(_loadtxt_chunksize, max_rows)
                next_arr = _load_from_filelike(data, delimiter=delimiter, comment=comment, quote=quote, imaginary_unit=imaginary_unit, usecols=usecols, skiplines=skiplines, max_rows=max_rows, converters=converters, dtype=dtype, encoding=encoding, filelike=filelike, byte_converters=byte_converters, c_byte_converters=c_byte_converters)
                chunks.append(next_arr.astype(read_dtype_via_object_chunks))
                skiprows = 0
                if max_rows >= 0:
                    max_rows -= chunk_size
                if len(next_arr) < chunk_size:
                    break
            if len(chunks) > 1 and len(chunks[-1]) == 0:
                del chunks[-1]
            if len(chunks) == 1:
                arr = chunks[0]
            else:
                arr = np.concatenate(chunks, axis=0)
    arr = _ensure_ndmin_ndarray(arr, ndmin=ndmin)
    if arr.shape:
        if arr.shape[0] == 0:
            warnings.warn(f'loadtxt: input contained no data: "{fname}"', category=UserWarning, stacklevel=3)
    if unpack:
        dt = arr.dtype
        if dt.names is not None:
            return [arr[field] for field in dt.names]
        else:
            return arr.T
    else:
        return arr

@set_array_function_like_doc
@set_module('numpy')
def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, quotechar=None, like=None):
    if False:
        i = 10
        return i + 15
    '\n    Load data from a text file.\n\n    Parameters\n    ----------\n    fname : file, str, pathlib.Path, list of str, generator\n        File, filename, list, or generator to read.  If the filename\n        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note\n        that generators must return bytes or strings. The strings\n        in a list or produced by a generator are treated as lines.\n    dtype : data-type, optional\n        Data-type of the resulting array; default: float.  If this is a\n        structured data-type, the resulting array will be 1-dimensional, and\n        each row will be interpreted as an element of the array.  In this\n        case, the number of columns used must match the number of fields in\n        the data-type.\n    comments : str or sequence of str or None, optional\n        The characters or list of characters used to indicate the start of a\n        comment. None implies no comments. For backwards compatibility, byte\n        strings will be decoded as \'latin1\'. The default is \'#\'.\n    delimiter : str, optional\n        The character used to separate the values. For backwards compatibility,\n        byte strings will be decoded as \'latin1\'. The default is whitespace.\n\n        .. versionchanged:: 1.23.0\n           Only single character delimiters are supported. Newline characters\n           cannot be used as the delimiter.\n\n    converters : dict or callable, optional\n        Converter functions to customize value parsing. If `converters` is\n        callable, the function is applied to all columns, else it must be a\n        dict that maps column number to a parser function.\n        See examples for further details.\n        Default: None.\n\n        .. versionchanged:: 1.23.0\n           The ability to pass a single callable to be applied to all columns\n           was added.\n\n    skiprows : int, optional\n        Skip the first `skiprows` lines, including comments; default: 0.\n    usecols : int or sequence, optional\n        Which columns to read, with 0 being the first. For example,\n        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.\n        The default, None, results in all columns being read.\n\n        .. versionchanged:: 1.11.0\n            When a single column has to be read it is possible to use\n            an integer instead of a tuple. E.g ``usecols = 3`` reads the\n            fourth column the same way as ``usecols = (3,)`` would.\n    unpack : bool, optional\n        If True, the returned array is transposed, so that arguments may be\n        unpacked using ``x, y, z = loadtxt(...)``.  When used with a\n        structured data-type, arrays are returned for each field.\n        Default is False.\n    ndmin : int, optional\n        The returned array will have at least `ndmin` dimensions.\n        Otherwise mono-dimensional axes will be squeezed.\n        Legal values: 0 (default), 1 or 2.\n\n        .. versionadded:: 1.6.0\n    encoding : str, optional\n        Encoding used to decode the inputfile. Does not apply to input streams.\n        The special value \'bytes\' enables backward compatibility workarounds\n        that ensures you receive byte arrays as results if possible and passes\n        \'latin1\' encoded strings to converters. Override this value to receive\n        unicode arrays and pass strings as input to converters.  If set to None\n        the system default is used. The default value is \'bytes\'.\n\n        .. versionadded:: 1.14.0\n    max_rows : int, optional\n        Read `max_rows` rows of content after `skiprows` lines. The default is\n        to read all the rows. Note that empty rows containing no data such as\n        empty lines and comment lines are not counted towards `max_rows`,\n        while such lines are counted in `skiprows`.\n\n        .. versionadded:: 1.16.0\n\n        .. versionchanged:: 1.23.0\n            Lines containing no data, including comment lines (e.g., lines\n            starting with \'#\' or as specified via `comments`) are not counted\n            towards `max_rows`.\n    quotechar : unicode character or None, optional\n        The character used to denote the start and end of a quoted item.\n        Occurrences of the delimiter or comment characters are ignored within\n        a quoted item. The default value is ``quotechar=None``, which means\n        quoting support is disabled.\n\n        If two consecutive instances of `quotechar` are found within a quoted\n        field, the first is treated as an escape character. See examples.\n\n        .. versionadded:: 1.23.0\n    ${ARRAY_FUNCTION_LIKE}\n\n        .. versionadded:: 1.20.0\n\n    Returns\n    -------\n    out : ndarray\n        Data read from the text file.\n\n    See Also\n    --------\n    load, fromstring, fromregex\n    genfromtxt : Load data with missing values handled as specified.\n    scipy.io.loadmat : reads MATLAB data files\n\n    Notes\n    -----\n    This function aims to be a fast reader for simply formatted files.  The\n    `genfromtxt` function provides more sophisticated handling of, e.g.,\n    lines with missing values.\n\n    Each row in the input text file must have the same number of values to be\n    able to read all values. If all rows do not have same number of values, a\n    subset of up to n columns (where n is the least number of values present\n    in all rows) can be read by specifying the columns via `usecols`.\n\n    .. versionadded:: 1.10.0\n\n    The strings produced by the Python float.hex method can be used as\n    input for floats.\n\n    Examples\n    --------\n    >>> from io import StringIO   # StringIO behaves like a file object\n    >>> c = StringIO("0 1\\n2 3")\n    >>> np.loadtxt(c)\n    array([[0., 1.],\n           [2., 3.]])\n\n    >>> d = StringIO("M 21 72\\nF 35 58")\n    >>> np.loadtxt(d, dtype={\'names\': (\'gender\', \'age\', \'weight\'),\n    ...                      \'formats\': (\'S1\', \'i4\', \'f4\')})\n    array([(b\'M\', 21, 72.), (b\'F\', 35, 58.)],\n          dtype=[(\'gender\', \'S1\'), (\'age\', \'<i4\'), (\'weight\', \'<f4\')])\n\n    >>> c = StringIO("1,0,2\\n3,0,4")\n    >>> x, y = np.loadtxt(c, delimiter=\',\', usecols=(0, 2), unpack=True)\n    >>> x\n    array([1., 3.])\n    >>> y\n    array([2., 4.])\n\n    The `converters` argument is used to specify functions to preprocess the\n    text prior to parsing. `converters` can be a dictionary that maps\n    preprocessing functions to each column:\n\n    >>> s = StringIO("1.618, 2.296\\n3.141, 4.669\\n")\n    >>> conv = {\n    ...     0: lambda x: np.floor(float(x)),  # conversion fn for column 0\n    ...     1: lambda x: np.ceil(float(x)),  # conversion fn for column 1\n    ... }\n    >>> np.loadtxt(s, delimiter=",", converters=conv)\n    array([[1., 3.],\n           [3., 5.]])\n\n    `converters` can be a callable instead of a dictionary, in which case it\n    is applied to all columns:\n\n    >>> s = StringIO("0xDE 0xAD\\n0xC0 0xDE")\n    >>> import functools\n    >>> conv = functools.partial(int, base=16)\n    >>> np.loadtxt(s, converters=conv)\n    array([[222., 173.],\n           [192., 222.]])\n\n    This example shows how `converters` can be used to convert a field\n    with a trailing minus sign into a negative number.\n\n    >>> s = StringIO(\'10.01 31.25-\\n19.22 64.31\\n17.57- 63.94\')\n    >>> def conv(fld):\n    ...     return -float(fld[:-1]) if fld.endswith(b\'-\') else float(fld)\n    ...\n    >>> np.loadtxt(s, converters=conv)\n    array([[ 10.01, -31.25],\n           [ 19.22,  64.31],\n           [-17.57,  63.94]])\n\n    Using a callable as the converter can be particularly useful for handling\n    values with different formatting, e.g. floats with underscores:\n\n    >>> s = StringIO("1 2.7 100_000")\n    >>> np.loadtxt(s, converters=float)\n    array([1.e+00, 2.7e+00, 1.e+05])\n\n    This idea can be extended to automatically handle values specified in\n    many different formats:\n\n    >>> def conv(val):\n    ...     try:\n    ...         return float(val)\n    ...     except ValueError:\n    ...         return float.fromhex(val)\n    >>> s = StringIO("1, 2.5, 3_000, 0b4, 0x1.4000000000000p+2")\n    >>> np.loadtxt(s, delimiter=",", converters=conv, encoding=None)\n    array([1.0e+00, 2.5e+00, 3.0e+03, 1.8e+02, 5.0e+00])\n\n    Note that with the default ``encoding="bytes"``, the inputs to the\n    converter function are latin-1 encoded byte strings. To deactivate the\n    implicit encoding prior to conversion, use ``encoding=None``\n\n    >>> s = StringIO(\'10.01 31.25-\\n19.22 64.31\\n17.57- 63.94\')\n    >>> conv = lambda x: -float(x[:-1]) if x.endswith(\'-\') else float(x)\n    >>> np.loadtxt(s, converters=conv, encoding=None)\n    array([[ 10.01, -31.25],\n           [ 19.22,  64.31],\n           [-17.57,  63.94]])\n\n    Support for quoted fields is enabled with the `quotechar` parameter.\n    Comment and delimiter characters are ignored when they appear within a\n    quoted item delineated by `quotechar`:\n\n    >>> s = StringIO(\'"alpha, #42", 10.0\\n"beta, #64", 2.0\\n\')\n    >>> dtype = np.dtype([("label", "U12"), ("value", float)])\n    >>> np.loadtxt(s, dtype=dtype, delimiter=",", quotechar=\'"\')\n    array([(\'alpha, #42\', 10.), (\'beta, #64\',  2.)],\n          dtype=[(\'label\', \'<U12\'), (\'value\', \'<f8\')])\n\n    Quoted fields can be separated by multiple whitespace characters:\n\n    >>> s = StringIO(\'"alpha, #42"       10.0\\n"beta, #64" 2.0\\n\')\n    >>> dtype = np.dtype([("label", "U12"), ("value", float)])\n    >>> np.loadtxt(s, dtype=dtype, delimiter=None, quotechar=\'"\')\n    array([(\'alpha, #42\', 10.), (\'beta, #64\',  2.)],\n          dtype=[(\'label\', \'<U12\'), (\'value\', \'<f8\')])\n\n    Two consecutive quote characters within a quoted field are treated as a\n    single escaped character:\n\n    >>> s = StringIO(\'"Hello, my name is ""Monty""!"\')\n    >>> np.loadtxt(s, dtype="U", delimiter=",", quotechar=\'"\')\n    array(\'Hello, my name is "Monty"!\', dtype=\'<U26\')\n\n    Read subset of columns when all rows do not contain equal number of values:\n\n    >>> d = StringIO("1 2\\n2 4\\n3 9 12\\n4 16 20")\n    >>> np.loadtxt(d, usecols=(0, 1))\n    array([[ 1.,  2.],\n           [ 2.,  4.],\n           [ 3.,  9.],\n           [ 4., 16.]])\n\n    '
    if like is not None:
        return _loadtxt_with_like(like, fname, dtype=dtype, comments=comments, delimiter=delimiter, converters=converters, skiprows=skiprows, usecols=usecols, unpack=unpack, ndmin=ndmin, encoding=encoding, max_rows=max_rows)
    if isinstance(delimiter, bytes):
        delimiter.decode('latin1')
    if dtype is None:
        dtype = np.float64
    comment = comments
    if comment is not None:
        if isinstance(comment, (str, bytes)):
            comment = [comment]
        comment = [x.decode('latin1') if isinstance(x, bytes) else x for x in comment]
    if isinstance(delimiter, bytes):
        delimiter = delimiter.decode('latin1')
    arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter, converters=converters, skiplines=skiprows, usecols=usecols, unpack=unpack, ndmin=ndmin, encoding=encoding, max_rows=max_rows, quote=quotechar)
    return arr
_loadtxt_with_like = array_function_dispatch()(loadtxt)

def _savetxt_dispatcher(fname, X, fmt=None, delimiter=None, newline=None, header=None, footer=None, comments=None, encoding=None):
    if False:
        i = 10
        return i + 15
    return (X,)

@array_function_dispatch(_savetxt_dispatcher)
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None):
    if False:
        return 10
    "\n    Save an array to a text file.\n\n    Parameters\n    ----------\n    fname : filename, file handle or pathlib.Path\n        If the filename ends in ``.gz``, the file is automatically saved in\n        compressed gzip format.  `loadtxt` understands gzipped files\n        transparently.\n    X : 1D or 2D array_like\n        Data to be saved to a text file.\n    fmt : str or sequence of strs, optional\n        A single format (%10.5f), a sequence of formats, or a\n        multi-format string, e.g. 'Iteration %d -- %10.5f', in which\n        case `delimiter` is ignored. For complex `X`, the legal options\n        for `fmt` are:\n\n        * a single specifier, `fmt='%.4e'`, resulting in numbers formatted\n          like `' (%s+%sj)' % (fmt, fmt)`\n        * a full string specifying every real and imaginary part, e.g.\n          `' %.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'` for 3 columns\n        * a list of specifiers, one per column - in this case, the real\n          and imaginary part must have separate specifiers,\n          e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns\n    delimiter : str, optional\n        String or character separating columns.\n    newline : str, optional\n        String or character separating lines.\n\n        .. versionadded:: 1.5.0\n    header : str, optional\n        String that will be written at the beginning of the file.\n\n        .. versionadded:: 1.7.0\n    footer : str, optional\n        String that will be written at the end of the file.\n\n        .. versionadded:: 1.7.0\n    comments : str, optional\n        String that will be prepended to the ``header`` and ``footer`` strings,\n        to mark them as comments. Default: '# ',  as expected by e.g.\n        ``numpy.loadtxt``.\n\n        .. versionadded:: 1.7.0\n    encoding : {None, str}, optional\n        Encoding used to encode the outputfile. Does not apply to output\n        streams. If the encoding is something other than 'bytes' or 'latin1'\n        you will not be able to load the file in NumPy versions < 1.14. Default\n        is 'latin1'.\n\n        .. versionadded:: 1.14.0\n\n\n    See Also\n    --------\n    save : Save an array to a binary file in NumPy ``.npy`` format\n    savez : Save several arrays into an uncompressed ``.npz`` archive\n    savez_compressed : Save several arrays into a compressed ``.npz`` archive\n\n    Notes\n    -----\n    Further explanation of the `fmt` parameter\n    (``%[flag]width[.precision]specifier``):\n\n    flags:\n        ``-`` : left justify\n\n        ``+`` : Forces to precede result with + or -.\n\n        ``0`` : Left pad the number with zeros instead of space (see width).\n\n    width:\n        Minimum number of characters to be printed. The value is not truncated\n        if it has more characters.\n\n    precision:\n        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of\n          digits.\n        - For ``e, E`` and ``f`` specifiers, the number of digits to print\n          after the decimal point.\n        - For ``g`` and ``G``, the maximum number of significant digits.\n        - For ``s``, the maximum number of characters.\n\n    specifiers:\n        ``c`` : character\n\n        ``d`` or ``i`` : signed decimal integer\n\n        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.\n\n        ``f`` : decimal floating point\n\n        ``g,G`` : use the shorter of ``e,E`` or ``f``\n\n        ``o`` : signed octal\n\n        ``s`` : string of characters\n\n        ``u`` : unsigned decimal integer\n\n        ``x,X`` : unsigned hexadecimal integer\n\n    This explanation of ``fmt`` is not complete, for an exhaustive\n    specification see [1]_.\n\n    References\n    ----------\n    .. [1] `Format Specification Mini-Language\n           <https://docs.python.org/library/string.html#format-specification-mini-language>`_,\n           Python Documentation.\n\n    Examples\n    --------\n    >>> x = y = z = np.arange(0.0,5.0,1.0)\n    >>> np.savetxt('test.out', x, delimiter=',')   # X is an array\n    >>> np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays\n    >>> np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation\n\n    "

    class WriteWrap:
        """Convert to bytes on bytestream inputs.

        """

        def __init__(self, fh, encoding):
            if False:
                print('Hello World!')
            self.fh = fh
            self.encoding = encoding
            self.do_write = self.first_write

        def close(self):
            if False:
                i = 10
                return i + 15
            self.fh.close()

        def write(self, v):
            if False:
                return 10
            self.do_write(v)

        def write_bytes(self, v):
            if False:
                print('Hello World!')
            if isinstance(v, bytes):
                self.fh.write(v)
            else:
                self.fh.write(v.encode(self.encoding))

        def write_normal(self, v):
            if False:
                print('Hello World!')
            self.fh.write(asunicode(v))

        def first_write(self, v):
            if False:
                return 10
            try:
                self.write_normal(v)
                self.write = self.write_normal
            except TypeError:
                self.write_bytes(v)
                self.write = self.write_bytes
    own_fh = False
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if _is_string_like(fname):
        open(fname, 'wt').close()
        fh = np.lib._datasource.open(fname, 'wt', encoding=encoding)
        own_fh = True
    elif hasattr(fname, 'write'):
        fh = WriteWrap(fname, encoding or 'latin1')
    else:
        raise ValueError('fname must be a string or file handle')
    try:
        X = np.asarray(X)
        if X.ndim == 0 or X.ndim > 2:
            raise ValueError('Expected 1D or 2D array, got %dD array instead' % X.ndim)
        elif X.ndim == 1:
            if X.dtype.names is None:
                X = np.atleast_2d(X).T
                ncol = 1
            else:
                ncol = len(X.dtype.names)
        else:
            ncol = X.shape[1]
        iscomplex_X = np.iscomplexobj(X)
        if type(fmt) in (list, tuple):
            if len(fmt) != ncol:
                raise AttributeError('fmt has wrong shape.  %s' % str(fmt))
            format = delimiter.join(fmt)
        elif isinstance(fmt, str):
            n_fmt_chars = fmt.count('%')
            error = ValueError('fmt has wrong number of %% formats:  %s' % fmt)
            if n_fmt_chars == 1:
                if iscomplex_X:
                    fmt = [' (%s+%sj)' % (fmt, fmt)] * ncol
                else:
                    fmt = [fmt] * ncol
                format = delimiter.join(fmt)
            elif iscomplex_X and n_fmt_chars != 2 * ncol:
                raise error
            elif not iscomplex_X and n_fmt_chars != ncol:
                raise error
            else:
                format = fmt
        else:
            raise ValueError('invalid fmt: %r' % (fmt,))
        if len(header) > 0:
            header = header.replace('\n', '\n' + comments)
            fh.write(comments + header + newline)
        if iscomplex_X:
            for row in X:
                row2 = []
                for number in row:
                    row2.append(number.real)
                    row2.append(number.imag)
                s = format % tuple(row2) + newline
                fh.write(s.replace('+-', '-'))
        else:
            for row in X:
                try:
                    v = format % tuple(row) + newline
                except TypeError as e:
                    raise TypeError("Mismatch between array dtype ('%s') and format specifier ('%s')" % (str(X.dtype), format)) from e
                fh.write(v)
        if len(footer) > 0:
            footer = footer.replace('\n', '\n' + comments)
            fh.write(comments + footer + newline)
    finally:
        if own_fh:
            fh.close()

@set_module('numpy')
def fromregex(file, regexp, dtype, encoding=None):
    if False:
        return 10
    '\n    Construct an array from a text file, using regular expression parsing.\n\n    The returned array is always a structured array, and is constructed from\n    all matches of the regular expression in the file. Groups in the regular\n    expression are converted to fields of the structured array.\n\n    Parameters\n    ----------\n    file : file, str, or pathlib.Path\n        Filename or file object to read.\n\n        .. versionchanged:: 1.22.0\n            Now accepts `os.PathLike` implementations.\n    regexp : str or regexp\n        Regular expression used to parse the file.\n        Groups in the regular expression correspond to fields in the dtype.\n    dtype : dtype or list of dtypes\n        Dtype for the structured array; must be a structured datatype.\n    encoding : str, optional\n        Encoding used to decode the inputfile. Does not apply to input streams.\n\n        .. versionadded:: 1.14.0\n\n    Returns\n    -------\n    output : ndarray\n        The output array, containing the part of the content of `file` that\n        was matched by `regexp`. `output` is always a structured array.\n\n    Raises\n    ------\n    TypeError\n        When `dtype` is not a valid dtype for a structured array.\n\n    See Also\n    --------\n    fromstring, loadtxt\n\n    Notes\n    -----\n    Dtypes for structured arrays can be specified in several forms, but all\n    forms specify at least the data type and field name. For details see\n    `basics.rec`.\n\n    Examples\n    --------\n    >>> from io import StringIO\n    >>> text = StringIO("1312 foo\\n1534  bar\\n444   qux")\n\n    >>> regexp = r"(\\d+)\\s+(...)"  # match [digits, whitespace, anything]\n    >>> output = np.fromregex(text, regexp,\n    ...                       [(\'num\', np.int64), (\'key\', \'S3\')])\n    >>> output\n    array([(1312, b\'foo\'), (1534, b\'bar\'), ( 444, b\'qux\')],\n          dtype=[(\'num\', \'<i8\'), (\'key\', \'S3\')])\n    >>> output[\'num\']\n    array([1312, 1534,  444])\n\n    '
    own_fh = False
    if not hasattr(file, 'read'):
        file = os.fspath(file)
        file = np.lib._datasource.open(file, 'rt', encoding=encoding)
        own_fh = True
    try:
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        if dtype.names is None:
            raise TypeError('dtype must be a structured datatype.')
        content = file.read()
        if isinstance(content, bytes) and isinstance(regexp, str):
            regexp = asbytes(regexp)
        if not hasattr(regexp, 'match'):
            regexp = re.compile(regexp)
        seq = regexp.findall(content)
        if seq and (not isinstance(seq[0], tuple)):
            newdtype = np.dtype(dtype[dtype.names[0]])
            output = np.array(seq, dtype=newdtype)
            output.dtype = dtype
        else:
            output = np.array(seq, dtype=dtype)
        return output
    finally:
        if own_fh:
            file.close()

@set_array_function_like_doc
@set_module('numpy')
def genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=''.join(sorted(NameValidator.defaultdeletechars)), replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes', *, ndmin=0, like=None):
    if False:
        print('Hello World!')
    '\n    Load data from a text file, with missing values handled as specified.\n\n    Each line past the first `skip_header` lines is split at the `delimiter`\n    character, and characters following the `comments` character are discarded.\n\n    Parameters\n    ----------\n    fname : file, str, pathlib.Path, list of str, generator\n        File, filename, list, or generator to read.  If the filename\n        extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note\n        that generators must return bytes or strings. The strings\n        in a list or produced by a generator are treated as lines.\n    dtype : dtype, optional\n        Data type of the resulting array.\n        If None, the dtypes will be determined by the contents of each\n        column, individually.\n    comments : str, optional\n        The character used to indicate the start of a comment.\n        All the characters occurring on a line after a comment are discarded.\n    delimiter : str, int, or sequence, optional\n        The string used to separate values.  By default, any consecutive\n        whitespaces act as delimiter.  An integer or sequence of integers\n        can also be provided as width(s) of each field.\n    skiprows : int, optional\n        `skiprows` was removed in numpy 1.10. Please use `skip_header` instead.\n    skip_header : int, optional\n        The number of lines to skip at the beginning of the file.\n    skip_footer : int, optional\n        The number of lines to skip at the end of the file.\n    converters : variable, optional\n        The set of functions that convert the data of a column to a value.\n        The converters can also be used to provide a default value\n        for missing data: ``converters = {3: lambda s: float(s or 0)}``.\n    missing : variable, optional\n        `missing` was removed in numpy 1.10. Please use `missing_values`\n        instead.\n    missing_values : variable, optional\n        The set of strings corresponding to missing data.\n    filling_values : variable, optional\n        The set of values to be used as default when the data are missing.\n    usecols : sequence, optional\n        Which columns to read, with 0 being the first.  For example,\n        ``usecols = (1, 4, 5)`` will extract the 2nd, 5th and 6th columns.\n    names : {None, True, str, sequence}, optional\n        If `names` is True, the field names are read from the first line after\n        the first `skip_header` lines. This line can optionally be preceded\n        by a comment delimiter. If `names` is a sequence or a single-string of\n        comma-separated names, the names will be used to define the field names\n        in a structured dtype. If `names` is None, the names of the dtype\n        fields will be used, if any.\n    excludelist : sequence, optional\n        A list of names to exclude. This list is appended to the default list\n        [\'return\',\'file\',\'print\']. Excluded names are appended with an\n        underscore: for example, `file` would become `file_`.\n    deletechars : str, optional\n        A string combining invalid characters that must be deleted from the\n        names.\n    defaultfmt : str, optional\n        A format used to define default field names, such as "f%i" or "f_%02i".\n    autostrip : bool, optional\n        Whether to automatically strip white spaces from the variables.\n    replace_space : char, optional\n        Character(s) used in replacement of white spaces in the variable\n        names. By default, use a \'_\'.\n    case_sensitive : {True, False, \'upper\', \'lower\'}, optional\n        If True, field names are case sensitive.\n        If False or \'upper\', field names are converted to upper case.\n        If \'lower\', field names are converted to lower case.\n    unpack : bool, optional\n        If True, the returned array is transposed, so that arguments may be\n        unpacked using ``x, y, z = genfromtxt(...)``.  When used with a\n        structured data-type, arrays are returned for each field.\n        Default is False.\n    usemask : bool, optional\n        If True, return a masked array.\n        If False, return a regular array.\n    loose : bool, optional\n        If True, do not raise errors for invalid values.\n    invalid_raise : bool, optional\n        If True, an exception is raised if an inconsistency is detected in the\n        number of columns.\n        If False, a warning is emitted and the offending lines are skipped.\n    max_rows : int,  optional\n        The maximum number of rows to read. Must not be used with skip_footer\n        at the same time.  If given, the value must be at least 1. Default is\n        to read the entire file.\n\n        .. versionadded:: 1.10.0\n    encoding : str, optional\n        Encoding used to decode the inputfile. Does not apply when `fname`\n        is a file object. The special value \'bytes\' enables backward \n        compatibility workarounds that ensure that you receive byte arrays\n        when possible and passes latin1 encoded strings to converters. \n        Override this value to receive unicode arrays and pass strings \n        as input to converters.  If set to None the system default is used.\n        The default value is \'bytes\'.\n\n        .. versionadded:: 1.14.0\n    ndmin : int, optional\n        Same parameter as `loadtxt`\n\n        .. versionadded:: 1.23.0\n    ${ARRAY_FUNCTION_LIKE}\n\n        .. versionadded:: 1.20.0\n\n    Returns\n    -------\n    out : ndarray\n        Data read from the text file. If `usemask` is True, this is a\n        masked array.\n\n    See Also\n    --------\n    numpy.loadtxt : equivalent function when no data is missing.\n\n    Notes\n    -----\n    * When spaces are used as delimiters, or when no delimiter has been given\n      as input, there should not be any missing data between two fields.\n    * When variables are named (either by a flexible dtype or with `names`),\n      there must not be any header in the file (else a ValueError\n      exception is raised).\n    * Individual values are not stripped of spaces by default.\n      When using a custom converter, make sure the function does remove spaces.\n    * Custom converters may receive unexpected values due to dtype\n      discovery. \n\n    References\n    ----------\n    .. [1] NumPy User Guide, section `I/O with NumPy\n           <https://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html>`_.\n\n    Examples\n    --------\n    >>> from io import StringIO\n    >>> import numpy as np\n\n    Comma delimited file with mixed dtype\n\n    >>> s = StringIO(u"1,1.3,abcde")\n    >>> data = np.genfromtxt(s, dtype=[(\'myint\',\'i8\'),(\'myfloat\',\'f8\'),\n    ... (\'mystring\',\'S5\')], delimiter=",")\n    >>> data\n    array((1, 1.3, b\'abcde\'),\n          dtype=[(\'myint\', \'<i8\'), (\'myfloat\', \'<f8\'), (\'mystring\', \'S5\')])\n\n    Using dtype = None\n\n    >>> _ = s.seek(0) # needed for StringIO example only\n    >>> data = np.genfromtxt(s, dtype=None,\n    ... names = [\'myint\',\'myfloat\',\'mystring\'], delimiter=",")\n    >>> data\n    array((1, 1.3, b\'abcde\'),\n          dtype=[(\'myint\', \'<i8\'), (\'myfloat\', \'<f8\'), (\'mystring\', \'S5\')])\n\n    Specifying dtype and names\n\n    >>> _ = s.seek(0)\n    >>> data = np.genfromtxt(s, dtype="i8,f8,S5",\n    ... names=[\'myint\',\'myfloat\',\'mystring\'], delimiter=",")\n    >>> data\n    array((1, 1.3, b\'abcde\'),\n          dtype=[(\'myint\', \'<i8\'), (\'myfloat\', \'<f8\'), (\'mystring\', \'S5\')])\n\n    An example with fixed-width columns\n\n    >>> s = StringIO(u"11.3abcde")\n    >>> data = np.genfromtxt(s, dtype=None, names=[\'intvar\',\'fltvar\',\'strvar\'],\n    ...     delimiter=[1,3,5])\n    >>> data\n    array((1, 1.3, b\'abcde\'),\n          dtype=[(\'intvar\', \'<i8\'), (\'fltvar\', \'<f8\'), (\'strvar\', \'S5\')])\n\n    An example to show comments\n\n    >>> f = StringIO(\'\'\'\n    ... text,# of chars\n    ... hello world,11\n    ... numpy,5\'\'\')\n    >>> np.genfromtxt(f, dtype=\'S12,S12\', delimiter=\',\')\n    array([(b\'text\', b\'\'), (b\'hello world\', b\'11\'), (b\'numpy\', b\'5\')],\n      dtype=[(\'f0\', \'S12\'), (\'f1\', \'S12\')])\n\n    '
    if like is not None:
        return _genfromtxt_with_like(like, fname, dtype=dtype, comments=comments, delimiter=delimiter, skip_header=skip_header, skip_footer=skip_footer, converters=converters, missing_values=missing_values, filling_values=filling_values, usecols=usecols, names=names, excludelist=excludelist, deletechars=deletechars, replace_space=replace_space, autostrip=autostrip, case_sensitive=case_sensitive, defaultfmt=defaultfmt, unpack=unpack, usemask=usemask, loose=loose, invalid_raise=invalid_raise, max_rows=max_rows, encoding=encoding, ndmin=ndmin)
    _ensure_ndmin_ndarray_check_param(ndmin)
    if max_rows is not None:
        if skip_footer:
            raise ValueError("The keywords 'skip_footer' and 'max_rows' can not be specified at the same time.")
        if max_rows < 1:
            raise ValueError("'max_rows' must be at least 1.")
    if usemask:
        from numpy.ma import MaskedArray, make_mask_descr
    user_converters = converters or {}
    if not isinstance(user_converters, dict):
        raise TypeError("The input argument 'converter' should be a valid dictionary (got '%s' instead)" % type(user_converters))
    if encoding == 'bytes':
        encoding = None
        byte_converters = True
    else:
        byte_converters = False
    if isinstance(fname, os.PathLike):
        fname = os.fspath(fname)
    if isinstance(fname, str):
        fid = np.lib._datasource.open(fname, 'rt', encoding=encoding)
        fid_ctx = contextlib.closing(fid)
    else:
        fid = fname
        fid_ctx = contextlib.nullcontext(fid)
    try:
        fhd = iter(fid)
    except TypeError as e:
        raise TypeError(f'fname must be a string, a filehandle, a sequence of strings,\nor an iterator of strings. Got {type(fname)} instead.') from e
    with fid_ctx:
        split_line = LineSplitter(delimiter=delimiter, comments=comments, autostrip=autostrip, encoding=encoding)
        validate_names = NameValidator(excludelist=excludelist, deletechars=deletechars, case_sensitive=case_sensitive, replace_space=replace_space)
        try:
            for i in range(skip_header):
                next(fhd)
            first_values = None
            while not first_values:
                first_line = _decode_line(next(fhd), encoding)
                if names is True and comments is not None:
                    if comments in first_line:
                        first_line = ''.join(first_line.split(comments)[1:])
                first_values = split_line(first_line)
        except StopIteration:
            first_line = ''
            first_values = []
            warnings.warn('genfromtxt: Empty input file: "%s"' % fname, stacklevel=2)
        if names is True:
            fval = first_values[0].strip()
            if comments is not None:
                if fval in comments:
                    del first_values[0]
        if usecols is not None:
            try:
                usecols = [_.strip() for _ in usecols.split(',')]
            except AttributeError:
                try:
                    usecols = list(usecols)
                except TypeError:
                    usecols = [usecols]
        nbcols = len(usecols or first_values)
        if names is True:
            names = validate_names([str(_.strip()) for _ in first_values])
            first_line = ''
        elif _is_string_like(names):
            names = validate_names([_.strip() for _ in names.split(',')])
        elif names:
            names = validate_names(names)
        if dtype is not None:
            dtype = easy_dtype(dtype, defaultfmt=defaultfmt, names=names, excludelist=excludelist, deletechars=deletechars, case_sensitive=case_sensitive, replace_space=replace_space)
        if names is not None:
            names = list(names)
        if usecols:
            for (i, current) in enumerate(usecols):
                if _is_string_like(current):
                    usecols[i] = names.index(current)
                elif current < 0:
                    usecols[i] = current + len(first_values)
            if dtype is not None and len(dtype) > nbcols:
                descr = dtype.descr
                dtype = np.dtype([descr[_] for _ in usecols])
                names = list(dtype.names)
            elif names is not None and len(names) > nbcols:
                names = [names[_] for _ in usecols]
        elif names is not None and dtype is not None:
            names = list(dtype.names)
        user_missing_values = missing_values or ()
        if isinstance(user_missing_values, bytes):
            user_missing_values = user_missing_values.decode('latin1')
        missing_values = [list(['']) for _ in range(nbcols)]
        if isinstance(user_missing_values, dict):
            for (key, val) in user_missing_values.items():
                if _is_string_like(key):
                    try:
                        key = names.index(key)
                    except ValueError:
                        continue
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                if isinstance(val, (list, tuple)):
                    val = [str(_) for _ in val]
                else:
                    val = [str(val)]
                if key is None:
                    for miss in missing_values:
                        miss.extend(val)
                else:
                    missing_values[key].extend(val)
        elif isinstance(user_missing_values, (list, tuple)):
            for (value, entry) in zip(user_missing_values, missing_values):
                value = str(value)
                if value not in entry:
                    entry.append(value)
        elif isinstance(user_missing_values, str):
            user_value = user_missing_values.split(',')
            for entry in missing_values:
                entry.extend(user_value)
        else:
            for entry in missing_values:
                entry.extend([str(user_missing_values)])
        user_filling_values = filling_values
        if user_filling_values is None:
            user_filling_values = []
        filling_values = [None] * nbcols
        if isinstance(user_filling_values, dict):
            for (key, val) in user_filling_values.items():
                if _is_string_like(key):
                    try:
                        key = names.index(key)
                    except ValueError:
                        continue
                if usecols:
                    try:
                        key = usecols.index(key)
                    except ValueError:
                        pass
                filling_values[key] = val
        elif isinstance(user_filling_values, (list, tuple)):
            n = len(user_filling_values)
            if n <= nbcols:
                filling_values[:n] = user_filling_values
            else:
                filling_values = user_filling_values[:nbcols]
        else:
            filling_values = [user_filling_values] * nbcols
        if dtype is None:
            converters = [StringConverter(None, missing_values=miss, default=fill) for (miss, fill) in zip(missing_values, filling_values)]
        else:
            dtype_flat = flatten_dtype(dtype, flatten_base=True)
            if len(dtype_flat) > 1:
                zipit = zip(dtype_flat, missing_values, filling_values)
                converters = [StringConverter(dt, locked=True, missing_values=miss, default=fill) for (dt, miss, fill) in zipit]
            else:
                zipit = zip(missing_values, filling_values)
                converters = [StringConverter(dtype, locked=True, missing_values=miss, default=fill) for (miss, fill) in zipit]
        uc_update = []
        for (j, conv) in user_converters.items():
            if _is_string_like(j):
                try:
                    j = names.index(j)
                    i = j
                except ValueError:
                    continue
            elif usecols:
                try:
                    i = usecols.index(j)
                except ValueError:
                    continue
            else:
                i = j
            if len(first_line):
                testing_value = first_values[j]
            else:
                testing_value = None
            if conv is bytes:
                user_conv = asbytes
            elif byte_converters:

                def tobytes_first(x, conv):
                    if False:
                        return 10
                    if type(x) is bytes:
                        return conv(x)
                    return conv(x.encode('latin1'))
                user_conv = functools.partial(tobytes_first, conv=conv)
            else:
                user_conv = conv
            converters[i].update(user_conv, locked=True, testing_value=testing_value, default=filling_values[i], missing_values=missing_values[i])
            uc_update.append((i, user_conv))
        user_converters.update(uc_update)
        rows = []
        append_to_rows = rows.append
        if usemask:
            masks = []
            append_to_masks = masks.append
        invalid = []
        append_to_invalid = invalid.append
        for (i, line) in enumerate(itertools.chain([first_line], fhd)):
            values = split_line(line)
            nbvalues = len(values)
            if nbvalues == 0:
                continue
            if usecols:
                try:
                    values = [values[_] for _ in usecols]
                except IndexError:
                    append_to_invalid((i + skip_header + 1, nbvalues))
                    continue
            elif nbvalues != nbcols:
                append_to_invalid((i + skip_header + 1, nbvalues))
                continue
            append_to_rows(tuple(values))
            if usemask:
                append_to_masks(tuple([v.strip() in m for (v, m) in zip(values, missing_values)]))
            if len(rows) == max_rows:
                break
    if dtype is None:
        for (i, converter) in enumerate(converters):
            current_column = [itemgetter(i)(_m) for _m in rows]
            try:
                converter.iterupgrade(current_column)
            except ConverterLockError:
                errmsg = 'Converter #%i is locked and cannot be upgraded: ' % i
                current_column = map(itemgetter(i), rows)
                for (j, value) in enumerate(current_column):
                    try:
                        converter.upgrade(value)
                    except (ConverterError, ValueError):
                        errmsg += "(occurred line #%i for value '%s')"
                        errmsg %= (j + 1 + skip_header, value)
                        raise ConverterError(errmsg)
    nbinvalid = len(invalid)
    if nbinvalid > 0:
        nbrows = len(rows) + nbinvalid - skip_footer
        template = '    Line #%%i (got %%i columns instead of %i)' % nbcols
        if skip_footer > 0:
            nbinvalid_skipped = len([_ for _ in invalid if _[0] > nbrows + skip_header])
            invalid = invalid[:nbinvalid - nbinvalid_skipped]
            skip_footer -= nbinvalid_skipped
        errmsg = [template % (i, nb) for (i, nb) in invalid]
        if len(errmsg):
            errmsg.insert(0, 'Some errors were detected !')
            errmsg = '\n'.join(errmsg)
            if invalid_raise:
                raise ValueError(errmsg)
            else:
                warnings.warn(errmsg, ConversionWarning, stacklevel=2)
    if skip_footer > 0:
        rows = rows[:-skip_footer]
        if usemask:
            masks = masks[:-skip_footer]
    if loose:
        rows = list(zip(*[[conv._loose_call(_r) for _r in map(itemgetter(i), rows)] for (i, conv) in enumerate(converters)]))
    else:
        rows = list(zip(*[[conv._strict_call(_r) for _r in map(itemgetter(i), rows)] for (i, conv) in enumerate(converters)]))
    data = rows
    if dtype is None:
        column_types = [conv.type for conv in converters]
        strcolidx = [i for (i, v) in enumerate(column_types) if v == np.str_]
        if byte_converters and strcolidx:
            warnings.warn('Reading unicode strings without specifying the encoding argument is deprecated. Set the encoding, use None for the system default.', np.exceptions.VisibleDeprecationWarning, stacklevel=2)

            def encode_unicode_cols(row_tup):
                if False:
                    while True:
                        i = 10
                row = list(row_tup)
                for i in strcolidx:
                    row[i] = row[i].encode('latin1')
                return tuple(row)
            try:
                data = [encode_unicode_cols(r) for r in data]
            except UnicodeEncodeError:
                pass
            else:
                for i in strcolidx:
                    column_types[i] = np.bytes_
        sized_column_types = column_types[:]
        for (i, col_type) in enumerate(column_types):
            if np.issubdtype(col_type, np.character):
                n_chars = max((len(row[i]) for row in data))
                sized_column_types[i] = (col_type, n_chars)
        if names is None:
            base = {c_type for (c, c_type) in zip(converters, column_types) if c._checked}
            if len(base) == 1:
                (uniform_type,) = base
                (ddtype, mdtype) = (uniform_type, bool)
            else:
                ddtype = [(defaultfmt % i, dt) for (i, dt) in enumerate(sized_column_types)]
                if usemask:
                    mdtype = [(defaultfmt % i, bool) for (i, dt) in enumerate(sized_column_types)]
        else:
            ddtype = list(zip(names, sized_column_types))
            mdtype = list(zip(names, [bool] * len(sized_column_types)))
        output = np.array(data, dtype=ddtype)
        if usemask:
            outputmask = np.array(masks, dtype=mdtype)
    else:
        if names and dtype.names is not None:
            dtype.names = names
        if len(dtype_flat) > 1:
            if 'O' in (_.char for _ in dtype_flat):
                if has_nested_fields(dtype):
                    raise NotImplementedError('Nested fields involving objects are not supported...')
                else:
                    output = np.array(data, dtype=dtype)
            else:
                rows = np.array(data, dtype=[('', _) for _ in dtype_flat])
                output = rows.view(dtype)
            if usemask:
                rowmasks = np.array(masks, dtype=np.dtype([('', bool) for t in dtype_flat]))
                mdtype = make_mask_descr(dtype)
                outputmask = rowmasks.view(mdtype)
        else:
            if user_converters:
                ishomogeneous = True
                descr = []
                for (i, ttype) in enumerate([conv.type for conv in converters]):
                    if i in user_converters:
                        ishomogeneous &= ttype == dtype.type
                        if np.issubdtype(ttype, np.character):
                            ttype = (ttype, max((len(row[i]) for row in data)))
                        descr.append(('', ttype))
                    else:
                        descr.append(('', dtype))
                if not ishomogeneous:
                    if len(descr) > 1:
                        dtype = np.dtype(descr)
                    else:
                        dtype = np.dtype(ttype)
            output = np.array(data, dtype)
            if usemask:
                if dtype.names is not None:
                    mdtype = [(_, bool) for _ in dtype.names]
                else:
                    mdtype = bool
                outputmask = np.array(masks, dtype=mdtype)
    names = output.dtype.names
    if usemask and names:
        for (name, conv) in zip(names, converters):
            missing_values = [conv(_) for _ in conv.missing_values if _ != '']
            for mval in missing_values:
                outputmask[name] |= output[name] == mval
    if usemask:
        output = output.view(MaskedArray)
        output._mask = outputmask
    output = _ensure_ndmin_ndarray(output, ndmin=ndmin)
    if unpack:
        if names is None:
            return output.T
        elif len(names) == 1:
            return output[names[0]]
        else:
            return [output[field] for field in names]
    return output
_genfromtxt_with_like = array_function_dispatch()(genfromtxt)

def recfromtxt(fname, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Load ASCII data from a file and return it in a record array.\n\n    If ``usemask=False`` a standard `recarray` is returned,\n    if ``usemask=True`` a MaskedRecords array is returned.\n\n    .. deprecated:: 2.0\n        Use `numpy.genfromtxt` instead.\n\n    Parameters\n    ----------\n    fname, kwargs : For a description of input parameters, see `genfromtxt`.\n\n    See Also\n    --------\n    numpy.genfromtxt : generic function\n\n    Notes\n    -----\n    By default, `dtype` is None, which means that the data-type of the output\n    array will be determined from the data.\n\n    '
    warnings.warn('`recfromtxt` is deprecated, use `numpy.genfromtxt` instead.(deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    kwargs.setdefault('dtype', None)
    usemask = kwargs.get('usemask', False)
    output = genfromtxt(fname, **kwargs)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output

def recfromcsv(fname, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Load ASCII data stored in a comma-separated file.\n\n    The returned array is a record array (if ``usemask=False``, see\n    `recarray`) or a masked record array (if ``usemask=True``,\n    see `ma.mrecords.MaskedRecords`).\n\n    .. deprecated:: 2.0\n        Use `numpy.genfromtxt` with comma as `delimiter` instead.\n\n    Parameters\n    ----------\n    fname, kwargs : For a description of input parameters, see `genfromtxt`.\n\n    See Also\n    --------\n    numpy.genfromtxt : generic function to load ASCII data.\n\n    Notes\n    -----\n    By default, `dtype` is None, which means that the data-type of the output\n    array will be determined from the data.\n\n    '
    warnings.warn('`recfromcsv` is deprecated, use `numpy.genfromtxt` with comma as `delimiter` instead. (deprecated in NumPy 2.0)', DeprecationWarning, stacklevel=2)
    kwargs.setdefault('case_sensitive', 'lower')
    kwargs.setdefault('names', True)
    kwargs.setdefault('delimiter', ',')
    kwargs.setdefault('dtype', None)
    output = genfromtxt(fname, **kwargs)
    usemask = kwargs.get('usemask', False)
    if usemask:
        from numpy.ma.mrecords import MaskedRecords
        output = output.view(MaskedRecords)
    else:
        output = output.view(np.recarray)
    return output