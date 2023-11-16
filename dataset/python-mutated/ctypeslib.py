"""
============================
``ctypes`` Utility Functions
============================

See Also
--------
load_library : Load a C library.
ndpointer : Array restype/argtype with verification.
as_ctypes : Create a ctypes array from an ndarray.
as_array : Create an ndarray from a ctypes array.

References
----------
.. [1] "SciPy Cookbook: ctypes", https://scipy-cookbook.readthedocs.io/items/Ctypes.html

Examples
--------
Load the C library:

>>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP

Our result type, an ndarray that must be of type double, be 1-dimensional
and is C-contiguous in memory:

>>> array_1d_double = np.ctypeslib.ndpointer(
...                          dtype=np.double,
...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP

Our C-function typically takes an array and updates its values
in-place.  For example::

    void foo_func(double* x, int length)
    {
        int i;
        for (i = 0; i < length; i++) {
            x[i] = i*i;
        }
    }

We wrap it using:

>>> _lib.foo_func.restype = None                      #doctest: +SKIP
>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP

Then, we're ready to call ``foo_func``:

>>> out = np.empty(15, dtype=np.double)
>>> _lib.foo_func(out, len(out))                #doctest: +SKIP

"""
__all__ = ['load_library', 'ndpointer', 'c_intp', 'as_ctypes', 'as_array', 'as_ctypes_type']
import os
from numpy import integer, ndarray, dtype as _dtype, asarray, frombuffer
from numpy._core.multiarray import _flagdict, flagsobj
try:
    import ctypes
except ImportError:
    ctypes = None
if ctypes is None:

    def _dummy(*args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dummy object that raises an ImportError if ctypes is not available.\n\n        Raises\n        ------\n        ImportError\n            If ctypes is not available.\n\n        '
        raise ImportError('ctypes is not available.')
    load_library = _dummy
    as_ctypes = _dummy
    as_array = _dummy
    from numpy import intp as c_intp
    _ndptr_base = object
else:
    import numpy._core._internal as nic
    c_intp = nic._getintp_ctype()
    del nic
    _ndptr_base = ctypes.c_void_p

    def load_library(libname, loader_path):
        if False:
            i = 10
            return i + 15
        "\n        It is possible to load a library using\n\n        >>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP\n\n        But there are cross-platform considerations, such as library file extensions,\n        plus the fact Windows will just load the first library it finds with that name.\n        NumPy supplies the load_library function as a convenience.\n\n        .. versionchanged:: 1.20.0\n            Allow libname and loader_path to take any\n            :term:`python:path-like object`.\n\n        Parameters\n        ----------\n        libname : path-like\n            Name of the library, which can have 'lib' as a prefix,\n            but without an extension.\n        loader_path : path-like\n            Where the library can be found.\n\n        Returns\n        -------\n        ctypes.cdll[libpath] : library object\n           A ctypes library object\n\n        Raises\n        ------\n        OSError\n            If there is no library with the expected extension, or the\n            library is defective and cannot be loaded.\n        "
        libname = os.fsdecode(libname)
        loader_path = os.fsdecode(loader_path)
        ext = os.path.splitext(libname)[1]
        if not ext:
            import sys
            import sysconfig
            base_ext = '.so'
            if sys.platform.startswith('darwin'):
                base_ext = '.dylib'
            elif sys.platform.startswith('win'):
                base_ext = '.dll'
            libname_ext = [libname + base_ext]
            so_ext = sysconfig.get_config_var('EXT_SUFFIX')
            if not so_ext == base_ext:
                libname_ext.insert(0, libname + so_ext)
        else:
            libname_ext = [libname]
        loader_path = os.path.abspath(loader_path)
        if not os.path.isdir(loader_path):
            libdir = os.path.dirname(loader_path)
        else:
            libdir = loader_path
        for ln in libname_ext:
            libpath = os.path.join(libdir, ln)
            if os.path.exists(libpath):
                try:
                    return ctypes.cdll[libpath]
                except OSError:
                    raise
        raise OSError('no file with expected extension')

def _num_fromflags(flaglist):
    if False:
        print('Hello World!')
    num = 0
    for val in flaglist:
        num += _flagdict[val]
    return num
_flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE', 'OWNDATA', 'WRITEBACKIFCOPY']

def _flags_fromnum(num):
    if False:
        while True:
            i = 10
    res = []
    for key in _flagnames:
        value = _flagdict[key]
        if num & value:
            res.append(key)
    return res

class _ndptr(_ndptr_base):

    @classmethod
    def from_param(cls, obj):
        if False:
            return 10
        if not isinstance(obj, ndarray):
            raise TypeError('argument must be an ndarray')
        if cls._dtype_ is not None and obj.dtype != cls._dtype_:
            raise TypeError('array must have data type %s' % cls._dtype_)
        if cls._ndim_ is not None and obj.ndim != cls._ndim_:
            raise TypeError('array must have %d dimension(s)' % cls._ndim_)
        if cls._shape_ is not None and obj.shape != cls._shape_:
            raise TypeError('array must have shape %s' % str(cls._shape_))
        if cls._flags_ is not None and obj.flags.num & cls._flags_ != cls._flags_:
            raise TypeError('array must have flags %s' % _flags_fromnum(cls._flags_))
        return obj.ctypes

class _concrete_ndptr(_ndptr):
    """
    Like _ndptr, but with `_shape_` and `_dtype_` specified.

    Notably, this means the pointer has enough information to reconstruct
    the array, which is not generally true.
    """

    def _check_retval_(self):
        if False:
            print('Hello World!')
        '\n        This method is called when this class is used as the .restype\n        attribute for a shared-library function, to automatically wrap the\n        pointer into an array.\n        '
        return self.contents

    @property
    def contents(self):
        if False:
            print('Hello World!')
        '\n        Get an ndarray viewing the data pointed to by this pointer.\n\n        This mirrors the `contents` attribute of a normal ctypes pointer\n        '
        full_dtype = _dtype((self._dtype_, self._shape_))
        full_ctype = ctypes.c_char * full_dtype.itemsize
        buffer = ctypes.cast(self, ctypes.POINTER(full_ctype)).contents
        return frombuffer(buffer, dtype=full_dtype).squeeze(axis=0)
_pointer_type_cache = {}

def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    if False:
        i = 10
        return i + 15
    "\n    Array-checking restype/argtypes.\n\n    An ndpointer instance is used to describe an ndarray in restypes\n    and argtypes specifications.  This approach is more flexible than\n    using, for example, ``POINTER(c_double)``, since several restrictions\n    can be specified, which are verified upon calling the ctypes function.\n    These include data type, number of dimensions, shape and flags.  If a\n    given array does not satisfy the specified restrictions,\n    a ``TypeError`` is raised.\n\n    Parameters\n    ----------\n    dtype : data-type, optional\n        Array data-type.\n    ndim : int, optional\n        Number of array dimensions.\n    shape : tuple of ints, optional\n        Array shape.\n    flags : str or tuple of str\n        Array flags; may be one or more of:\n\n        - C_CONTIGUOUS / C / CONTIGUOUS\n        - F_CONTIGUOUS / F / FORTRAN\n        - OWNDATA / O\n        - WRITEABLE / W\n        - ALIGNED / A\n        - WRITEBACKIFCOPY / X\n\n    Returns\n    -------\n    klass : ndpointer type object\n        A type object, which is an ``_ndtpr`` instance containing\n        dtype, ndim, shape and flags information.\n\n    Raises\n    ------\n    TypeError\n        If a given array does not satisfy the specified restrictions.\n\n    Examples\n    --------\n    >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,\n    ...                                                  ndim=1,\n    ...                                                  flags='C_CONTIGUOUS')]\n    ... #doctest: +SKIP\n    >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))\n    ... #doctest: +SKIP\n\n    "
    if dtype is not None:
        dtype = _dtype(dtype)
    num = None
    if flags is not None:
        if isinstance(flags, str):
            flags = flags.split(',')
        elif isinstance(flags, (int, integer)):
            num = flags
            flags = _flags_fromnum(num)
        elif isinstance(flags, flagsobj):
            num = flags.num
            flags = _flags_fromnum(num)
        if num is None:
            try:
                flags = [x.strip().upper() for x in flags]
            except Exception as e:
                raise TypeError('invalid flags specification') from e
            num = _num_fromflags(flags)
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)
    cache_key = (dtype, ndim, shape, num)
    try:
        return _pointer_type_cache[cache_key]
    except KeyError:
        pass
    if dtype is None:
        name = 'any'
    elif dtype.names is not None:
        name = str(id(dtype))
    else:
        name = dtype.str
    if ndim is not None:
        name += '_%dd' % ndim
    if shape is not None:
        name += '_' + 'x'.join((str(x) for x in shape))
    if flags is not None:
        name += '_' + '_'.join(flags)
    if dtype is not None and shape is not None:
        base = _concrete_ndptr
    else:
        base = _ndptr
    klass = type('ndpointer_%s' % name, (base,), {'_dtype_': dtype, '_shape_': shape, '_ndim_': ndim, '_flags_': num})
    _pointer_type_cache[cache_key] = klass
    return klass
if ctypes is not None:

    def _ctype_ndarray(element_type, shape):
        if False:
            i = 10
            return i + 15
        ' Create an ndarray of the given element type and shape '
        for dim in shape[::-1]:
            element_type = dim * element_type
            element_type.__module__ = None
        return element_type

    def _get_scalar_type_map():
        if False:
            return 10
        '\n        Return a dictionary mapping native endian scalar dtype to ctypes types\n        '
        ct = ctypes
        simple_types = [ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong, ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong, ct.c_float, ct.c_double, ct.c_bool]
        return {_dtype(ctype): ctype for ctype in simple_types}
    _scalar_type_map = _get_scalar_type_map()

    def _ctype_from_dtype_scalar(dtype):
        if False:
            return 10
        dtype_with_endian = dtype.newbyteorder('S').newbyteorder('S')
        dtype_native = dtype.newbyteorder('=')
        try:
            ctype = _scalar_type_map[dtype_native]
        except KeyError as e:
            raise NotImplementedError('Converting {!r} to a ctypes type'.format(dtype)) from None
        if dtype_with_endian.byteorder == '>':
            ctype = ctype.__ctype_be__
        elif dtype_with_endian.byteorder == '<':
            ctype = ctype.__ctype_le__
        return ctype

    def _ctype_from_dtype_subarray(dtype):
        if False:
            while True:
                i = 10
        (element_dtype, shape) = dtype.subdtype
        ctype = _ctype_from_dtype(element_dtype)
        return _ctype_ndarray(ctype, shape)

    def _ctype_from_dtype_structured(dtype):
        if False:
            return 10
        field_data = []
        for name in dtype.names:
            (field_dtype, offset) = dtype.fields[name][:2]
            field_data.append((offset, name, _ctype_from_dtype(field_dtype)))
        field_data = sorted(field_data, key=lambda f: f[0])
        if len(field_data) > 1 and all((offset == 0 for (offset, name, ctype) in field_data)):
            size = 0
            _fields_ = []
            for (offset, name, ctype) in field_data:
                _fields_.append((name, ctype))
                size = max(size, ctypes.sizeof(ctype))
            if dtype.itemsize != size:
                _fields_.append(('', ctypes.c_char * dtype.itemsize))
            return type('union', (ctypes.Union,), dict(_fields_=_fields_, _pack_=1, __module__=None))
        else:
            last_offset = 0
            _fields_ = []
            for (offset, name, ctype) in field_data:
                padding = offset - last_offset
                if padding < 0:
                    raise NotImplementedError('Overlapping fields')
                if padding > 0:
                    _fields_.append(('', ctypes.c_char * padding))
                _fields_.append((name, ctype))
                last_offset = offset + ctypes.sizeof(ctype)
            padding = dtype.itemsize - last_offset
            if padding > 0:
                _fields_.append(('', ctypes.c_char * padding))
            return type('struct', (ctypes.Structure,), dict(_fields_=_fields_, _pack_=1, __module__=None))

    def _ctype_from_dtype(dtype):
        if False:
            print('Hello World!')
        if dtype.fields is not None:
            return _ctype_from_dtype_structured(dtype)
        elif dtype.subdtype is not None:
            return _ctype_from_dtype_subarray(dtype)
        else:
            return _ctype_from_dtype_scalar(dtype)

    def as_ctypes_type(dtype):
        if False:
            i = 10
            return i + 15
        '\n        Convert a dtype into a ctypes type.\n\n        Parameters\n        ----------\n        dtype : dtype\n            The dtype to convert\n\n        Returns\n        -------\n        ctype\n            A ctype scalar, union, array, or struct\n\n        Raises\n        ------\n        NotImplementedError\n            If the conversion is not possible\n\n        Notes\n        -----\n        This function does not losslessly round-trip in either direction.\n\n        ``np.dtype(as_ctypes_type(dt))`` will:\n\n        - insert padding fields\n        - reorder fields to be sorted by offset\n        - discard field titles\n\n        ``as_ctypes_type(np.dtype(ctype))`` will:\n\n        - discard the class names of `ctypes.Structure`\\ s and\n          `ctypes.Union`\\ s\n        - convert single-element `ctypes.Union`\\ s into single-element\n          `ctypes.Structure`\\ s\n        - insert padding fields\n\n        '
        return _ctype_from_dtype(_dtype(dtype))

    def as_array(obj, shape=None):
        if False:
            return 10
        '\n        Create a numpy array from a ctypes array or POINTER.\n\n        The numpy array shares the memory with the ctypes object.\n\n        The shape parameter must be given if converting from a ctypes POINTER.\n        The shape parameter is ignored if converting from a ctypes array\n        '
        if isinstance(obj, ctypes._Pointer):
            if shape is None:
                raise TypeError('as_array() requires a shape argument when called on a pointer')
            p_arr_type = ctypes.POINTER(_ctype_ndarray(obj._type_, shape))
            obj = ctypes.cast(obj, p_arr_type).contents
        return asarray(obj)

    def as_ctypes(obj):
        if False:
            while True:
                i = 10
        'Create and return a ctypes object from a numpy array.  Actually\n        anything that exposes the __array_interface__ is accepted.'
        ai = obj.__array_interface__
        if ai['strides']:
            raise TypeError('strided arrays not supported')
        if ai['version'] != 3:
            raise TypeError('only __array_interface__ version 3 supported')
        (addr, readonly) = ai['data']
        if readonly:
            raise TypeError('readonly arrays unsupported')
        ctype_scalar = as_ctypes_type(ai['typestr'])
        result_type = _ctype_ndarray(ctype_scalar, ai['shape'])
        result = result_type.from_address(addr)
        result.__keep = obj
        return result