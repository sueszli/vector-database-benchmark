"""
Record Arrays
=============
Record arrays expose the fields of structured arrays as properties.

Most commonly, ndarrays contain elements of a single type, e.g. floats,
integers, bools etc.  However, it is possible for elements to be combinations
of these using structured types, such as::

  >>> a = np.array([(1, 2.0), (1, 2.0)], 
  ...     dtype=[('x', np.int64), ('y', np.float64)])
  >>> a
  array([(1, 2.), (1, 2.)], dtype=[('x', '<i8'), ('y', '<f8')])

Here, each element consists of two fields: x (and int), and y (a float).
This is known as a structured array.  The different fields are analogous
to columns in a spread-sheet.  The different fields can be accessed as
one would a dictionary::

  >>> a['x']
  array([1, 1])

  >>> a['y']
  array([2., 2.])

Record arrays allow us to access fields as properties::

  >>> ar = np.rec.array(a)

  >>> ar.x
  array([1, 1])

  >>> ar.y
  array([2., 2.])

"""
import os
import warnings
from collections import Counter
from contextlib import nullcontext
from .._utils import set_module
from . import numeric as sb
from . import numerictypes as nt
from .arrayprint import _get_legacy_print_mode
__all__ = ['record', 'recarray', 'format_parser', 'fromarrays', 'fromrecords', 'fromstring', 'fromfile', 'array', 'find_duplicate']
ndarray = sb.ndarray
_byteorderconv = {'b': '>', 'l': '<', 'n': '=', 'B': '>', 'L': '<', 'N': '=', 'S': 's', 's': 's', '>': '>', '<': '<', '=': '=', '|': '|', 'I': '|', 'i': '|'}
numfmt = nt.sctypeDict

@set_module('numpy.rec')
def find_duplicate(list):
    if False:
        print('Hello World!')
    'Find duplication in a list, return a list of duplicated elements'
    return [item for (item, counts) in Counter(list).items() if counts > 1]

@set_module('numpy.rec')
class format_parser:
    """
    Class to convert formats, names, titles description to a dtype.

    After constructing the format_parser object, the dtype attribute is
    the converted data-type:
    ``dtype = format_parser(formats, names, titles).dtype``

    Attributes
    ----------
    dtype : dtype
        The converted data-type.

    Parameters
    ----------
    formats : str or list of str
        The format description, either specified as a string with
        comma-separated format descriptions in the form ``'f8, i4, S5'``, or
        a list of format description strings  in the form
        ``['f8', 'i4', 'S5']``.
    names : str or list/tuple of str
        The field names, either specified as a comma-separated string in the
        form ``'col1, col2, col3'``, or as a list or tuple of strings in the
        form ``['col1', 'col2', 'col3']``.
        An empty list can be used, in that case default field names
        ('f0', 'f1', ...) are used.
    titles : sequence
        Sequence of title strings. An empty list can be used to leave titles
        out.
    aligned : bool, optional
        If True, align the fields by padding as the C-compiler would.
        Default is False.
    byteorder : str, optional
        If specified, all the fields will be changed to the
        provided byte-order.  Otherwise, the default byte-order is
        used. For all available string specifiers, see `dtype.newbyteorder`.

    See Also
    --------
    numpy.dtype, numpy.typename

    Examples
    --------
    >>> np.rec.format_parser(['<f8', '<i4'], ['col1', 'col2'],
    ...                      ['T1', 'T2']).dtype
    dtype([(('T1', 'col1'), '<f8'), (('T2', 'col2'), '<i4')])

    `names` and/or `titles` can be empty lists. If `titles` is an empty list,
    titles will simply not appear. If `names` is empty, default field names
    will be used.

    >>> np.rec.format_parser(['f8', 'i4', 'a5'], ['col1', 'col2', 'col3'],
    ...                      []).dtype
    dtype([('col1', '<f8'), ('col2', '<i4'), ('col3', '<S5')])
    >>> np.rec.format_parser(['<f8', '<i4', '<a5'], [], []).dtype
    dtype([('f0', '<f8'), ('f1', '<i4'), ('f2', 'S5')])

    """

    def __init__(self, formats, names, titles, aligned=False, byteorder=None):
        if False:
            while True:
                i = 10
        self._parseFormats(formats, aligned)
        self._setfieldnames(names, titles)
        self._createdtype(byteorder)

    def _parseFormats(self, formats, aligned=False):
        if False:
            while True:
                i = 10
        ' Parse the field formats '
        if formats is None:
            raise ValueError('Need formats argument')
        if isinstance(formats, list):
            dtype = sb.dtype([('f{}'.format(i), format_) for (i, format_) in enumerate(formats)], aligned)
        else:
            dtype = sb.dtype(formats, aligned)
        fields = dtype.fields
        if fields is None:
            dtype = sb.dtype([('f1', dtype)], aligned)
            fields = dtype.fields
        keys = dtype.names
        self._f_formats = [fields[key][0] for key in keys]
        self._offsets = [fields[key][1] for key in keys]
        self._nfields = len(keys)

    def _setfieldnames(self, names, titles):
        if False:
            i = 10
            return i + 15
        'convert input field names into a list and assign to the _names\n        attribute '
        if names:
            if type(names) in [list, tuple]:
                pass
            elif isinstance(names, str):
                names = names.split(',')
            else:
                raise NameError('illegal input names %s' % repr(names))
            self._names = [n.strip() for n in names[:self._nfields]]
        else:
            self._names = []
        self._names += ['f%d' % i for i in range(len(self._names), self._nfields)]
        _dup = find_duplicate(self._names)
        if _dup:
            raise ValueError('Duplicate field names: %s' % _dup)
        if titles:
            self._titles = [n.strip() for n in titles[:self._nfields]]
        else:
            self._titles = []
            titles = []
        if self._nfields > len(titles):
            self._titles += [None] * (self._nfields - len(titles))

    def _createdtype(self, byteorder):
        if False:
            return 10
        dtype = sb.dtype({'names': self._names, 'formats': self._f_formats, 'offsets': self._offsets, 'titles': self._titles})
        if byteorder is not None:
            byteorder = _byteorderconv[byteorder[0]]
            dtype = dtype.newbyteorder(byteorder)
        self.dtype = dtype

class record(nt.void):
    """A data-type scalar that allows field access as attribute lookup.
    """
    __name__ = 'record'
    __module__ = 'numpy'

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if _get_legacy_print_mode() <= 113:
            return self.__str__()
        return super().__repr__()

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if _get_legacy_print_mode() <= 113:
            return str(self.item())
        return super().__str__()

    def __getattribute__(self, attr):
        if False:
            i = 10
            return i + 15
        if attr in ('setfield', 'getfield', 'dtype'):
            return nt.void.__getattribute__(self, attr)
        try:
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            obj = self.getfield(*res[:2])
            try:
                dt = obj.dtype
            except AttributeError:
                return obj
            if dt.names is not None:
                return obj.view((self.__class__, obj.dtype))
            return obj
        else:
            raise AttributeError("'record' object has no attribute '%s'" % attr)

    def __setattr__(self, attr, val):
        if False:
            i = 10
            return i + 15
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError("Cannot set '%s' attribute" % attr)
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            return self.setfield(val, *res[:2])
        elif getattr(self, attr, None):
            return nt.void.__setattr__(self, attr, val)
        else:
            raise AttributeError("'record' object has no attribute '%s'" % attr)

    def __getitem__(self, indx):
        if False:
            i = 10
            return i + 15
        obj = nt.void.__getitem__(self, indx)
        if isinstance(obj, nt.void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        else:
            return obj

    def pprint(self):
        if False:
            print('Hello World!')
        'Pretty-print all fields.'
        names = self.dtype.names
        maxlen = max((len(name) for name in names))
        fmt = '%% %ds: %%s' % maxlen
        rows = [fmt % (name, getattr(self, name)) for name in names]
        return '\n'.join(rows)

@set_module('numpy.rec')
class recarray(ndarray):
    """Construct an ndarray that allows field access using attributes.

    Arrays may have a data-types containing fields, analogous
    to columns in a spread sheet.  An example is ``[(x, int), (y, float)]``,
    where each entry in the array is a pair of ``(int, float)``.  Normally,
    these attributes are accessed using dictionary lookups such as ``arr['x']``
    and ``arr['y']``.  Record arrays allow the fields to be accessed as members
    of the array, using ``arr.x`` and ``arr.y``.

    Parameters
    ----------
    shape : tuple
        Shape of output array.
    dtype : data-type, optional
        The desired data-type.  By default, the data-type is determined
        from `formats`, `names`, `titles`, `aligned` and `byteorder`.
    formats : list of data-types, optional
        A list containing the data-types for the different columns, e.g.
        ``['i4', 'f8', 'i4']``.  `formats` does *not* support the new
        convention of using types directly, i.e. ``(int, float, int)``.
        Note that `formats` must be a list, not a tuple.
        Given that `formats` is somewhat limited, we recommend specifying
        `dtype` instead.
    names : tuple of str, optional
        The name of each column, e.g. ``('x', 'y', 'z')``.
    buf : buffer, optional
        By default, a new array is created of the given shape and data-type.
        If `buf` is specified and is an object exposing the buffer interface,
        the array will use the memory from the existing buffer.  In this case,
        the `offset` and `strides` keywords are available.

    Other Parameters
    ----------------
    titles : tuple of str, optional
        Aliases for column names.  For example, if `names` were
        ``('x', 'y', 'z')`` and `titles` is
        ``('x_coordinate', 'y_coordinate', 'z_coordinate')``, then
        ``arr['x']`` is equivalent to both ``arr.x`` and ``arr.x_coordinate``.
    byteorder : {'<', '>', '='}, optional
        Byte-order for all fields.
    aligned : bool, optional
        Align the fields in memory as the C-compiler would.
    strides : tuple of ints, optional
        Buffer (`buf`) is interpreted according to these strides (strides
        define how many bytes each array element, row, column, etc.
        occupy in memory).
    offset : int, optional
        Start reading buffer (`buf`) from this offset onwards.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    rec : recarray
        Empty array of the given shape and type.

    See Also
    --------
    _core.records.fromrecords : Construct a record array from data.
    record : fundamental data-type for `recarray`.
    numpy.rec.format_parser : determine data-type from formats, names, titles.

    Notes
    -----
    This constructor can be compared to ``empty``: it creates a new record
    array but does not fill it with data.  To create a record array from data,
    use one of the following methods:

    1. Create a standard ndarray and convert it to a record array,
       using ``arr.view(np.recarray)``
    2. Use the `buf` keyword.
    3. Use `np.rec.fromrecords`.

    Examples
    --------
    Create an array with two fields, ``x`` and ``y``:

    >>> x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', '<f8'), ('y', '<i8')])
    >>> x
    array([(1., 2), (3., 4)], dtype=[('x', '<f8'), ('y', '<i8')])

    >>> x['x']
    array([1., 3.])

    View the array as a record array:

    >>> x = x.view(np.recarray)

    >>> x.x
    array([1., 3.])

    >>> x.y
    array([2, 4])

    Create a new, empty record array:

    >>> np.recarray((2,),
    ... dtype=[('x', int), ('y', float), ('z', int)]) #doctest: +SKIP
    rec.array([(-1073741821, 1.2249118382103472e-301, 24547520),
           (3471280, 1.2134086255804012e-316, 0)],
          dtype=[('x', '<i4'), ('y', '<f8'), ('z', '<i4')])

    """

    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None, formats=None, names=None, titles=None, byteorder=None, aligned=False, order='C'):
        if False:
            print('Hello World!')
        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(formats, names, titles, aligned, byteorder).dtype
        if buf is None:
            self = ndarray.__new__(subtype, shape, (record, descr), order=order)
        else:
            self = ndarray.__new__(subtype, shape, (record, descr), buffer=buf, offset=offset, strides=strides, order=order)
        return self

    def __array_finalize__(self, obj):
        if False:
            while True:
                i = 10
        if self.dtype.type is not record and self.dtype.names is not None:
            self.dtype = self.dtype

    def __getattribute__(self, attr):
        if False:
            print('Hello World!')
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError('recarray has no attribute %s' % attr) from e
        obj = self.getfield(*res)
        if obj.dtype.names is not None:
            if issubclass(obj.dtype.type, nt.void):
                return obj.view(dtype=(self.dtype.type, obj.dtype))
            return obj
        else:
            return obj.view(ndarray)

    def __setattr__(self, attr, val):
        if False:
            i = 10
            return i + 15
        if attr == 'dtype' and issubclass(val.type, nt.void) and (val.names is not None):
            val = sb.dtype((record, val))
        newattr = attr not in self.__dict__
        try:
            ret = object.__setattr__(self, attr, val)
        except Exception:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                raise
        else:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                return ret
            if newattr:
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError('record array has no attribute %s' % attr) from e
        return self.setfield(val, *res)

    def __getitem__(self, indx):
        if False:
            while True:
                i = 10
        obj = super().__getitem__(indx)
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                obj = obj.view(type(self))
                if issubclass(obj.dtype.type, nt.void):
                    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=ndarray)
        else:
            return obj

    def __repr__(self):
        if False:
            print('Hello World!')
        repr_dtype = self.dtype
        if self.dtype.type is record or not issubclass(self.dtype.type, nt.void):
            if repr_dtype.type is record:
                repr_dtype = sb.dtype((nt.void, repr_dtype))
            prefix = 'rec.array('
            fmt = 'rec.array(%s,%sdtype=%s)'
        else:
            prefix = 'array('
            fmt = 'array(%s,%sdtype=%s).view(numpy.recarray)'
        if self.size > 0 or self.shape == (0,):
            lst = sb.array2string(self, separator=', ', prefix=prefix, suffix=',')
        else:
            lst = '[], shape=%s' % (repr(self.shape),)
        lf = '\n' + ' ' * len(prefix)
        if _get_legacy_print_mode() <= 113:
            lf = ' ' + lf
        return fmt % (lst, lf, repr_dtype)

    def field(self, attr, val=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(attr, int):
            names = ndarray.__getattribute__(self, 'dtype').names
            attr = names[attr]
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        res = fielddict[attr][:2]
        if val is None:
            obj = self.getfield(*res)
            if obj.dtype.names is not None:
                return obj
            return obj.view(ndarray)
        else:
            return self.setfield(val, *res)

def _deprecate_shape_0_as_None(shape):
    if False:
        for i in range(10):
            print('nop')
    if shape == 0:
        warnings.warn('Passing `shape=0` to have the shape be inferred is deprecated, and in future will be equivalent to `shape=(0,)`. To infer the shape and suppress this warning, pass `shape=None` instead.', FutureWarning, stacklevel=3)
        return None
    else:
        return shape

@set_module('numpy.rec')
def fromarrays(arrayList, dtype=None, shape=None, formats=None, names=None, titles=None, aligned=False, byteorder=None):
    if False:
        print('Hello World!')
    "Create a record array from a (flat) list of arrays\n\n    Parameters\n    ----------\n    arrayList : list or tuple\n        List of array-like objects (such as lists, tuples,\n        and ndarrays).\n    dtype : data-type, optional\n        valid dtype for all arrays\n    shape : int or tuple of ints, optional\n        Shape of the resulting array. If not provided, inferred from\n        ``arrayList[0]``.\n    formats, names, titles, aligned, byteorder :\n        If `dtype` is ``None``, these arguments are passed to\n        `numpy.format_parser` to construct a dtype. See that function for\n        detailed documentation.\n\n    Returns\n    -------\n    np.recarray\n        Record array consisting of given arrayList columns.\n\n    Examples\n    --------\n    >>> x1=np.array([1,2,3,4])\n    >>> x2=np.array(['a','dd','xyz','12'])\n    >>> x3=np.array([1.1,2,3,4])\n    >>> r = np._core.records.fromarrays([x1,x2,x3],names='a,b,c')\n    >>> print(r[1])\n    (2, 'dd', 2.0) # may vary\n    >>> x1[1]=34\n    >>> r.a\n    array([1, 2, 3, 4])\n\n    >>> x1 = np.array([1, 2, 3, 4])\n    >>> x2 = np.array(['a', 'dd', 'xyz', '12'])\n    >>> x3 = np.array([1.1, 2, 3,4])\n    >>> r = np._core.records.fromarrays(\n    ...     [x1, x2, x3],\n    ...     dtype=np.dtype([('a', np.int32), ('b', 'S3'), ('c', np.float32)]))\n    >>> r\n    rec.array([(1, b'a', 1.1), (2, b'dd', 2. ), (3, b'xyz', 3. ),\n               (4, b'12', 4. )],\n              dtype=[('a', '<i4'), ('b', 'S3'), ('c', '<f4')])\n    "
    arrayList = [sb.asarray(x) for x in arrayList]
    shape = _deprecate_shape_0_as_None(shape)
    if shape is None:
        shape = arrayList[0].shape
    elif isinstance(shape, int):
        shape = (shape,)
    if formats is None and dtype is None:
        formats = [obj.dtype for obj in arrayList]
    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    _names = descr.names
    if len(descr) != len(arrayList):
        raise ValueError('mismatch between the number of fields and the number of arrays')
    d0 = descr[0].shape
    nn = len(d0)
    if nn > 0:
        shape = shape[:-nn]
    _array = recarray(shape, descr)
    for (k, obj) in enumerate(arrayList):
        nn = descr[k].ndim
        testshape = obj.shape[:obj.ndim - nn]
        name = _names[k]
        if testshape != shape:
            raise ValueError(f'array-shape mismatch in array {k} ("{name}")')
        _array[name] = obj
    return _array

@set_module('numpy.rec')
def fromrecords(recList, dtype=None, shape=None, formats=None, names=None, titles=None, aligned=False, byteorder=None):
    if False:
        while True:
            i = 10
    "Create a recarray from a list of records in text form.\n\n    Parameters\n    ----------\n    recList : sequence\n        data in the same field may be heterogeneous - they will be promoted\n        to the highest data type.\n    dtype : data-type, optional\n        valid dtype for all arrays\n    shape : int or tuple of ints, optional\n        shape of each array.\n    formats, names, titles, aligned, byteorder :\n        If `dtype` is ``None``, these arguments are passed to\n        `numpy.format_parser` to construct a dtype. See that function for\n        detailed documentation.\n\n        If both `formats` and `dtype` are None, then this will auto-detect\n        formats. Use list of tuples rather than list of lists for faster\n        processing.\n\n    Returns\n    -------\n    np.recarray\n        record array consisting of given recList rows.\n\n    Examples\n    --------\n    >>> r=np._core.records.fromrecords([(456,'dbe',1.2),(2,'de',1.3)],\n    ... names='col1,col2,col3')\n    >>> print(r[0])\n    (456, 'dbe', 1.2)\n    >>> r.col1\n    array([456,   2])\n    >>> r.col2\n    array(['dbe', 'de'], dtype='<U3')\n    >>> import pickle\n    >>> pickle.loads(pickle.dumps(r))\n    rec.array([(456, 'dbe', 1.2), (  2, 'de', 1.3)],\n              dtype=[('col1', '<i8'), ('col2', '<U3'), ('col3', '<f8')])\n    "
    if formats is None and dtype is None:
        obj = sb.array(recList, dtype=object)
        arrlist = [sb.array(obj[..., i].tolist()) for i in range(obj.shape[-1])]
        return fromarrays(arrlist, formats=formats, shape=shape, names=names, titles=titles, aligned=aligned, byteorder=byteorder)
    if dtype is not None:
        descr = sb.dtype((record, dtype))
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    try:
        retval = sb.array(recList, dtype=descr)
    except (TypeError, ValueError):
        shape = _deprecate_shape_0_as_None(shape)
        if shape is None:
            shape = len(recList)
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) > 1:
            raise ValueError('Can only deal with 1-d array.')
        _array = recarray(shape, descr)
        for k in range(_array.size):
            _array[k] = tuple(recList[k])
        warnings.warn('fromrecords expected a list of tuples, may have received a list of lists instead. In the future that will raise an error', FutureWarning, stacklevel=2)
        return _array
    else:
        if shape is not None and retval.shape != shape:
            retval.shape = shape
    res = retval.view(recarray)
    return res

@set_module('numpy.rec')
def fromstring(datastring, dtype=None, shape=None, offset=0, formats=None, names=None, titles=None, aligned=False, byteorder=None):
    if False:
        for i in range(10):
            print('nop')
    "Create a record array from binary data\n\n    Note that despite the name of this function it does not accept `str`\n    instances.\n\n    Parameters\n    ----------\n    datastring : bytes-like\n        Buffer of binary data\n    dtype : data-type, optional\n        Valid dtype for all arrays\n    shape : int or tuple of ints, optional\n        Shape of each array.\n    offset : int, optional\n        Position in the buffer to start reading from.\n    formats, names, titles, aligned, byteorder :\n        If `dtype` is ``None``, these arguments are passed to\n        `numpy.format_parser` to construct a dtype. See that function for\n        detailed documentation.\n\n\n    Returns\n    -------\n    np.recarray\n        Record array view into the data in datastring. This will be readonly\n        if `datastring` is readonly.\n\n    See Also\n    --------\n    numpy.frombuffer\n\n    Examples\n    --------\n    >>> a = b'\\x01\\x02\\x03abc'\n    >>> np._core.records.fromstring(a, dtype='u1,u1,u1,S3')\n    rec.array([(1, 2, 3, b'abc')],\n            dtype=[('f0', 'u1'), ('f1', 'u1'), ('f2', 'u1'), ('f3', 'S3')])\n\n    >>> grades_dtype = [('Name', (np.str_, 10)), ('Marks', np.float64),\n    ...                 ('GradeLevel', np.int32)]\n    >>> grades_array = np.array([('Sam', 33.3, 3), ('Mike', 44.4, 5),\n    ...                         ('Aadi', 66.6, 6)], dtype=grades_dtype)\n    >>> np._core.records.fromstring(grades_array.tobytes(), dtype=grades_dtype)\n    rec.array([('Sam', 33.3, 3), ('Mike', 44.4, 5), ('Aadi', 66.6, 6)],\n            dtype=[('Name', '<U10'), ('Marks', '<f8'), ('GradeLevel', '<i4')])\n\n    >>> s = '\\x01\\x02\\x03abc'\n    >>> np._core.records.fromstring(s, dtype='u1,u1,u1,S3')\n    Traceback (most recent call last)\n       ...\n    TypeError: a bytes-like object is required, not 'str'\n    "
    if dtype is None and formats is None:
        raise TypeError("fromstring() needs a 'dtype' or 'formats' argument")
    if dtype is not None:
        descr = sb.dtype(dtype)
    else:
        descr = format_parser(formats, names, titles, aligned, byteorder).dtype
    itemsize = descr.itemsize
    shape = _deprecate_shape_0_as_None(shape)
    if shape in (None, -1):
        shape = (len(datastring) - offset) // itemsize
    _array = recarray(shape, descr, buf=datastring, offset=offset)
    return _array

def get_remaining_size(fd):
    if False:
        print('Hello World!')
    pos = fd.tell()
    try:
        fd.seek(0, 2)
        return fd.tell() - pos
    finally:
        fd.seek(pos, 0)

@set_module('numpy.rec')
def fromfile(fd, dtype=None, shape=None, offset=0, formats=None, names=None, titles=None, aligned=False, byteorder=None):
    if False:
        return 10
    "Create an array from binary file data\n\n    Parameters\n    ----------\n    fd : str or file type\n        If file is a string or a path-like object then that file is opened,\n        else it is assumed to be a file object. The file object must\n        support random access (i.e. it must have tell and seek methods).\n    dtype : data-type, optional\n        valid dtype for all arrays\n    shape : int or tuple of ints, optional\n        shape of each array.\n    offset : int, optional\n        Position in the file to start reading from.\n    formats, names, titles, aligned, byteorder :\n        If `dtype` is ``None``, these arguments are passed to\n        `numpy.format_parser` to construct a dtype. See that function for\n        detailed documentation\n\n    Returns\n    -------\n    np.recarray\n        record array consisting of data enclosed in file.\n\n    Examples\n    --------\n    >>> from tempfile import TemporaryFile\n    >>> a = np.empty(10,dtype='f8,i4,a5')\n    >>> a[5] = (0.5,10,'abcde')\n    >>>\n    >>> fd=TemporaryFile()\n    >>> a = a.view(a.dtype.newbyteorder('<'))\n    >>> a.tofile(fd)\n    >>>\n    >>> _ = fd.seek(0)\n    >>> r=np._core.records.fromfile(fd, formats='f8,i4,a5', shape=10,\n    ... byteorder='<')\n    >>> print(r[5])\n    (0.5, 10, 'abcde')\n    >>> r.shape\n    (10,)\n    "
    if dtype is None and formats is None:
        raise TypeError("fromfile() needs a 'dtype' or 'formats' argument")
    shape = _deprecate_shape_0_as_None(shape)
    if shape is None:
        shape = (-1,)
    elif isinstance(shape, int):
        shape = (shape,)
    if hasattr(fd, 'readinto'):
        ctx = nullcontext(fd)
    else:
        ctx = open(os.fspath(fd), 'rb')
    with ctx as fd:
        if offset > 0:
            fd.seek(offset, 1)
        size = get_remaining_size(fd)
        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(formats, names, titles, aligned, byteorder).dtype
        itemsize = descr.itemsize
        shapeprod = sb.array(shape).prod(dtype=nt.intp)
        shapesize = shapeprod * itemsize
        if shapesize < 0:
            shape = list(shape)
            shape[shape.index(-1)] = size // -shapesize
            shape = tuple(shape)
            shapeprod = sb.array(shape).prod(dtype=nt.intp)
        nbytes = shapeprod * itemsize
        if nbytes > size:
            raise ValueError('Not enough bytes left in file for specified shape and type.')
        _array = recarray(shape, descr)
        nbytesread = fd.readinto(_array.data)
        if nbytesread != nbytes:
            raise OSError("Didn't read as many bytes as expected")
    return _array

@set_module('numpy.rec')
def array(obj, dtype=None, shape=None, offset=0, strides=None, formats=None, names=None, titles=None, aligned=False, byteorder=None, copy=True):
    if False:
        return 10
    "\n    Construct a record array from a wide-variety of objects.\n\n    A general-purpose record array constructor that dispatches to the\n    appropriate `recarray` creation function based on the inputs (see Notes).\n\n    Parameters\n    ----------\n    obj : any\n        Input object. See Notes for details on how various input types are\n        treated.\n    dtype : data-type, optional\n        Valid dtype for array.\n    shape : int or tuple of ints, optional\n        Shape of each array.\n    offset : int, optional\n        Position in the file or buffer to start reading from.\n    strides : tuple of ints, optional\n        Buffer (`buf`) is interpreted according to these strides (strides\n        define how many bytes each array element, row, column, etc.\n        occupy in memory).\n    formats, names, titles, aligned, byteorder :\n        If `dtype` is ``None``, these arguments are passed to\n        `numpy.format_parser` to construct a dtype. See that function for\n        detailed documentation.\n    copy : bool, optional\n        Whether to copy the input object (True), or to use a reference instead.\n        This option only applies when the input is an ndarray or recarray.\n        Defaults to True.\n\n    Returns\n    -------\n    np.recarray\n        Record array created from the specified object.\n\n    Notes\n    -----\n    If `obj` is ``None``, then call the `~numpy.recarray` constructor. If\n    `obj` is a string, then call the `fromstring` constructor. If `obj` is a\n    list or a tuple, then if the first object is an `~numpy.ndarray`, call\n    `fromarrays`, otherwise call `fromrecords`. If `obj` is a\n    `~numpy.recarray`, then make a copy of the data in the recarray\n    (if ``copy=True``) and use the new formats, names, and titles. If `obj`\n    is a file, then call `fromfile`. Finally, if obj is an `ndarray`, then\n    return ``obj.view(recarray)``, making a copy of the data if ``copy=True``.\n\n    Examples\n    --------\n    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    array([[1, 2, 3],\n           [4, 5, 6],\n           [7, 8, 9]])\n\n    >>> np.rec.array(a)\n    rec.array([[1, 2, 3],\n               [4, 5, 6],\n               [7, 8, 9]],\n        dtype=int32)\n\n    >>> b = [(1, 1), (2, 4), (3, 9)]\n    >>> c = np.rec.array(b, formats = ['i2', 'f2'], names = ('x', 'y'))\n    >>> c\n    rec.array([(1, 1.0), (2, 4.0), (3, 9.0)],\n              dtype=[('x', '<i2'), ('y', '<f2')])\n\n    >>> c.x\n    rec.array([1, 2, 3], dtype=int16)\n\n    >>> c.y\n    rec.array([ 1.0,  4.0,  9.0], dtype=float16)\n\n    >>> r = np.rec.array(['abc','def'], names=['col1','col2'])\n    >>> print(r.col1)\n    abc\n\n    >>> r.col1\n    array('abc', dtype='<U3')\n\n    >>> r.col2\n    array('def', dtype='<U3')\n    "
    if (isinstance(obj, (type(None), str)) or hasattr(obj, 'readinto')) and formats is None and (dtype is None):
        raise ValueError('Must define formats (or dtype) if object is None, string, or an open file')
    kwds = {}
    if dtype is not None:
        dtype = sb.dtype(dtype)
    elif formats is not None:
        dtype = format_parser(formats, names, titles, aligned, byteorder).dtype
    else:
        kwds = {'formats': formats, 'names': names, 'titles': titles, 'aligned': aligned, 'byteorder': byteorder}
    if obj is None:
        if shape is None:
            raise ValueError('Must define a shape if obj is None')
        return recarray(shape, dtype, buf=obj, offset=offset, strides=strides)
    elif isinstance(obj, bytes):
        return fromstring(obj, dtype, shape=shape, offset=offset, **kwds)
    elif isinstance(obj, (list, tuple)):
        if isinstance(obj[0], (tuple, list)):
            return fromrecords(obj, dtype=dtype, shape=shape, **kwds)
        else:
            return fromarrays(obj, dtype=dtype, shape=shape, **kwds)
    elif isinstance(obj, recarray):
        if dtype is not None and obj.dtype != dtype:
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new
    elif hasattr(obj, 'readinto'):
        return fromfile(obj, dtype=dtype, shape=shape, offset=offset)
    elif isinstance(obj, ndarray):
        if dtype is not None and obj.dtype != dtype:
            new = obj.view(dtype)
        else:
            new = obj
        if copy:
            new = new.copy()
        return new.view(recarray)
    else:
        interface = getattr(obj, '__array_interface__', None)
        if interface is None or not isinstance(interface, dict):
            raise ValueError('Unknown input type')
        obj = sb.array(obj)
        if dtype is not None and obj.dtype != dtype:
            obj = obj.view(dtype)
        return obj.view(recarray)