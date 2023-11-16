"""
This module defines the SArray class which provides the
ability to create, access and manipulate a remote scalable array object.

SArray acts similarly to pandas.Series but without indexing.
The data is immutable and homogeneous.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .._connect import main as glconnect
from .._cython.cy_flexible_type import pytype_from_dtype, pytype_from_array_typecode
from .._cython.cy_flexible_type import infer_type_of_list, infer_type_of_sequence
from .._cython.cy_sarray import UnitySArrayProxy
from .._cython.context import debug_trace as cython_context
from ..util import _is_non_string_iterable, _make_internal_url
from ..visualization import Plot, LABEL_DEFAULT
from .image import Image as _Image
from .. import aggregate as _aggregate
from .._deps import numpy, HAS_NUMPY
from .._deps import pandas, HAS_PANDAS
import time
import sys
import array
import collections
import datetime
import numbers
import six
import types
__all__ = ['SArray']
if sys.version_info.major > 2:
    long = int

def _create_sequential_sarray(size, start=0, reverse=False):
    if False:
        for i in range(10):
            print('nop')
    if type(size) is not int:
        raise TypeError('size must be int')
    if type(start) is not int:
        raise TypeError('size must be int')
    if type(reverse) is not bool:
        raise TypeError('reverse must me bool')
    with cython_context():
        return SArray(_proxy=glconnect.get_unity().create_sequential_sarray(size, start, reverse))

def load_sarray(filename):
    if False:
        print('Hello World!')
    "\n    Load an SArray. The filename extension is used to determine the format\n    automatically. This function is particurly useful for SArrays previously\n    saved in binary format. If the SArray is in binary format, ``filename`` is\n    actually a directory, created when the SArray is saved.\n\n    Paramaters\n    ----------\n    filename : string\n        Location of the file to load. Can be a local path or a remote URL.\n\n    Returns\n    -------\n    out : SArray\n\n    See Also\n    --------\n    SArray.save\n\n    Examples\n    --------\n    >>> sa = turicreate.SArray(data=[1,2,3,4,5])\n    >>> sa.save('./my_sarray')\n    >>> sa_loaded = turicreate.load_sarray('./my_sarray')\n    "
    sa = SArray(data=filename)
    return sa

class SArray(object):
    """
    An immutable, homogeneously typed array object backed by persistent storage.

    SArray is scaled to hold data that are much larger than the machine's main
    memory. It fully supports missing values and random access. The
    data backing an SArray is located on the same machine as the Turi
    Server process. Each column in an :py:class:`~turicreate.SFrame` is an
    SArray.

    Parameters
    ----------
    data : list | numpy.ndarray | pandas.Series | string | generator | map | range | filter
        The input data. If this is a list or can generate a list from map,
        filter, and generator, numpy.ndarray, or pandas.Series,
        the data in the list is converted and stored in an SArray.
        Alternatively if this is a string, it is interpreted as a path (or
        url) to a text file. Each line of the text file is loaded as a
        separate row. If ``data`` is a directory where an SArray was previously
        saved, this is loaded as an SArray read directly out of that
        directory.

    dtype : {None, int, float, str, list, array.array, dict, datetime.datetime, turicreate.Image}, optional
        The data type of the SArray. If not specified (None), we attempt to
        infer it from the input. If it is a numpy array or a Pandas series, the
        dtype of the array/series is used. If it is a list, the dtype is
        inferred from the inner list. If it is a URL or path to a text file, we
        default the dtype to str.

    ignore_cast_failure : bool, optional
        If True, ignores casting failures but warns when elements cannot be
        casted into the specified dtype.

    Notes
    -----
    - If ``data`` is pandas.Series, the index will be ignored.
    - The datetime is based on the Boost datetime format (see http://www.boost.org/doc/libs/1_48_0/doc/html/date_time/date_time_io.html
      for details)

    Examples
    --------
    SArray can be constructed in various ways:

    Construct an SArray from list.

    >>> from turicreate import SArray
    >>> sa = SArray(data=[1,2,3,4,5], dtype=int)

    Construct an SArray from numpy.ndarray.

    >>> sa = SArray(data=numpy.asarray([1,2,3,4,5]), dtype=int)
    or:
    >>> sa = SArray(numpy.asarray([1,2,3,4,5]), int)

    Construct an SArray from pandas.Series.

    >>> sa = SArray(data=pd.Series([1,2,3,4,5]), dtype=int)
    or:
    >>> sa = SArray(pd.Series([1,2,3,4,5]), int)

    Construct an SArray from range (xrange for py2):
    .. warning::
        if no step is provided from range, SArray.from_sequence is preferred in terms of performance.

    >>> sa = SArray(data=range(1, 100, 2), dtype=int)
    or:
    >>> sa = SArray(data=range(1, 100, 2))

    Construct an SArray from map:

    >>> sa = SArray(data=map(lambda x : x**2, [1, 2, 3]), dtype=int)
    or:
    >>> sa = SArray(data=map(lambda x : x**2, [1, 2, 3]))

    Construct an SArray from filter:

    >>> sa = SArray(data=filter(lambda x : x > 2, [1, 2, 3]), dtype=int)
    or:
    >>> sa = SArray(data=filter(lambda x : x > 2, [1, 2, 3]))

    Construct an SArray from generator:

    def gen():
        x = 0
        while x < 10:
            yield x
            x += 1

    >>> sa = SArray(data=gen(), dtype=int)
    or:
    >>> sa = SArray(data=gen())

    If the type is not specified, automatic inference is attempted:

    >>> SArray(data=[1,2,3,4,5]).dtype
    int
    >>> SArray(data=[1,2,3,4,5.0]).dtype
    float

    The SArray supports standard datatypes such as: integer, float and string.
    It also supports three higher level datatypes: float arrays, dict
    and list (array of arbitrary types).

    Create an SArray from a list of strings:

    >>> sa = SArray(data=['a','b'])

    Create an SArray from a list of float arrays;

    >>> sa = SArray([[1,2,3], [3,4,5]])

    Create an SArray from a list of lists:

    >>> sa = SArray(data=[['a', 1, {'work': 3}], [2, 2.0]])

    Create an SArray from a list of dictionaries:

    >>> sa = SArray(data=[{'a':1, 'b': 2}, {'b':2, 'c': 1}])

    Create an SArray from a list of datetime objects:

    >>> sa = SArray(data=[datetime.datetime(2011, 10, 20, 9, 30, 10)])

    Construct an SArray from local text file. (Only works for local server).

    >>> sa = SArray('/tmp/a_to_z.txt.gz')

    Construct an SArray from a text file downloaded from a URL.

    >>> sa = SArray('http://s3-us-west-2.amazonaws.com/testdatasets/a_to_z.txt.gz')

    **Numeric Operators**

    SArrays support a large number of vectorized operations on numeric types.
    For instance:

    >>> sa = SArray([1,1,1,1,1])
    >>> sb = SArray([2,2,2,2,2])
    >>> sc = sa + sb
    >>> sc
    dtype: int
    Rows: 5
    [3, 3, 3, 3, 3]
    >>> sc + 2
    dtype: int
    Rows: 5
    [5, 5, 5, 5, 5]

    Operators which are supported include all numeric operators (+,-,*,/), as
    well as comparison operators (>, >=, <, <=), and logical operators (&, | ).

    For instance:

    >>> sa = SArray([1,2,3,4,5])
    >>> (sa >= 2) & (sa <= 4)
    dtype: int
    Rows: 5
    [0, 1, 1, 1, 0]

    The numeric operators (+,-,*,/) also work on array types:

    >>> sa = SArray(data=[[1.0,1.0], [2.0,2.0]])
    >>> sa + 1
    dtype: list
    Rows: 2
    [array('f', [2.0, 2.0]), array('f', [3.0, 3.0])]
    >>> sa + sa
    dtype: list
    Rows: 2
    [array('f', [2.0, 2.0]), array('f', [4.0, 4.0])]

    The addition operator (+) can also be used for string concatenation:

    >>> sa = SArray(data=['a','b'])
    >>> sa + "x"
    dtype: str
    Rows: 2
    ['ax', 'bx']

    This can be useful for performing type interpretation of lists or
    dictionaries stored as strings:

    >>> sa = SArray(data=['a,b','c,d'])
    >>> ("[" + sa + "]").astype(list) # adding brackets make it look like a list
    dtype: list
    Rows: 2
    [['a', 'b'], ['c', 'd']]

    All comparison operations and boolean operators are supported and emit
    binary SArrays.

    >>> sa = SArray([1,2,3,4,5])
    >>> sa >= 2
    dtype: int
    Rows: 3
    [0, 1, 1, 1, 1]
    >>> (sa >= 2) & (sa <= 4)
    dtype: int
    Rows: 3
    [0, 1, 1, 1, 0]


    **Element Access and Slicing**
    SArrays can be accessed by integer keys just like a regular python list.
    Such operations may not be fast on large datasets so looping over an SArray
    should be avoided.

    >>> sa = SArray([1,2,3,4,5])
    >>> sa[0]
    1
    >>> sa[2]
    3
    >>> sa[5]
    IndexError: SFrame index out of range

    Negative indices can be used to access elements from the tail of the array

    >>> sa[-1] # returns the last element
    5
    >>> sa[-2] # returns the second to last element
    4

    The SArray also supports the full range of python slicing operators:

    >>> sa[1000:] # Returns an SArray containing rows 1000 to the end
    >>> sa[:1000] # Returns an SArray containing rows 0 to row 999 inclusive
    >>> sa[0:1000:2] # Returns an SArray containing rows 0 to row 1000 in steps of 2
    >>> sa[-100:] # Returns an SArray containing last 100 rows
    >>> sa[-100:len(sa):2] # Returns an SArray containing last 100 rows in steps of 2

    **Logical Filter**

    An SArray can be filtered using

    >>> array[binary_filter]

    where array and binary_filter are SArrays of the same length. The result is
    a new SArray which contains only elements of 'array' where its matching row
    in the binary_filter is non zero.

    This permits the use of boolean operators that can be used to perform
    logical filtering operations.  For instance:

    >>> sa = SArray([1,2,3,4,5])
    >>> sa[(sa >= 2) & (sa <= 4)]
    dtype: int
    Rows: 3
    [2, 3, 4]

    This can also be used more generally to provide filtering capability which
    is otherwise not expressible with simple boolean functions. For instance:

    >>> sa = SArray([1,2,3,4,5])
    >>> sa[sa.apply(lambda x: math.log(x) <= 1)]
    dtype: int
    Rows: 3
    [1, 2]

    This is equivalent to

    >>> sa.filter(lambda x: math.log(x) <= 1)
    dtype: int
    Rows: 3
    [1, 2]

    **Iteration**

    The SArray is also iterable, but not efficiently since this involves a
    streaming transmission of data from the server to the client. This should
    not be used for large data.

    >>> sa = SArray([1,2,3,4,5])
    >>> [i + 1 for i in sa]
    [2, 3, 4, 5, 6]

    This can be used to convert an SArray to a list:

    >>> sa = SArray([1,2,3,4,5])
    >>> l = list(sa)
    >>> l
    [1, 2, 3, 4, 5]
    """
    __slots__ = ['__proxy__', '_getitem_cache']

    @classmethod
    def _is_iterable_required_to_listify(cls, obj):
        if False:
            i = 10
            return i + 15
        return isinstance(obj, types.GeneratorType) or (sys.version_info.major < 3 and isinstance(obj, six.moves.xrange)) or (sys.version_info.major >= 3 and isinstance(obj, (range, filter, map, collections.abc.KeysView, collections.abc.ValuesView)))

    def __init__(self, data=[], dtype=None, ignore_cast_failure=False, _proxy=None):
        if False:
            return 10
        '\n        __init__(data=list(), dtype=None, ignore_cast_failure=False)\n\n        Construct a new SArray. The source of data includes: list,\n        range, generators, map, filter, numpy.ndarray, pandas.Series, and urls.\n        '
        if dtype is not None and type(dtype) != type:
            raise TypeError("dtype must be a type, e.g. use int rather than 'int'")
        if _proxy:
            self.__proxy__ = _proxy
        elif isinstance(data, SArray):
            if dtype is None:
                self.__proxy__ = data.__proxy__
            else:
                self.__proxy__ = data.astype(dtype).__proxy__
        else:
            self.__proxy__ = UnitySArrayProxy()
            if self._is_iterable_required_to_listify(data):
                data = list(data)
            if dtype is None:
                if HAS_PANDAS and isinstance(data, pandas.Series):
                    dtype = pytype_from_dtype(data.dtype)
                    if dtype == object:
                        dtype = infer_type_of_sequence(data.values)
                elif HAS_NUMPY and isinstance(data, numpy.ndarray):
                    try:
                        from .. import numpy_loader
                        if numpy_loader.numpy_activation_successful():
                            from ..numpy import _fast_numpy_to_sarray
                            ret = _fast_numpy_to_sarray(data)
                            (self.__proxy__, ret.__proxy__) = (ret.__proxy__, self.__proxy__)
                            return
                        else:
                            dtype = infer_type_of_sequence(data)
                    except:
                        pass
                    dtype = pytype_from_dtype(data.dtype)
                    if dtype == object:
                        dtype = infer_type_of_sequence(data)
                    if len(data.shape) == 2:
                        if dtype == float or dtype == int:
                            dtype = array.array
                        else:
                            dtype = list
                    elif len(data.shape) > 2:
                        raise TypeError('Cannot convert Numpy arrays of greater than 2 dimensions')
                elif isinstance(data, str) or (sys.version_info.major < 3 and isinstance(data, unicode)):
                    dtype = str
                elif isinstance(data, array.array):
                    dtype = pytype_from_array_typecode(data.typecode)
                elif isinstance(data, collections.Sequence):
                    dtype = infer_type_of_sequence(data)
                else:
                    dtype = None
            if HAS_PANDAS and isinstance(data, pandas.Series):
                with cython_context():
                    self.__proxy__.load_from_iterable(data.values, dtype, ignore_cast_failure)
            elif isinstance(data, str) or (sys.version_info.major <= 2 and isinstance(data, unicode)):
                internal_url = _make_internal_url(data)
                with cython_context():
                    self.__proxy__.load_autodetect(internal_url, dtype)
            elif HAS_NUMPY and isinstance(data, numpy.ndarray) or isinstance(data, array.array) or isinstance(data, collections.Sequence):
                with cython_context():
                    self.__proxy__.load_from_iterable(data, dtype, ignore_cast_failure)
            else:
                raise TypeError('Unexpected data source. Possible data source types are: list, numpy.ndarray, pandas.Series, and string(url)')

    @classmethod
    def date_range(cls, start_time, end_time, freq):
        if False:
            i = 10
            return i + 15
        '\n        Returns a new SArray that represents a fixed frequency datetime index.\n\n        Parameters\n        ----------\n        start_time : datetime.datetime\n          Left bound for generating dates.\n\n        end_time : datetime.datetime\n          Right bound for generating dates.\n\n        freq : datetime.timedelta\n          Fixed frequency between two consecutive data points.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import datetime as dt\n        >>> start = dt.datetime(2013, 5, 7, 10, 4, 10)\n        >>> end = dt.datetime(2013, 5, 10, 10, 4, 10)\n        >>> sa = tc.SArray.date_range(start,end,dt.timedelta(1))\n        >>> print sa\n        dtype: datetime\n        Rows: 4\n        [datetime.datetime(2013, 5, 7, 10, 4, 10),\n         datetime.datetime(2013, 5, 8, 10, 4, 10),\n         datetime.datetime(2013, 5, 9, 10, 4, 10),\n         datetime.datetime(2013, 5, 10, 10, 4, 10)]\n       '
        if not isinstance(start_time, datetime.datetime):
            raise TypeError('The ``start_time`` argument must be from type datetime.datetime.')
        if not isinstance(end_time, datetime.datetime):
            raise TypeError('The ``end_time`` argument must be from type datetime.datetime.')
        if not isinstance(freq, datetime.timedelta):
            raise TypeError('The ``freq`` argument must be from type datetime.timedelta.')
        from .. import extensions
        return extensions.date_range(start_time, end_time, freq.total_seconds())

    @classmethod
    def from_const(cls, value, size, dtype=type(None)):
        if False:
            return 10
        '\n        Constructs an SArray of size with a const value.\n\n        Parameters\n        ----------\n        value : [int | float | str | array.array | list | dict | datetime]\n          The value to fill the SArray\n        size : int\n          The size of the SArray\n        dtype : type\n          The type of the SArray. If not specified, is automatically detected\n          from the value. This should be specified if value=None since the\n          actual type of the SArray can be anything.\n\n        Examples\n        --------\n        Construct an SArray consisting of 10 zeroes:\n\n        >>> turicreate.SArray.from_const(0, 10)\n\n        Construct an SArray consisting of 10 missing string values:\n\n        >>> turicreate.SArray.from_const(None, 10, str)\n        '
        assert isinstance(size, (int, long)) and size >= 0, 'size must be a positive int'
        if not isinstance(value, (type(None), int, float, str, array.array, list, dict, datetime.datetime)):
            raise TypeError('Cannot create sarray of value type %s' % str(type(value)))
        proxy = UnitySArrayProxy()
        proxy.load_from_const(value, size, dtype)
        return cls(_proxy=proxy)

    @classmethod
    def from_sequence(cls, *args):
        if False:
            while True:
                i = 10
        '\n        from_sequence(start=0, stop)\n\n        Create an SArray from sequence\n\n        .. sourcecode:: python\n\n            Construct an SArray of integer values from 0 to 999\n\n            >>> tc.SArray.from_sequence(1000)\n\n            This is equivalent, but more efficient than:\n\n            >>> tc.SArray(range(1000))\n\n            Construct an SArray of integer values from 10 to 999\n\n            >>> tc.SArray.from_sequence(10, 1000)\n\n            This is equivalent, but more efficient than:\n\n            >>> tc.SArray(range(10, 1000))\n\n        Parameters\n        ----------\n        start : int, optional\n            The start of the sequence. The sequence will contain this value.\n\n        stop : int\n          The end of the sequence. The sequence will not contain this value.\n\n        '
        start = None
        stop = None
        if len(args) == 1:
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
        if stop is None and start is None:
            raise TypeError('from_sequence expects at least 1 argument. got 0')
        elif start is None:
            return _create_sequential_sarray(stop)
        else:
            size = stop - start
            if size < 0:
                size = 0
            return _create_sequential_sarray(size, start)

    @classmethod
    def read_json(cls, filename):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct an SArray from a json file or glob of json files.\n        The json file must contain a list. Every element in the list\n        must also have the same type. The returned SArray type will be\n        inferred from the elements type.\n\n        Parameters\n        ----------\n        filename : str\n          The filename or glob to load into an SArray.\n\n        Examples\n        --------\n        Construct an SArray from a local JSON file named 'data.json':\n\n        >>> turicreate.SArray.read_json('/data/data.json')\n\n        Construct an SArray from all JSON files /data/data*.json\n\n        >>> turicreate.SArray.read_json('/data/data*.json')\n\n        "
        proxy = UnitySArrayProxy()
        proxy.load_from_json_record_files(_make_internal_url(filename))
        return cls(_proxy=proxy)

    @classmethod
    def where(cls, condition, istrue, isfalse, dtype=None):
        if False:
            print('Hello World!')
        '\n        Selects elements from either istrue or isfalse depending on the value\n        of the condition SArray.\n\n        Parameters\n        ----------\n        condition : SArray\n        An SArray of values such that for each value, if non-zero, yields a\n        value from istrue, otherwise from isfalse.\n\n        istrue : SArray or constant\n        The elements selected if condition is true. If istrue is an SArray,\n        this must be of the same length as condition.\n\n        isfalse : SArray or constant\n        The elements selected if condition is false. If istrue is an SArray,\n        this must be of the same length as condition.\n\n        dtype : type\n        The type of result SArray. This is required if both istrue and isfalse\n        are constants of ambiguous types.\n\n        Examples\n        --------\n\n        Returns an SArray with the same values as g with values above 10\n        clipped to 10\n\n        >>> g = SArray([6,7,8,9,10,11,12,13])\n        >>> SArray.where(g > 10, 10, g)\n        dtype: int\n        Rows: 8\n        [6, 7, 8, 9, 10, 10, 10, 10]\n\n        Returns an SArray with the same values as g with values below 10\n        clipped to 10\n\n        >>> SArray.where(g > 10, g, 10)\n        dtype: int\n        Rows: 8\n        [10, 10, 10, 10, 10, 11, 12, 13]\n\n        Returns an SArray with the same values of g with all values == 1\n        replaced by None\n\n        >>> g = SArray([1,2,3,4,1,2,3,4])\n        >>> SArray.where(g == 1, None, g)\n        dtype: int\n        Rows: 8\n        [None, 2, 3, 4, None, 2, 3, 4]\n\n        Returns an SArray with the same values of g, but with each missing value\n        replaced by its corresponding element in replace_none\n\n        >>> g = SArray([1,2,None,None])\n        >>> replace_none = SArray([3,3,2,2])\n        >>> SArray.where(g != None, g, replace_none)\n        dtype: int\n        Rows: 4\n        [1, 2, 2, 2]\n        '
        true_is_sarray = isinstance(istrue, SArray)
        false_is_sarray = isinstance(isfalse, SArray)
        if not true_is_sarray and false_is_sarray:
            istrue = cls(_proxy=condition.__proxy__.to_const(istrue, isfalse.dtype))
        if true_is_sarray and (not false_is_sarray):
            isfalse = cls(_proxy=condition.__proxy__.to_const(isfalse, istrue.dtype))
        if not true_is_sarray and (not false_is_sarray):
            if dtype is None:
                if istrue is None:
                    dtype = type(isfalse)
                elif isfalse is None:
                    dtype = type(istrue)
                elif type(istrue) != type(isfalse):
                    raise TypeError('true and false inputs are of different types')
                elif type(istrue) == type(isfalse):
                    dtype = type(istrue)
            if dtype is None:
                raise TypeError('Both true and false are None. Resultant type cannot be inferred.')
            istrue = cls(_proxy=condition.__proxy__.to_const(istrue, dtype))
            isfalse = cls(_proxy=condition.__proxy__.to_const(isfalse, dtype))
        return cls(_proxy=condition.__proxy__.ternary_operator(istrue.__proxy__, isfalse.__proxy__))

    def to_numpy(self):
        if False:
            i = 10
            return i + 15
        '\n        Converts this SArray to a numpy array\n\n        This operation will construct a numpy array in memory. Care must\n        be taken when size of the returned object is big.\n\n        Returns\n        -------\n        out : numpy.ndarray\n            A Numpy Array containing all the values of the SArray\n\n        '
        assert HAS_NUMPY, 'numpy is not installed.'
        import numpy
        return numpy.asarray(self)

    def __get_content_identifier__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the unique identifier of the content that backs the SArray\n\n        Notes\n        -----\n        Meant for internal use only.\n        '
        with cython_context():
            return self.__proxy__.get_content_identifier()

    def save(self, filename, format=None):
        if False:
            i = 10
            return i + 15
        "\n        Saves the SArray to file.\n\n        The saved SArray will be in a directory named with the `targetfile`\n        parameter.\n\n        Parameters\n        ----------\n        filename : string\n            A local path or a remote URL.  If format is 'text', it will be\n            saved as a text file. If format is 'binary', a directory will be\n            created at the location which will contain the SArray.\n\n        format : {'binary', 'text', 'csv'}, optional\n            Format in which to save the SFrame. Binary saved SArrays can be\n            loaded much faster and without any format conversion losses.\n            'text' and 'csv' are synonymous: Each SArray row will be written\n            as a single line in an output text file. If not\n            given, will try to infer the format from filename given. If file\n            name ends with 'csv', 'txt' or '.csv.gz', then save as 'csv' format,\n            otherwise save as 'binary' format.\n        "
        from .sframe import SFrame as _SFrame
        if format is None:
            if filename.endswith(('.csv', '.csv.gz', 'txt')):
                format = 'text'
            else:
                format = 'binary'
        if format == 'binary':
            with cython_context():
                self.__proxy__.save(_make_internal_url(filename))
        elif format == 'text' or format == 'csv':
            sf = _SFrame({'X1': self})
            with cython_context():
                sf.__proxy__.save_as_csv(_make_internal_url(filename), {'header': False})
        else:
            raise ValueError('Unsupported format: {}'.format(format))

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Returns a string description of the SArray.\n        '
        data_str = self.__str__()
        ret = 'dtype: ' + str(self.dtype.__name__) + '\n'
        if self.__has_size__():
            ret = ret + 'Rows: ' + str(len(self)) + '\n'
        else:
            ret = ret + 'Rows: ?\n'
        ret = ret + data_str
        return ret

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Returns a string containing the first 100 elements of the array.\n        '
        if self.dtype == _Image:
            headln = str(list(self.astype(str).head(100)))
        elif sys.version_info.major < 3:
            headln = str(list(self.head(100)))
            headln = unicode(headln.decode('string_escape'), 'utf-8', errors='replace').encode('utf-8')
        else:
            headln = str(list(self.head(100)))
        if self.__proxy__.has_size() is False or len(self) > 100:
            headln = headln[0:-1] + ', ... ]'
        return headln

    def __nonzero__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Raises a ValueError exception.\n        The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().\n        '
        raise ValueError('The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')

    def __bool__(self):
        if False:
            return 10
        '\n        Raises a ValueError exception.\n        The truth value of an array with more than one element is ambiguous. Use a.any() or a.all().\n        '
        raise ValueError('The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()')

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the length of the array\n        '
        return self.__proxy__.size()

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Provides an iterator to the contents of the array.\n        '

        def generator():
            if False:
                i = 10
                return i + 15
            elems_at_a_time = 262144
            self.__proxy__.begin_iterator()
            ret = self.__proxy__.iterator_get_next(elems_at_a_time)
            while True:
                for j in ret:
                    yield j
                if len(ret) == elems_at_a_time:
                    ret = self.__proxy__.iterator_get_next(elems_at_a_time)
                else:
                    break
        return generator()

    def __contains__(self, item):
        if False:
            print('Hello World!')
        '\n        Returns true if any element in this SArray is identically equal to item.\n\n        Following are equivalent:\n\n        >>> element in sa\n        >>> sa.__contains__(element)\n\n        For an element-wise contains see ``SArray.contains``\n\n        '
        return (self == item).any()

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        '\n        The shape of the SArray, in a tuple. The first entry is the number of\n        rows.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1,2,3])\n        >>> sa.shape\n        (3,)\n        '
        return (len(self),)

    def contains(self, item):
        if False:
            i = 10
            return i + 15
        '\n        Performs an element-wise search of "item" in the SArray.\n\n        Conceptually equivalent to:\n\n        >>> sa.apply(lambda x: item in x)\n\n        If the current SArray contains strings and item is a string. Produces a 1\n        for each row if \'item\' is a substring of the row and 0 otherwise.\n\n        If the current SArray contains list or arrays, this produces a 1\n        for each row if \'item\' is an element of the list or array.\n\n        If the current SArray contains dictionaries, this produces a 1\n        for each row if \'item\' is a key in the dictionary.\n\n        Parameters\n        ----------\n        item : any type\n            The item to search for.\n\n        Returns\n        -------\n        out : SArray\n            A binary SArray where a non-zero value denotes that the item\n            was found in the row. And 0 if it is not found.\n\n        Examples\n        --------\n        >>> SArray([\'abc\',\'def\',\'ghi\']).contains(\'a\')\n        dtype: int\n        Rows: 3\n        [1, 0, 0]\n        >>> SArray([[\'a\',\'b\'],[\'b\',\'c\'],[\'c\',\'d\']]).contains(\'b\')\n        dtype: int\n        Rows: 3\n        [1, 1, 0]\n        >>> SArray([{\'a\':1},{\'a\':2,\'b\':1}, {\'c\':1}]).contains(\'a\')\n        dtype: int\n        Rows: 3\n        [1, 1, 0]\n\n        See Also\n        --------\n        is_in\n        '
        return SArray(_proxy=self.__proxy__.left_scalar_operator(item, 'in'))

    def is_in(self, other):
        if False:
            for i in range(10):
                print('nop')
        "\n        Performs an element-wise search for each row in 'other'.\n\n        Conceptually equivalent to:\n\n        >>> sa.apply(lambda x: x in other)\n\n        If the current SArray contains strings and other is a string. Produces a 1\n        for each row if the row is a substring of 'other', and 0 otherwise.\n\n        If the 'other' is a list or array, this produces a 1\n        for each row if the row is an element of 'other'\n\n        Parameters\n        ----------\n        other : list, array.array, str\n            The variable to search in.\n\n        Returns\n        -------\n        out : SArray\n            A binary SArray where a non-zero value denotes that row was\n            was found in 'other'. And 0 if it is not found.\n\n        Examples\n        --------\n        >>> SArray(['ab','bc','cd']).is_in('abc')\n        dtype: int\n        Rows: 3\n        [1, 1, 0]\n        >>> SArray(['a','b','c']).is_in(['a','b'])\n        dtype: int\n        Rows: 3\n        [1, 1, 0]\n\n        See Also\n        --------\n        contains\n        "
        return SArray(_proxy=self.__proxy__.right_scalar_operator(other, 'in'))

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        If other is a scalar value, adds it to the current array, returning\n        the new result. If other is an SArray, performs an element-wise\n        addition of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '+'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '+'))

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        '\n        If other is a scalar value, subtracts it from the current array, returning\n        the new result. If other is an SArray, performs an element-wise\n        subtraction of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '-'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '-'))

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        If other is a scalar value, multiplies it to the current array, returning\n        the new result. If other is an SArray, performs an element-wise\n        multiplication of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '*'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '*'))

    def __div__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        If other is a scalar value, divides each element of the current array\n        by the value, returning the result. If other is an SArray, performs\n        an element-wise division of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '/'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '/'))

    def __truediv__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        If other is a scalar value, divides each element of the current array\n        by the value, returning the result. If other is an SArray, performs\n        an element-wise division of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '/'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '/'))

    def __floordiv__(self, other):
        if False:
            print('Hello World!')
        '\n        If other is a scalar value, divides each element of the current array\n        by the value, returning floor of the result. If other is an SArray, performs\n        an element-wise division of the two arrays returning the floor of the result.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '//'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '//'))

    def __pow__(self, other):
        if False:
            print('Hello World!')
        '\n        If other is a scalar value, raises each element of the current array to\n        the power of that value, returning floor of the result. If other\n        is an SArray, performs an element-wise power of the two\n        arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '**'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '**'))

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the negative of each element.\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(0, '-'))

    def __pos__(self):
        if False:
            print('Hello World!')
        if self.dtype not in [int, long, float, array.array]:
            raise RuntimeError('Runtime Exception. Unsupported type operation. cannot perform operation + on type %s' % str(self.dtype))
        with cython_context():
            return SArray(_proxy=self.__proxy__)

    def __abs__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the absolute value of each element.\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.left_scalar_operator(0, 'left_abs'))

    def __mod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Other must be a scalar value. Performs an element wise division remainder.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '%'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '%'))

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        If other is a scalar value, compares each element of the current array\n        by the value, returning the result. If other is an SArray, performs\n        an element-wise comparison of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '<'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '<'))

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        If other is a scalar value, compares each element of the current array\n        by the value, returning the result. If other is an SArray, performs\n        an element-wise comparison of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '>'))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '>'))

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        If other is a scalar value, compares each element of the current array\n        by the value, returning the result. If other is an SArray, performs\n        an element-wise comparison of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '<='))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '<='))

    def __ge__(self, other):
        if False:
            print('Hello World!')
        '\n        If other is a scalar value, compares each element of the current array\n        by the value, returning the result. If other is an SArray, performs\n        an element-wise comparison of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '>='))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '>='))

    def __radd__(self, other):
        if False:
            print('Hello World!')
        '\n        Adds a scalar value to the current array.\n        Returned array has the same type as the array on the right hand side\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '+'))

    def __rsub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Subtracts a scalar value from the current array.\n        Returned array has the same type as the array on the right hand side\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '-'))

    def __rmul__(self, other):
        if False:
            while True:
                i = 10
        '\n        Multiplies a scalar value to the current array.\n        Returned array has the same type as the array on the right hand side\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '*'))

    def __rdiv__(self, other):
        if False:
            while True:
                i = 10
        '\n        Divides a scalar value by each element in the array\n        Returned array has the same type as the array on the right hand side\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '/'))

    def __rtruediv__(self, other):
        if False:
            return 10
        '\n        Divides a scalar value by each element in the array\n        Returned array has the same type as the array on the right hand side\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '/'))

    def __rfloordiv__(self, other):
        if False:
            while True:
                i = 10
        '\n        Divides a scalar value by each element in the array returning the\n        floored result.  Returned array has the same type as the array on the\n        right hand side\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '/')).astype(int)

    def __rmod__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Divides a scalar value by each element in the array returning the\n        floored result.  Returned array has the same type as the array on the\n        right hand side\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '%'))

    def __rpow__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Raises each element of the current array to the power of that\n        value, returning floor of the result.\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.right_scalar_operator(other, '**'))

    def __eq__(self, other):
        if False:
            return 10
        '\n        If other is a scalar value, compares each element of the current array\n        by the value, returning the new result. If other is an SArray, performs\n        an element-wise comparison of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '=='))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '=='))

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        If other is a scalar value, compares each element of the current array\n        by the value, returning the new result. If other is an SArray, performs\n        an element-wise comparison of the two arrays.\n        '
        with cython_context():
            if type(other) is SArray:
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '!='))
            else:
                return SArray(_proxy=self.__proxy__.left_scalar_operator(other, '!='))

    def __and__(self, other):
        if False:
            while True:
                i = 10
        "\n        Perform a logical element-wise 'and' against another SArray.\n        "
        if type(other) is SArray:
            with cython_context():
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '&'))
        else:
            raise TypeError('SArray can only perform logical and against another SArray')

    def __or__(self, other):
        if False:
            while True:
                i = 10
        "\n        Perform a logical element-wise 'or' against another SArray.\n        "
        if type(other) is SArray:
            with cython_context():
                return SArray(_proxy=self.__proxy__.vector_operator(other.__proxy__, '|'))
        else:
            raise TypeError('SArray can only perform logical or against another SArray')

    def __has_size__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns whether or not the size of the SArray is known.\n        '
        return self.__proxy__.has_size()

    def __getitem__(self, other):
        if False:
            while True:
                i = 10
        '\n        If the key is an SArray of identical length, this function performs a\n        logical filter: i.e. it subselects all the elements in this array\n        where the corresponding value in the other array evaluates to true.\n        If the key is an integer this returns a single row of\n        the SArray. If the key is a slice, this returns an SArray with the\n        sliced rows. See the Turi Create User Guide for usage examples.\n        '
        if isinstance(other, numbers.Integral):
            sa_len = len(self)
            if other < 0:
                other += sa_len
            if other >= sa_len:
                raise IndexError('SArray index out of range')
            try:
                (lb, ub, value_list) = self._getitem_cache
                if lb <= other < ub:
                    return value_list[other - lb]
            except AttributeError:
                pass
            block_size = 1024 * (32 if self.dtype in [int, long, float] else 4)
            if self.dtype in [numpy.ndarray, _Image, dict, list]:
                block_size = 16
            block_num = int(other // block_size)
            lb = block_num * block_size
            ub = min(sa_len, lb + block_size)
            val_list = list(SArray(_proxy=self.__proxy__.copy_range(lb, 1, ub)))
            self._getitem_cache = (lb, ub, val_list)
            return val_list[other - lb]
        elif type(other) is SArray:
            if self.__has_size__() and other.__has_size__() and (len(other) != len(self)):
                raise IndexError('Cannot perform logical indexing on arrays of different length.')
            with cython_context():
                return SArray(_proxy=self.__proxy__.logical_filter(other.__proxy__))
        elif type(other) is slice:
            sa_len = len(self)
            start = other.start
            stop = other.stop
            step = other.step
            if start is None:
                start = 0
            if stop is None:
                stop = sa_len
            if step is None:
                step = 1
            if start < 0:
                start = sa_len + start
            if stop < 0:
                stop = sa_len + stop
            return SArray(_proxy=self.__proxy__.copy_range(start, step, stop))
        else:
            raise IndexError('Invalid type to use for indexing')

    def materialize(self):
        if False:
            while True:
                i = 10
        '\n        For a SArray that is lazily evaluated, force persist this sarray\n        to disk, committing all lazy evaluated operations.\n        '
        with cython_context():
            self.__proxy__.materialize()

    def is_materialized(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns whether or not the sarray has been materialized.\n        '
        return self.__is_materialized__()

    def __is_materialized__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns whether or not the sarray has been materialized.\n        '
        return self.__proxy__.is_materialized()

    @property
    def dtype(self):
        if False:
            return 10
        '\n        The data type of the SArray.\n\n        Returns\n        -------\n        out : type\n            The type of the SArray.\n\n        Examples\n        --------\n        >>> sa = tc.SArray(["The quick brown fox jumps over the lazy dog."])\n        >>> sa.dtype\n        str\n        >>> sa = tc.SArray(range(10))\n        >>> sa.dtype\n        int\n        '
        return self.__proxy__.dtype()

    def head(self, n=10):
        if False:
            i = 10
            return i + 15
        '\n        Returns an SArray which contains the first n rows of this SArray.\n\n        Parameters\n        ----------\n        n : int\n            The number of rows to fetch.\n\n        Returns\n        -------\n        out : SArray\n            A new SArray which contains the first n rows of the current SArray.\n\n        Examples\n        --------\n        >>> tc.SArray(range(10)).head(5)\n        dtype: int\n        Rows: 5\n        [0, 1, 2, 3, 4]\n        '
        return SArray(_proxy=self.__proxy__.head(n))

    def vector_slice(self, start, end=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        If this SArray contains vectors or lists, this returns a new SArray\n        containing each individual element sliced, between start and\n        end (exclusive).\n\n        Parameters\n        ----------\n        start : int\n            The start position of the slice.\n\n        end : int, optional.\n            The end position of the slice. Note that the end position\n            is NOT included in the slice. Thus a g.vector_slice(1,3) will extract\n            entries in position 1 and 2. If end is not specified, the return\n            array will contain only one element, the element at the start\n            position.\n\n        Returns\n        -------\n        out : SArray\n            Each individual vector sliced according to the arguments.\n\n        Examples\n        --------\n\n        If g is a vector of floats:\n\n        >>> g = SArray([[1,2,3],[2,3,4]])\n        >>> g\n        dtype: array\n        Rows: 2\n        [array('d', [1.0, 2.0, 3.0]), array('d', [2.0, 3.0, 4.0])]\n\n        >>> g.vector_slice(0) # extracts the first element of each vector\n        dtype: float\n        Rows: 2\n        [1.0, 2.0]\n\n        >>> g.vector_slice(0, 2) # extracts the first two elements of each vector\n        dtype: array.array\n        Rows: 2\n        [array('d', [1.0, 2.0]), array('d', [2.0, 3.0])]\n\n        If a vector cannot be sliced, the result will be None:\n\n        >>> g = SArray([[1],[1,2],[1,2,3]])\n        >>> g\n        dtype: array.array\n        Rows: 3\n        [array('d', [1.0]), array('d', [1.0, 2.0]), array('d', [1.0, 2.0, 3.0])]\n\n        >>> g.vector_slice(2)\n        dtype: float\n        Rows: 3\n        [None, None, 3.0]\n\n        >>> g.vector_slice(0,2)\n        dtype: list\n        Rows: 3\n        [None, array('d', [1.0, 2.0]), array('d', [1.0, 2.0])]\n\n        If g is a vector of mixed types (float, int, str, array, list, etc.):\n\n        >>> g = SArray([['a',1,1.0],['b',2,2.0]])\n        >>> g\n        dtype: list\n        Rows: 2\n        [['a', 1, 1.0], ['b', 2, 2.0]]\n\n        >>> g.vector_slice(0) # extracts the first element of each vector\n        dtype: list\n        Rows: 2\n        [['a'], ['b']]\n        "
        if self.dtype != array.array and self.dtype != list:
            raise RuntimeError('Only Vector type can be sliced')
        if end is None:
            end = start + 1
        with cython_context():
            return SArray(_proxy=self.__proxy__.vector_slice(start, end))

    def element_slice(self, start=None, stop=None, step=None):
        if False:
            while True:
                i = 10
        '\n        This returns an SArray with each element sliced accordingly to the\n        slice specified. This is conceptually equivalent to:\n\n        >>> g.apply(lambda x: x[start:step:stop])\n\n        The SArray must be of type list, vector, or string.\n\n        For instance:\n\n        >>> g = SArray(["abcdef","qwerty"])\n        >>> g.element_slice(start=0, stop=2)\n        dtype: str\n        Rows: 2\n        ["ab", "qw"]\n        >>> g.element_slice(3,-1)\n        dtype: str\n        Rows: 2\n        ["de", "rt"]\n        >>> g.element_slice(3)\n        dtype: str\n        Rows: 2\n        ["def", "rty"]\n\n        >>> g = SArray([[1,2,3], [4,5,6]])\n        >>> g.element_slice(0, 1)\n        dtype: str\n        Rows: 2\n        [[1], [4]]\n\n        Parameters\n        ----------\n        start : int or None (default)\n            The start position of the slice\n\n        stop: int or None (default)\n            The stop position of the slice\n\n        step: int or None (default)\n            The step size of the slice\n\n        Returns\n        -------\n        out : SArray\n            Each individual vector/string/list sliced according to the arguments.\n\n        '
        if self.dtype not in [str, array.array, list]:
            raise TypeError('SArray must contain strings, arrays or lists')
        with cython_context():
            return SArray(_proxy=self.__proxy__.subslice(start, step, stop))

    def dict_trim_by_keys(self, keys, exclude=True):
        if False:
            while True:
                i = 10
        '\n        Filter an SArray of dictionary type by the given keys. By default, all\n        keys that are in the provided list in ``keys`` are *excluded* from the\n        returned SArray.\n\n        Parameters\n        ----------\n        keys : list\n            A collection of keys to trim down the elements in the SArray.\n\n        exclude : bool, optional\n            If True, all keys that are in the input key list are removed. If\n            False, only keys that are in the input key list are retained.\n\n        Returns\n        -------\n        out : SArray\n            A SArray of dictionary type, with each dictionary element trimmed\n            according to the input criteria.\n\n        See Also\n        --------\n        dict_trim_by_values\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([{"this":1, "is":1, "dog":2},\n                                  {"this": 2, "are": 2, "cat": 1}])\n        >>> sa.dict_trim_by_keys(["this", "is", "and", "are"], exclude=True)\n        dtype: dict\n        Rows: 2\n        [{\'dog\': 2}, {\'cat\': 1}]\n        '
        if not _is_non_string_iterable(keys):
            keys = [keys]
        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_trim_by_keys(keys, exclude))

    def dict_trim_by_values(self, lower=None, upper=None):
        if False:
            print('Hello World!')
        '\n        Filter dictionary values to a given range (inclusive). Trimming is only\n        performed on values which can be compared to the bound values. Fails on\n        SArrays whose data type is not ``dict``.\n\n        Parameters\n        ----------\n        lower : int or long or float, optional\n            The lowest dictionary value that would be retained in the result. If\n            not given, lower bound is not applied.\n\n        upper : int or long or float, optional\n            The highest dictionary value that would be retained in the result.\n            If not given, upper bound is not applied.\n\n        Returns\n        -------\n        out : SArray\n            An SArray of dictionary type, with each dict element trimmed\n            according to the input criteria.\n\n        See Also\n        --------\n        dict_trim_by_keys\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},\n                                  {"this": 2, "are": 1, "cat": 5}])\n        >>> sa.dict_trim_by_values(2,5)\n        dtype: dict\n        Rows: 2\n        [{\'is\': 5}, {\'this\': 2, \'cat\': 5}]\n\n        >>> sa.dict_trim_by_values(upper=5)\n        dtype: dict\n        Rows: 2\n        [{\'this\': 1, \'is\': 5}, {\'this\': 2, \'are\': 1, \'cat\': 5}]\n        '
        if not (lower is None or isinstance(lower, numbers.Number)):
            raise TypeError('lower bound has to be a numeric value')
        if not (upper is None or isinstance(upper, numbers.Number)):
            raise TypeError('upper bound has to be a numeric value')
        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_trim_by_values(lower, upper))

    def dict_keys(self):
        if False:
            i = 10
            return i + 15
        '\n        Create an SArray that contains all the keys from each dictionary\n        element as a list. Fails on SArrays whose data type is not ``dict``.\n\n        Returns\n        -------\n        out : SArray\n            A SArray of list type, where each element is a list of keys\n            from the input SArray element.\n\n        See Also\n        --------\n        dict_values\n\n        Examples\n        ---------\n        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},\n                                  {"this": 2, "are": 1, "cat": 5}])\n        >>> sa.dict_keys()\n        dtype: list\n        Rows: 2\n        [[\'this\', \'is\', \'dog\'], [\'this\', \'are\', \'cat\']]\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_keys())

    def dict_values(self):
        if False:
            i = 10
            return i + 15
        '\n        Create an SArray that contains all the values from each dictionary\n        element as a list. Fails on SArrays whose data type is not ``dict``.\n\n        Returns\n        -------\n        out : SArray\n            A SArray of list type, where each element is a list of values\n            from the input SArray element.\n\n        See Also\n        --------\n        dict_keys\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},\n                                 {"this": 2, "are": 1, "cat": 5}])\n        >>> sa.dict_values()\n        dtype: list\n        Rows: 2\n        [[1, 5, 7], [2, 1, 5]]\n\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_values())

    def dict_has_any_keys(self, keys):
        if False:
            while True:
                i = 10
        '\n        Create a boolean SArray by checking the keys of an SArray of\n        dictionaries. An element of the output SArray is True if the\n        corresponding input element\'s dictionary has any of the given keys.\n        Fails on SArrays whose data type is not ``dict``.\n\n        Parameters\n        ----------\n        keys : list\n            A list of key values to check each dictionary against.\n\n        Returns\n        -------\n        out : SArray\n            A SArray of int type, where each element indicates whether the\n            input SArray element contains any key in the input list.\n\n        See Also\n        --------\n        dict_has_all_keys\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7}, {"animal":1},\n                                 {"this": 2, "are": 1, "cat": 5}])\n        >>> sa.dict_has_any_keys(["is", "this", "are"])\n        dtype: int\n        Rows: 3\n        [1, 0, 1]\n        '
        if not _is_non_string_iterable(keys):
            keys = [keys]
        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_has_any_keys(keys))

    def dict_has_all_keys(self, keys):
        if False:
            while True:
                i = 10
        '\n        Create a boolean SArray by checking the keys of an SArray of\n        dictionaries. An element of the output SArray is True if the\n        corresponding input element\'s dictionary has all of the given keys.\n        Fails on SArrays whose data type is not ``dict``.\n\n        Parameters\n        ----------\n        keys : list\n            A list of key values to check each dictionary against.\n\n        Returns\n        -------\n        out : SArray\n            A SArray of int type, where each element indicates whether the\n            input SArray element contains all keys in the input list.\n\n        See Also\n        --------\n        dict_has_any_keys\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([{"this":1, "is":5, "dog":7},\n                                 {"this": 2, "are": 1, "cat": 5}])\n        >>> sa.dict_has_all_keys(["is", "this"])\n        dtype: int\n        Rows: 2\n        [1, 0]\n        '
        if not _is_non_string_iterable(keys):
            keys = [keys]
        with cython_context():
            return SArray(_proxy=self.__proxy__.dict_has_all_keys(keys))

    def apply(self, fn, dtype=None, skip_na=True):
        if False:
            while True:
                i = 10
        '\n        Transform each element of the SArray by a given function. The result\n        SArray is of type ``dtype``. ``fn`` should be a function that returns\n        exactly one value which can be cast into the type specified by\n        ``dtype``. If ``dtype`` is not specified, the first 100 elements of the\n        SArray are used to make a guess about the data type.\n\n        Parameters\n        ----------\n        fn : function\n            The function to transform each element. Must return exactly one\n            value which can be cast into the type specified by ``dtype``.\n            This can also be a toolkit extension function which is compiled\n            as a native shared library using SDK.\n\n\n        dtype : {None, int, float, str, list, array.array, dict, turicreate.Image}, optional\n            The data type of the new SArray. If ``None``, the first 100 elements\n            of the array are used to guess the target data type.\n\n        skip_na : bool, optional\n            If True, will not apply ``fn`` to any undefined values.\n\n        Returns\n        -------\n        out : SArray\n            The SArray transformed by ``fn``. Each element of the SArray is of\n            type ``dtype``.\n\n        See Also\n        --------\n        SFrame.apply\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1,2,3])\n        >>> sa.apply(lambda x: x*2)\n        dtype: int\n        Rows: 3\n        [2, 4, 6]\n\n        Using native toolkit extension function:\n\n        .. code-block:: c++\n\n            #include <model_server/lib/toolkit_function_macros.hpp>\n            #include <cmath>\n\n            using namespace turi;\n            double logx(const flexible_type& x, double base) {\n              return log((double)(x)) / log(base);\n            }\n\n            BEGIN_FUNCTION_REGISTRATION\n            REGISTER_FUNCTION(logx, "x", "base");\n            END_FUNCTION_REGISTRATION\n\n        compiled into example.so\n\n        >>> import example\n\n        >>> sa = turicreate.SArray([1,2,4])\n        >>> sa.apply(lambda x: example.logx(x, 2))\n        dtype: float\n        Rows: 3\n        [0.0, 1.0, 2.0]\n        '
        assert callable(fn), 'Input function must be callable.'
        dryrun = [fn(i) for i in self.head(100) if i is not None]
        if dtype is None:
            dtype = infer_type_of_list(dryrun)
        seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        nativefn = None
        try:
            from .. import extensions
            nativefn = extensions._build_native_function_call(fn)
        except:
            pass
        if nativefn is not None:
            nativefn.native_fn_name = nativefn.native_fn_name.encode()
            with cython_context():
                return SArray(_proxy=self.__proxy__.transform_native(nativefn, dtype, skip_na, seed))
        with cython_context():
            return SArray(_proxy=self.__proxy__.transform(fn, dtype, skip_na, seed))

    def filter(self, fn, skip_na=True, seed=None):
        if False:
            return 10
        "\n        Filter this SArray by a function.\n\n        Returns a new SArray filtered by this SArray.  If `fn` evaluates an\n        element to true, this element is copied to the new SArray. If not, it\n        isn't. Throws an exception if the return type of `fn` is not castable\n        to a boolean value.\n\n        Parameters\n        ----------\n        fn : function\n            Function that filters the SArray. Must evaluate to bool or int.\n\n        skip_na : bool, optional\n            If True, will not apply fn to any undefined values.\n\n        seed : int, optional\n            Used as the seed if a random number generator is included in fn.\n\n        Returns\n        -------\n        out : SArray\n            The SArray filtered by fn. Each element of the SArray is of\n            type int.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1,2,3])\n        >>> sa.filter(lambda x: x < 3)\n        dtype: int\n        Rows: 2\n        [1, 2]\n        "
        assert callable(fn), 'Input must be callable'
        if seed is None:
            seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        with cython_context():
            return SArray(_proxy=self.__proxy__.filter(fn, skip_na, seed))

    def sample(self, fraction, seed=None, exact=False):
        if False:
            print('Hello World!')
        '\n        Create an SArray which contains a subsample of the current SArray.\n\n        Parameters\n        ----------\n        fraction : float\n            Fraction of the rows to fetch. Must be between 0 and 1.\n            if exact is False (default), the number of rows returned is\n            approximately the fraction times the number of rows.\n\n        seed : int, optional\n            The random seed for the random number generator.\n\n        exact: bool, optional\n            Defaults to False. If exact=True, an exact fraction is returned,\n            but at a performance penalty.\n\n        Returns\n        -------\n        out : SArray\n            The new SArray which contains the subsampled rows.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray(range(10))\n        >>> sa.sample(.3)\n        dtype: int\n        Rows: 3\n        [2, 6, 9]\n        '
        if fraction > 1 or fraction < 0:
            raise ValueError('Invalid sampling rate: ' + str(fraction))
        if len(self) == 0:
            return SArray()
        if seed is None:
            seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        with cython_context():
            return SArray(_proxy=self.__proxy__.sample(fraction, seed, exact))

    def hash(self, seed=0):
        if False:
            i = 10
            return i + 15
        '\n        Returns an SArray with a hash of each element. seed can be used\n        to change the hash function to allow this method to be used for\n        random number generation.\n\n        Parameters\n        ----------\n        seed : int\n            Defaults to 0. Can be changed to different values to get\n            different hash results.\n\n        Returns\n        -------\n        out : SArray\n            An integer SArray with a hash value for each element. Identical\n            elements are hashed to the same value\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.hash(seed))

    @classmethod
    def random_integers(cls, size, seed=None):
        if False:
            while True:
                i = 10
        '\n        Returns an SArray with random integer values.\n        '
        if seed is None:
            seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        return cls.from_sequence(size).hash(seed)

    def _save_as_text(self, url):
        if False:
            while True:
                i = 10
        '\n        Save the SArray to disk as text file.\n        '
        raise NotImplementedError

    def all(self):
        if False:
            return 10
        '\n        Return True if every element of the SArray evaluates to True. For\n        numeric SArrays zeros and missing values (``None``) evaluate to False,\n        while all non-zero, non-missing values evaluate to True. For string,\n        list, and dictionary SArrays, empty values (zero length strings, lists\n        or dictionaries) or missing values (``None``) evaluate to False. All\n        other values evaluate to True.\n\n        Returns True on an empty SArray.\n\n        Returns\n        -------\n        out : bool\n\n        See Also\n        --------\n        any\n\n        Examples\n        --------\n        >>> turicreate.SArray([1, None]).all()\n        False\n        >>> turicreate.SArray([1, 0]).all()\n        False\n        >>> turicreate.SArray([1, 2]).all()\n        True\n        >>> turicreate.SArray(["hello", "world"]).all()\n        True\n        >>> turicreate.SArray(["hello", ""]).all()\n        False\n        >>> turicreate.SArray([]).all()\n        True\n        '
        with cython_context():
            return self.__proxy__.all()

    def any(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if any element of the SArray evaluates to True. For numeric\n        SArrays any non-zero value evaluates to True. For string, list, and\n        dictionary SArrays, any element of non-zero length evaluates to True.\n\n        Returns False on an empty SArray.\n\n        Returns\n        -------\n        out : bool\n\n        See Also\n        --------\n        all\n\n        Examples\n        --------\n        >>> turicreate.SArray([1, None]).any()\n        True\n        >>> turicreate.SArray([1, 0]).any()\n        True\n        >>> turicreate.SArray([0, 0]).any()\n        False\n        >>> turicreate.SArray(["hello", "world"]).any()\n        True\n        >>> turicreate.SArray(["hello", ""]).any()\n        True\n        >>> turicreate.SArray(["", ""]).any()\n        False\n        >>> turicreate.SArray([]).any()\n        False\n        '
        with cython_context():
            return self.__proxy__.any()

    def max(self):
        if False:
            return 10
        '\n        Get maximum numeric value in SArray.\n\n        Returns None on an empty SArray. Raises an exception if called on an\n        SArray with non-numeric type.\n\n        Returns\n        -------\n        out : type of SArray\n            Maximum value of SArray\n\n        See Also\n        --------\n        min\n\n        Examples\n        --------\n        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).max()\n        96\n        '
        with cython_context():
            return self.__proxy__.max()

    def min(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get minimum numeric value in SArray.\n\n        Returns None on an empty SArray. Raises an exception if called on an\n        SArray with non-numeric type.\n\n        Returns\n        -------\n        out : type of SArray\n            Minimum value of SArray\n\n        See Also\n        --------\n        max\n\n        Examples\n        --------\n        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).min()\n\n        '
        with cython_context():
            return self.__proxy__.min()

    def argmax(self):
        if False:
            print('Hello World!')
        '\n        Get the index of the maximum numeric value in SArray.\n\n        Returns None on an empty SArray. Raises an exception if called on an\n        SArray with non-numeric type.\n\n        Returns\n        -------\n        out : int\n            Index of the maximum value of SArray\n\n        See Also\n        --------\n        argmin\n\n        Examples\n        --------\n        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).argmax()\n\n        '
        from .sframe import SFrame as _SFrame
        if len(self) == 0:
            return None
        if not any([isinstance(self[0], i) for i in [int, float, long]]):
            raise TypeError("SArray must be of type 'int', 'long', or 'float'.")
        sf = _SFrame(self).add_row_number()
        sf_out = sf.groupby(key_column_names=[], operations={'maximum_x1': _aggregate.ARGMAX('X1', 'id')})
        return sf_out['maximum_x1'][0]

    def argmin(self):
        if False:
            return 10
        '\n        Get the index of the minimum numeric value in SArray.\n\n        Returns None on an empty SArray. Raises an exception if called on an\n        SArray with non-numeric type.\n\n        Returns\n        -------\n        out : int\n            index of the minimum value of SArray\n\n        See Also\n        --------\n        argmax\n\n        Examples\n        --------\n        >>> turicreate.SArray([14, 62, 83, 72, 77, 96, 5, 25, 69, 66]).argmin()\n\n        '
        from .sframe import SFrame as _SFrame
        if len(self) == 0:
            return None
        if not any([isinstance(self[0], i) for i in [int, float, long]]):
            raise TypeError("SArray must be of type 'int', 'long', or 'float'.")
        sf = _SFrame(self).add_row_number()
        sf_out = sf.groupby(key_column_names=[], operations={'minimum_x1': _aggregate.ARGMIN('X1', 'id')})
        return sf_out['minimum_x1'][0]

    def sum(self):
        if False:
            return 10
        '\n        Sum of all values in this SArray.\n\n        Raises an exception if called on an SArray of strings, lists, or\n        dictionaries. If the SArray contains numeric arrays (array.array) and\n        all the arrays are the same length, the sum over all the arrays will be\n        returned. Returns None on an empty SArray. For large values, this may\n        overflow without warning.\n\n        Returns\n        -------\n        out : type of SArray\n            Sum of all values in SArray\n        '
        with cython_context():
            return self.__proxy__.sum()

    def mean(self):
        if False:
            i = 10
            return i + 15
        '\n        Mean of all the values in the SArray, or mean image.\n\n        Returns None on an empty SArray. Raises an exception if called on an\n        SArray with non-numeric type or non-Image type.\n\n        Returns\n        -------\n        out : float | turicreate.Image\n            Mean of all values in SArray, or image holding per-pixel mean\n            across the input SArray.\n\n        See Also\n        --------\n        median\n        '
        with cython_context():
            if self.dtype == _Image:
                from .. import extensions
                return extensions.generate_mean(self)
            else:
                return self.__proxy__.mean()

    def median(self, approximate=False):
        if False:
            print('Hello World!')
        '\n        Median of all the values in the SArray.\n\n        Note: no linear smoothing is performed. If the lenght of the SArray is\n        an odd number. Then a value between `a` and `b` will be used, where\n        `a` and `b` are the two middle values.\n\n        Parameters\n        ----------\n        approximate : bool\n            If True an approximate value will be returned. Calculating\n            an approximate value is faster. The approximate value will\n            be within 5% of the exact value.\n\n        Returns\n        -------\n        out : float | turicreate.Image\n            Median of all values in SArray\n\n        See Also\n        --------\n        mean\n        '
        if not isinstance(approximate, bool):
            raise '"approximate" must be a bool.'
        return self.__proxy__.median(approximate)

    def std(self, ddof=0):
        if False:
            while True:
                i = 10
        '\n        Standard deviation of all the values in the SArray.\n\n        Returns None on an empty SArray. Raises an exception if called on an\n        SArray with non-numeric type or if `ddof` >= length of SArray.\n\n        Parameters\n        ----------\n        ddof : int, optional\n            "delta degrees of freedom" in the variance calculation.\n\n        Returns\n        -------\n        out : float\n            The standard deviation of all the values.\n        '
        with cython_context():
            return self.__proxy__.std(ddof)

    def var(self, ddof=0):
        if False:
            print('Hello World!')
        '\n        Variance of all the values in the SArray.\n\n        Returns None on an empty SArray. Raises an exception if called on an\n        SArray with non-numeric type or if `ddof` >= length of SArray.\n\n        Parameters\n        ----------\n        ddof : int, optional\n            "delta degrees of freedom" in the variance calculation.\n\n        Returns\n        -------\n        out : float\n            Variance of all values in SArray.\n        '
        with cython_context():
            return self.__proxy__.var(ddof)

    def countna(self):
        if False:
            while True:
                i = 10
        '\n        Number of missing elements in the SArray.\n\n        Returns\n        -------\n        out : int\n            Number of missing values.\n        '
        with cython_context():
            return self.__proxy__.num_missing()

    def nnz(self):
        if False:
            i = 10
            return i + 15
        '\n        Number of non-zero elements in the SArray.\n\n        Returns\n        -------\n        out : int\n            Number of non-zero elements.\n        '
        with cython_context():
            return self.__proxy__.nnz()

    def datetime_to_str(self, format='%Y-%m-%dT%H:%M:%S%ZP'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new SArray with all the values cast to str. The string format is\n        specified by the \'format\' parameter.\n\n        Parameters\n        ----------\n        format : str\n            The format to output the string. Default format is "%Y-%m-%dT%H:%M:%S%ZP".\n\n        Returns\n        -------\n        out : SArray[str]\n            The SArray converted to the type \'str\'.\n\n        Examples\n        --------\n        >>> dt = datetime.datetime(2011, 10, 20, 9, 30, 10, tzinfo=GMT(-5))\n        >>> sa = turicreate.SArray([dt])\n        >>> sa.datetime_to_str("%e %b %Y %T %ZP")\n        dtype: str\n        Rows: 1\n        [20 Oct 2011 09:30:10 GMT-05:00]\n\n        See Also\n        ----------\n        str_to_datetime\n\n        References\n        ----------\n        [1] Boost date time from string conversion guide (http://www.boost.org/doc/libs/1_48_0/doc/html/date_time/date_time_io.html)\n\n        '
        if self.dtype != datetime.datetime:
            raise TypeError('datetime_to_str expects SArray of datetime as input SArray')
        with cython_context():
            return SArray(_proxy=self.__proxy__.datetime_to_str(format))

    def str_to_datetime(self, format='%Y-%m-%dT%H:%M:%S%ZP'):
        if False:
            while True:
                i = 10
        '\n        Create a new SArray with all the values cast to datetime. The string format is\n        specified by the \'format\' parameter.\n\n        Parameters\n        ----------\n        format : str\n            The string format of the input SArray. Default format is "%Y-%m-%dT%H:%M:%S%ZP".\n            If format is "ISO", the the format is "%Y%m%dT%H%M%S%F%q"\n        Returns\n        -------\n        out : SArray[datetime.datetime]\n            The SArray converted to the type \'datetime\'.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray(["20-Oct-2011 09:30:10 GMT-05:30"])\n        >>> sa.str_to_datetime("%d-%b-%Y %H:%M:%S %ZP")\n        dtype: datetime\n        Rows: 1\n        datetime.datetime(2011, 10, 20, 9, 30, 10, tzinfo=GMT(-5.5))\n\n        See Also\n        ----------\n        datetime_to_str\n\n        References\n        ----------\n        [1] boost date time to string conversion guide (http://www.boost.org/doc/libs/1_48_0/doc/html/date_time/date_time_io.html)\n\n        '
        if self.dtype != str:
            raise TypeError('str_to_datetime expects SArray of str as input SArray')
        with cython_context():
            return SArray(_proxy=self.__proxy__.str_to_datetime(format))

    def pixel_array_to_image(self, width, height, channels, undefined_on_failure=True, allow_rounding=False):
        if False:
            i = 10
            return i + 15
        "\n        Create a new SArray with all the values cast to :py:class:`turicreate.image.Image`\n        of uniform size.\n\n        Parameters\n        ----------\n        width: int\n            The width of the new images.\n\n        height: int\n            The height of the new images.\n\n        channels: int.\n            Number of channels of the new images.\n\n        undefined_on_failure: bool , optional , default True\n            If True, return None type instead of Image type in failure instances.\n            If False, raises error upon failure.\n\n        allow_rounding: bool, optional , default False\n            If True, rounds non-integer values when converting to Image type.\n            If False, raises error upon rounding.\n\n        Returns\n        -------\n        out : SArray[turicreate.Image]\n            The SArray converted to the type 'turicreate.Image'.\n\n        See Also\n        --------\n        astype, str_to_datetime, datetime_to_str\n\n        Examples\n        --------\n        The MNIST data is scaled from 0 to 1, but our image type only loads integer  pixel values\n        from 0 to 255. If we just convert without scaling, all values below one would be cast to\n        0.\n\n        >>> mnist_array = turicreate.SArray('https://static.turi.com/datasets/mnist/mnist_vec_sarray')\n        >>> scaled_mnist_array = mnist_array * 255\n        >>> mnist_img_sarray = tc.SArray.pixel_array_to_image(scaled_mnist_array, 28, 28, 1, allow_rounding = True)\n\n        "
        if self.dtype != array.array:
            raise TypeError('array_to_img expects SArray of arrays as input SArray')
        num_to_test = 10
        num_test = min(len(self), num_to_test)
        mod_values = [val % 1 for x in range(num_test) for val in self[x]]
        out_of_range_values = [val > 255 or val < 0 for x in range(num_test) for val in self[x]]
        if sum(mod_values) != 0.0 and (not allow_rounding):
            raise ValueError("There are non-integer values in the array data. Images only support integer data values between 0 and 255. To permit rounding, set the 'allow_rounding' parameter to 1.")
        if sum(out_of_range_values) != 0:
            raise ValueError('There are values outside the range of 0 to 255. Images only support integer data values between 0 and 255.')
        from .. import extensions
        return extensions.vector_sarray_to_image_sarray(self, width, height, channels, undefined_on_failure)

    def astype(self, dtype, undefined_on_failure=False):
        if False:
            i = 10
            return i + 15
        "\n        Create a new SArray with all values cast to the given type. Throws an\n        exception if the types are not castable to the given type.\n\n        Parameters\n        ----------\n        dtype : {int, float, str, list, array.array, dict, datetime.datetime}\n            The type to cast the elements to in SArray\n\n        undefined_on_failure: bool, optional\n            If set to True, runtime cast failures will be emitted as missing\n            values rather than failing.\n\n        Returns\n        -------\n        out : SArray [dtype]\n            The SArray converted to the type ``dtype``.\n\n        Notes\n        -----\n        - The string parsing techniques used to handle conversion to dictionary\n          and list types are quite generic and permit a variety of interesting\n          formats to be interpreted. For instance, a JSON string can usually be\n          interpreted as a list or a dictionary type. See the examples below.\n        - For datetime-to-string  and string-to-datetime conversions,\n          use sa.datetime_to_str() and sa.str_to_datetime() functions.\n        - For array.array to turicreate.Image conversions, use sa.pixel_array_to_image()\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray(['1','2','3','4'])\n        >>> sa.astype(int)\n        dtype: int\n        Rows: 4\n        [1, 2, 3, 4]\n\n        Given an SArray of strings that look like dicts, convert to a dictionary\n        type:\n\n        >>> sa = turicreate.SArray(['{1:2 3:4}', '{a:b c:d}'])\n        >>> sa.astype(dict)\n        dtype: dict\n        Rows: 2\n        [{1: 2, 3: 4}, {'a': 'b', 'c': 'd'}]\n        "
        if dtype == _Image and self.dtype == array.array:
            raise TypeError('Cannot cast from image type to array with sarray.astype(). Please use sarray.pixel_array_to_image() instead.')
        if float('nan') in self:
            import turicreate as _tc
            self = _tc.SArray.where(self == float('nan'), None, self)
        with cython_context():
            return SArray(_proxy=self.__proxy__.astype(dtype, undefined_on_failure))

    def clip(self, lower=float('nan'), upper=float('nan')):
        if False:
            print('Hello World!')
        '\n        Create a new SArray with each value clipped to be within the given\n        bounds.\n\n        In this case, "clipped" means that values below the lower bound will be\n        set to the lower bound value. Values above the upper bound will be set\n        to the upper bound value. This function can operate on SArrays of\n        numeric type as well as array type, in which case each individual\n        element in each array is clipped. By default ``lower`` and ``upper`` are\n        set to ``float(\'nan\')`` which indicates the respective bound should be\n        ignored. The method fails if invoked on an SArray of non-numeric type.\n\n        Parameters\n        ----------\n        lower : int, optional\n            The lower bound used to clip. Ignored if equal to ``float(\'nan\')``\n            (the default).\n\n        upper : int, optional\n            The upper bound used to clip. Ignored if equal to ``float(\'nan\')``\n            (the default).\n\n        Returns\n        -------\n        out : SArray\n\n        See Also\n        --------\n        clip_lower, clip_upper\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1,2,3])\n        >>> sa.clip(2,2)\n        dtype: int\n        Rows: 3\n        [2, 2, 2]\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.clip(lower, upper))

    def clip_lower(self, threshold):
        if False:
            while True:
                i = 10
        '\n        Create new SArray with all values clipped to the given lower bound. This\n        function can operate on numeric arrays, as well as vector arrays, in\n        which case each individual element in each vector is clipped. Throws an\n        exception if the SArray is empty or the types are non-numeric.\n\n        Parameters\n        ----------\n        threshold : float\n            The lower bound used to clip values.\n\n        Returns\n        -------\n        out : SArray\n\n        See Also\n        --------\n        clip, clip_upper\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1,2,3])\n        >>> sa.clip_lower(2)\n        dtype: int\n        Rows: 3\n        [2, 2, 3]\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.clip(threshold, float('nan')))

    def clip_upper(self, threshold):
        if False:
            return 10
        '\n        Create new SArray with all values clipped to the given upper bound. This\n        function can operate on numeric arrays, as well as vector arrays, in\n        which case each individual element in each vector is clipped.\n\n        Parameters\n        ----------\n        threshold : float\n            The upper bound used to clip values.\n\n        Returns\n        -------\n        out : SArray\n\n        See Also\n        --------\n        clip, clip_lower\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1,2,3])\n        >>> sa.clip_upper(2)\n        dtype: int\n        Rows: 3\n        [1, 2, 2]\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.clip(float('nan'), threshold))

    def tail(self, n=10):
        if False:
            i = 10
            return i + 15
        '\n        Get an SArray that contains the last n elements in the SArray.\n\n        Parameters\n        ----------\n        n : int\n            The number of elements to fetch\n\n        Returns\n        -------\n        out : SArray\n            A new SArray which contains the last n rows of the current SArray.\n        '
        with cython_context():
            return SArray(_proxy=self.__proxy__.tail(n))

    def dropna(self):
        if False:
            while True:
                i = 10
        "\n        Create new SArray containing only the non-missing values of the\n        SArray.\n\n        A missing value shows up in an SArray as 'None'.  This will also drop\n        float('nan').\n\n        Returns\n        -------\n        out : SArray\n            The new SArray with missing values removed.\n        "
        with cython_context():
            return SArray(_proxy=self.__proxy__.drop_missing_values())

    def fillna(self, value):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create new SArray with all missing values (None or NaN) filled in\n        with the given value.\n\n        The size of the new SArray will be the same as the original SArray. If\n        the given value is not the same type as the values in the SArray,\n        `fillna` will attempt to convert the value to the original SArray's\n        type. If this fails, an error will be raised.\n\n        Parameters\n        ----------\n        value : type convertible to SArray's type\n            The value used to replace all missing values\n\n        Returns\n        -------\n        out : SArray\n            A new SArray with all missing values filled\n        "
        with cython_context():
            return SArray(_proxy=self.__proxy__.fill_missing_values(value))

    def is_topk(self, topk=10, reverse=False):
        if False:
            return 10
        "\n        Create an SArray indicating which elements are in the top k.\n\n        Entries are '1' if the corresponding element in the current SArray is a\n        part of the top k elements, and '0' if that corresponding element is\n        not. Order is descending by default.\n\n        Parameters\n        ----------\n        topk : int\n            The number of elements to determine if 'top'\n\n        reverse : bool\n            If True, return the topk elements in ascending order\n\n        Returns\n        -------\n        out : SArray (of type int)\n\n        Notes\n        -----\n        This is used internally by SFrame's topk function.\n        "
        with cython_context():
            return SArray(_proxy=self.__proxy__.topk_index(topk, reverse))

    def summary(self, background=False, sub_sketch_keys=None):
        if False:
            while True:
                i = 10
        '\n        Summary statistics that can be calculated with one pass over the SArray.\n\n        Returns a turicreate.Sketch object which can be further queried for many\n        descriptive statistics over this SArray. Many of the statistics are\n        approximate. See the :class:`~turicreate.Sketch` documentation for more\n        detail.\n\n        Parameters\n        ----------\n        background : boolean, optional\n          If True, the sketch construction will return immediately and the\n          sketch will be constructed in the background. While this is going on,\n          the sketch can be queried incrementally, but at a performance penalty.\n          Defaults to False.\n\n        sub_sketch_keys : int | str | list of int | list of str, optional\n            For SArray of dict type, also constructs sketches for a given set of keys,\n            For SArray of array type, also constructs sketches for the given indexes.\n            The sub sketches may be queried using: :py:func:`~turicreate.Sketch.element_sub_sketch()`.\n            Defaults to None in which case no subsketches will be constructed.\n\n        Returns\n        -------\n        out : Sketch\n            Sketch object that contains descriptive statistics for this SArray.\n            Many of the statistics are approximate.\n        '
        from ..data_structures.sketch import Sketch
        if self.dtype == _Image:
            raise TypeError('summary() is not supported for arrays of image type')
        if type(background) != bool:
            raise TypeError("'background' parameter has to be a boolean value")
        if sub_sketch_keys is not None:
            if self.dtype != dict and self.dtype != array.array:
                raise TypeError('sub_sketch_keys is only supported for SArray of dictionary or array type')
            if not _is_non_string_iterable(sub_sketch_keys):
                sub_sketch_keys = [sub_sketch_keys]
            value_types = set([type(i) for i in sub_sketch_keys])
            if len(value_types) != 1:
                raise ValueError('sub_sketch_keys member values need to have the same type.')
            value_type = value_types.pop()
            if self.dtype == dict and value_type != str:
                raise TypeError('Only string value(s) can be passed to sub_sketch_keys for SArray of dictionary type. ' + 'For dictionary types, sketch summary is computed by casting keys to string values.')
            if self.dtype == array.array and value_type != int:
                raise TypeError('Only int value(s) can be passed to sub_sketch_keys for SArray of array type')
        else:
            sub_sketch_keys = list()
        return Sketch(self, background, sub_sketch_keys=sub_sketch_keys)

    def value_counts(self):
        if False:
            return 10
        "\n        Return an SFrame containing counts of unique values. The resulting\n        SFrame will be sorted in descending frequency.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame containing 2 columns : 'value', and 'count'. The SFrame will\n            be sorted in descending order by the column 'count'.\n\n        See Also\n        --------\n        SFrame.summary\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1,1,2,2,2,2,3,3,3,3,3,3,3])\n        >>> sa.value_counts()\n            Columns:\n                    value\tint\n                    count\tint\n            Rows: 3\n            Data:\n            +-------+-------+\n            | value | count |\n            +-------+-------+\n            |   3   |   7   |\n            |   2   |   4   |\n            |   1   |   2   |\n            +-------+-------+\n            [3 rows x 2 columns]\n        "
        from .sframe import SFrame as _SFrame
        return _SFrame({'value': self}).groupby('value', {'count': _aggregate.COUNT}).sort('count', ascending=False)

    def append(self, other):
        if False:
            while True:
                i = 10
        '\n        Append an SArray to the current SArray. Creates a new SArray with the\n        rows from both SArrays. Both SArrays must be of the same type.\n\n        Parameters\n        ----------\n        other : SArray\n            Another SArray whose rows are appended to current SArray.\n\n        Returns\n        -------\n        out : SArray\n            A new SArray that contains rows from both SArrays, with rows from\n            the ``other`` SArray coming after all rows from the current SArray.\n\n        See Also\n        --------\n        SFrame.append\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray([1, 2, 3])\n        >>> sa2 = turicreate.SArray([4, 5, 6])\n        >>> sa.append(sa2)\n        dtype: int\n        Rows: 6\n        [1, 2, 3, 4, 5, 6]\n        '
        if type(other) is not SArray:
            raise RuntimeError('SArray append can only work with SArray')
        if self.dtype != other.dtype:
            if len(other) == 0:
                other = other.astype(self.dtype)
            elif len(self) == 0:
                self = self.astype(other.dtype)
            else:
                raise RuntimeError('Data types in both SArrays have to be the same')
        with cython_context():
            return SArray(_proxy=self.__proxy__.append(other.__proxy__))

    def unique(self):
        if False:
            while True:
                i = 10
        '\n        Get all unique values in the current SArray.\n\n        Raises a TypeError if the SArray is of dictionary type. Will not\n        necessarily preserve the order of the given SArray in the new SArray.\n\n\n        Returns\n        -------\n        out : SArray\n            A new SArray that contains the unique values of the current SArray.\n\n        See Also\n        --------\n        SFrame.unique\n        '
        from .sframe import SFrame as _SFrame
        tmp_sf = _SFrame()
        tmp_sf.add_column(self, 'X1', inplace=True)
        res = tmp_sf.groupby('X1', {})
        return SArray(_proxy=res['X1'].__proxy__)

    def explore(self, title=None):
        if False:
            while True:
                i = 10
        '\n        Explore the SArray in an interactive GUI. Opens a new app window.\n\n        Parameters\n        ----------\n        title : str\n            The plot title to show for the resulting visualization. Defaults to None.\n            If the title is None, a default title will be provided.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n        Suppose \'sa\' is an SArray, we can view it using:\n\n        >>> sa.explore()\n\n        To override the default plot title and axis labels:\n\n        >>> sa.explore(title="My Plot Title")\n        '
        from .sframe import SFrame as _SFrame
        _SFrame({'SArray': self}).explore()

    def show(self, title=LABEL_DEFAULT, xlabel=LABEL_DEFAULT, ylabel=LABEL_DEFAULT):
        if False:
            for i in range(10):
                print('nop')
        '\n        Visualize the SArray.\n\n        Notes\n        -----\n        - The plot will render either inline in a Jupyter Notebook, in a web\n          browser, or in a native GUI window, depending on the value provided in\n          `turicreate.visualization.set_target` (defaults to \'auto\').\n\n        Parameters\n        ----------\n        title : str\n            The plot title to show for the resulting visualization.\n            If the title is None, the title will be omitted.\n\n        xlabel : str\n            The X axis label to show for the resulting visualization.\n            If the xlabel is None, the X axis label will be omitted.\n\n        ylabel : str\n            The Y axis label to show for the resulting visualization.\n            If the ylabel is None, the Y axis label will be omitted.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n        Suppose \'sa\' is an SArray, we can view it using:\n\n        >>> sa.show()\n\n        To override the default plot title and axis labels:\n\n        >>> sa.show(title="My Plot Title", xlabel="My X Axis", ylabel="My Y Axis")\n        '
        returned_plot = self.plot(title, xlabel, ylabel)
        returned_plot.show()

    def plot(self, title=LABEL_DEFAULT, xlabel=LABEL_DEFAULT, ylabel=LABEL_DEFAULT):
        if False:
            return 10
        '\n        Create a Plot object representing the SArray.\n\n        Parameters\n        ----------\n        title : str\n            The plot title to show for the resulting visualization.\n            If the title is None, the title will be omitted.\n\n        xlabel : str\n            The X axis label to show for the resulting visualization.\n            If the xlabel is None, the X axis label will be omitted.\n\n        ylabel : str\n            The Y axis label to show for the resulting visualization.\n            If the ylabel is None, the Y axis label will be omitted.\n\n        Returns\n        -------\n        out : Plot\n        A :class: Plot object that is the visualization of the SArray.\n\n        Examples\n        --------\n        Suppose \'sa\' is an SArray, we can create a plot of it using:\n\n        >>> plt = sa.plot()\n\n        To override the default plot title and axis labels:\n\n        >>> plt = sa.plot(title="My Plot Title", xlabel="My X Axis", ylabel="My Y Axis")\n\n        We can then visualize the plot using:\n\n        >>> plt.show()\n        '
        if title == '':
            title = ' '
        if xlabel == '':
            xlabel = ' '
        if ylabel == '':
            ylabel = ' '
        if title is None:
            title = ''
        if xlabel is None:
            xlabel = ''
        if ylabel is None:
            ylabel = ''
        return Plot(_proxy=self.__proxy__.plot(title, xlabel, ylabel))

    def item_length(self):
        if False:
            return 10
        '\n        Length of each element in the current SArray.\n\n        Only works on SArrays of dict, array, or list type. If a given element\n        is a missing value, then the output elements is also a missing value.\n        This function is equivalent to the following but more performant:\n\n            sa_item_len =  sa.apply(lambda x: len(x) if x is not None else None)\n\n        Returns\n        -------\n        out_sf : SArray\n            A new SArray, each element in the SArray is the len of the corresponding\n            items in original SArray.\n\n        Examples\n        --------\n        >>> sa = SArray([\n        ...  {"is_restaurant": 1, "is_electronics": 0},\n        ...  {"is_restaurant": 1, "is_retail": 1, "is_electronics": 0},\n        ...  {"is_restaurant": 0, "is_retail": 1, "is_electronics": 0},\n        ...  {"is_restaurant": 0},\n        ...  {"is_restaurant": 1, "is_electronics": 1},\n        ...  None])\n        >>> sa.item_length()\n        dtype: int\n        Rows: 6\n        [2, 3, 3, 1, 2, None]\n        '
        if self.dtype not in [list, dict, array.array]:
            raise TypeError('item_length() is only applicable for SArray of type list, dict and array.')
        with cython_context():
            return SArray(_proxy=self.__proxy__.item_length())

    def shuffle(self):
        if False:
            return 10
        '\n        Randomly shuffles the elements of the SArray.\n\n        Returns\n        -------\n        out : [SArray]\n            An SArray with all the same elements but in a random order.\n\n        Examples\n        --------\n        >>> sa = turicreate.SArray(["a", "b", "c", "d", "x", "Y", "z"])\n        >>> shuffled_sa = sa.shuffle()\n        >>> print(shuffled_sa)\n        [\'z\', \'d\', \'Y\', \'c\', \'a\', \'x\', \'b\']\n        '
        from .sframe import SFrame
        sf = SFrame({'content': self})
        sf = sf.shuffle()
        return sf['content']

    def random_split(self, fraction, seed=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Randomly split the rows of an SArray into two SArrays. The first SArray\n        contains *M* rows, sampled uniformly (without replacement) from the\n        original SArray. *M* is approximately the fraction times the original\n        number of rows. The second SArray contains the remaining rows of the\n        original SArray.\n\n        Parameters\n        ----------\n        fraction : float\n            Approximate fraction of the rows to fetch for the first returned\n            SArray. Must be between 0 and 1.\n\n        seed : int, optional\n            Seed for the random number generator used to split.\n\n        Returns\n        -------\n        out : tuple [SArray]\n            Two new SArrays.\n\n        Examples\n        --------\n        Suppose we have an SArray with 1,024 rows and we want to randomly split\n        it into training and testing datasets with about a 90%/10% split.\n\n        >>> sa = turicreate.SArray(range(1024))\n        >>> sa_train, sa_test = sa.random_split(.9, seed=5)\n        >>> print(len(sa_train), len(sa_test))\n        922 102\n        '
        from .sframe import SFrame
        temporary_sf = SFrame()
        temporary_sf['X1'] = self
        (train, test) = temporary_sf.random_split(fraction, seed)
        return (train['X1'], test['X1'])

    def split_datetime(self, column_name_prefix='X', limit=None, timezone=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Splits an SArray of datetime type to multiple columns, return a\n        new SFrame that contains expanded columns. A SArray of datetime will be\n        split by default into an SFrame of 6 columns, one for each\n        year/month/day/hour/minute/second element.\n\n        **Column Naming**\n\n        When splitting a SArray of datetime type, new columns are named:\n        prefix.year, prefix.month, etc. The prefix is set by the parameter\n        "column_name_prefix" and defaults to \'X\'. If column_name_prefix is\n        None or empty, then no prefix is used.\n\n        **Timezone Column**\n        If timezone parameter is True, then timezone information is represented\n        as one additional column which is a float shows the offset from\n        GMT(0.0) or from UTC.\n\n\n        Parameters\n        ----------\n        column_name_prefix: str, optional\n            If provided, expanded column names would start with the given prefix.\n            Defaults to "X".\n\n        limit: list[str], optional\n            Limits the set of datetime elements to expand.\n            Possible values are \'year\',\'month\',\'day\',\'hour\',\'minute\',\'second\',\n            \'weekday\', \'isoweekday\', \'tmweekday\', and \'us\'.\n            If not provided, only [\'year\',\'month\',\'day\',\'hour\',\'minute\',\'second\']\n            are expanded.\n\n            - \'year\': The year number\n            - \'month\': A value between 1 and 12 where 1 is January.\n            - \'day\': Day of the months. Begins at 1.\n            - \'hour\': Hours since midnight.\n            - \'minute\': Minutes after the hour.\n            - \'second\': Seconds after the minute.\n            - \'us\': Microseconds after the second. Between 0 and 999,999.\n            - \'weekday\': A value between 0 and 6 where 0 is Monday.\n            - \'isoweekday\': A value between 1 and 7 where 1 is Monday.\n            - \'tmweekday\': A value between 0 and 7 where 0 is Sunday\n\n        timezone: bool, optional\n            A boolean parameter that determines whether to show timezone column or not.\n            Defaults to False.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that contains all expanded columns\n\n        Examples\n        --------\n        To expand only day and year elements of a datetime SArray\n\n         >>> sa = SArray(\n            [datetime(2011, 1, 21, 7, 7, 21, tzinfo=GMT(0)),\n             datetime(2010, 2, 5, 7, 8, 21, tzinfo=GMT(4.5)])\n\n         >>> sa.split_datetime(column_name_prefix=None,limit=[\'day\',\'year\'])\n            Columns:\n                day   int\n                year  int\n            Rows: 2\n            Data:\n            +-------+--------+\n            |  day  |  year  |\n            +-------+--------+\n            |   21  |  2011  |\n            |   5   |  2010  |\n            +-------+--------+\n            [2 rows x 2 columns]\n\n\n        To expand only year and timezone elements of a datetime SArray\n        with timezone column represented as a string. Columns are named with prefix:\n        \'Y.column_name\'.\n\n        >>> sa.split_datetime(column_name_prefix="Y",limit=[\'year\'],timezone=True)\n            Columns:\n                Y.year  int\n                Y.timezone float\n            Rows: 2\n            Data:\n            +----------+---------+\n            |  Y.year  | Y.timezone |\n            +----------+---------+\n            |    2011  |  0.0    |\n            |    2010  |  4.5    |\n            +----------+---------+\n            [2 rows x 2 columns]\n        '
        from .sframe import SFrame as _SFrame
        if self.dtype != datetime.datetime:
            raise TypeError('Only column of datetime type is supported.')
        if column_name_prefix is None:
            column_name_prefix = ''
        if six.PY2 and type(column_name_prefix) == unicode:
            column_name_prefix = column_name_prefix.encode('utf-8')
        if type(column_name_prefix) != str:
            raise TypeError("'column_name_prefix' must be a string")
        if limit is not None:
            if not _is_non_string_iterable(limit):
                raise TypeError("'limit' must be a list")
            name_types = set([type(i) for i in limit])
            if len(name_types) != 1:
                raise TypeError("'limit' contains values that are different types")
            if name_types.pop() != str:
                raise TypeError("'limit' must contain string values.")
            if len(set(limit)) != len(limit):
                raise ValueError("'limit' contains duplicate values")
        column_types = []
        if limit is None:
            limit = ['year', 'month', 'day', 'hour', 'minute', 'second']
        column_types = [int] * len(limit)
        if timezone == True:
            limit += ['timezone']
            column_types += [float]
        with cython_context():
            return _SFrame(_proxy=self.__proxy__.expand(column_name_prefix, limit, column_types))

    def stack(self, new_column_name=None, drop_na=False, new_column_type=None):
        if False:
            return 10
        '\n        Convert a "wide" SArray to one or two "tall" columns in an SFrame by\n        stacking all values.\n\n        The stack works only for columns of dict, list, or array type.  If the\n        column is dict type, two new columns are created as a result of\n        stacking: one column holds the key and another column holds the value.\n        The rest of the columns are repeated for each key/value pair.\n\n        If the column is array or list type, one new column is created as a\n        result of stacking. With each row holds one element of the array or list\n        value, and the rest columns from the same original row repeated.\n\n        The returned SFrame includes the newly created column(s).\n\n        Parameters\n        --------------\n        new_column_name : str | list of str, optional\n            The new column name(s). If original column is list/array type,\n            new_column_name must a string. If original column is dict type,\n            new_column_name must be a list of two strings. If not given, column\n            names are generated automatically.\n\n        drop_na : boolean, optional\n            If True, missing values and empty list/array/dict are all dropped\n            from the resulting column(s). If False, missing values are\n            maintained in stacked column(s).\n\n        new_column_type : type | list of types, optional\n            The new column types. If original column is a list/array type\n            new_column_type must be a single type, or a list of one type. If\n            original column is of dict type, new_column_type must be a list of\n            two types. If not provided, the types are automatically inferred\n            from the first 100 values of the SFrame.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that contains the newly stacked column(s).\n\n        Examples\n        ---------\n        Suppose \'sa\' is an SArray of dict type:\n\n        >>> sa = turicreate.SArray([{\'a\':3, \'cat\':2},\n        ...                         {\'a\':1, \'the\':2},\n        ...                         {\'the\':1, \'dog\':3},\n        ...                         {}])\n        [{\'a\': 3, \'cat\': 2}, {\'a\': 1, \'the\': 2}, {\'the\': 1, \'dog\': 3}, {}]\n\n        Stack would stack all keys in one column and all values in another\n        column:\n\n        >>> sa.stack(new_column_name=[\'word\', \'count\'])\n        +------+-------+\n        | word | count |\n        +------+-------+\n        |  a   |   3   |\n        | cat  |   2   |\n        |  a   |   1   |\n        | the  |   2   |\n        | the  |   1   |\n        | dog  |   3   |\n        | None |  None |\n        +------+-------+\n        [7 rows x 2 columns]\n\n        Observe that since topic 4 had no words, an empty row is inserted.\n        To drop that row, set drop_na=True in the parameters to stack.\n        '
        from .sframe import SFrame as _SFrame
        return _SFrame({'SArray': self}).stack('SArray', new_column_name=new_column_name, drop_na=drop_na, new_column_type=new_column_type)

    def unpack(self, column_name_prefix='X', column_types=None, na_value=None, limit=None):
        if False:
            i = 10
            return i + 15
        "\n        Convert an SArray of list, array, or dict type to an SFrame with\n        multiple columns.\n\n        `unpack` expands an SArray using the values of each list/array/dict as\n        elements in a new SFrame of multiple columns. For example, an SArray of\n        lists each of length 4 will be expanded into an SFrame of 4 columns,\n        one for each list element. An SArray of lists/arrays of varying size\n        will be expand to a number of columns equal to the longest list/array.\n        An SArray of dictionaries will be expanded into as many columns as\n        there are keys.\n\n        When unpacking an SArray of list or array type, new columns are named:\n        `column_name_prefix`.0, `column_name_prefix`.1, etc. If unpacking a\n        column of dict type, unpacked columns are named\n        `column_name_prefix`.key1, `column_name_prefix`.key2, etc.\n\n        When unpacking an SArray of list or dictionary types, missing values in\n        the original element remain as missing values in the resultant columns.\n        If the `na_value` parameter is specified, all values equal to this\n        given value are also replaced with missing values. In an SArray of\n        array.array type, NaN is interpreted as a missing value.\n\n        :py:func:`turicreate.SFrame.pack_columns()` is the reverse effect of unpack\n\n        Parameters\n        ----------\n        column_name_prefix: str, optional\n            If provided, unpacked column names would start with the given prefix.\n\n        column_types: list[type], optional\n            Column types for the unpacked columns. If not provided, column\n            types are automatically inferred from first 100 rows. Defaults to\n            None.\n\n        na_value: optional\n            Convert all values that are equal to `na_value` to\n            missing value if specified.\n\n        limit: list, optional\n            Limits the set of list/array/dict keys to unpack.\n            For list/array SArrays, 'limit' must contain integer indices.\n            For dict SArray, 'limit' must contain dictionary keys.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that contains all unpacked columns\n\n        Examples\n        --------\n        To unpack a dict SArray\n\n        >>> sa = SArray([{ 'word': 'a',     'count': 1},\n        ...              { 'word': 'cat',   'count': 2},\n        ...              { 'word': 'is',    'count': 3},\n        ...              { 'word': 'coming','count': 4}])\n\n        Normal case of unpacking SArray of type dict:\n\n        >>> sa.unpack(column_name_prefix=None)\n        Columns:\n            count   int\n            word    str\n        <BLANKLINE>\n        Rows: 4\n        <BLANKLINE>\n        Data:\n        +-------+--------+\n        | count |  word  |\n        +-------+--------+\n        |   1   |   a    |\n        |   2   |  cat   |\n        |   3   |   is   |\n        |   4   | coming |\n        +-------+--------+\n        [4 rows x 2 columns]\n        <BLANKLINE>\n\n        Unpack only keys with 'word':\n\n        >>> sa.unpack(limit=['word'])\n        Columns:\n            X.word  str\n        <BLANKLINE>\n        Rows: 4\n        <BLANKLINE>\n        Data:\n        +--------+\n        | X.word |\n        +--------+\n        |   a    |\n        |  cat   |\n        |   is   |\n        | coming |\n        +--------+\n        [4 rows x 1 columns]\n        <BLANKLINE>\n\n        >>> sa2 = SArray([\n        ...               [1, 0, 1],\n        ...               [1, 1, 1],\n        ...               [0, 1]])\n\n        Convert all zeros to missing values:\n\n        >>> sa2.unpack(column_types=[int, int, int], na_value=0)\n        Columns:\n            X.0     int\n            X.1     int\n            X.2     int\n        <BLANKLINE>\n        Rows: 3\n        <BLANKLINE>\n        Data:\n        +------+------+------+\n        | X.0  | X.1  | X.2  |\n        +------+------+------+\n        |  1   | None |  1   |\n        |  1   |  1   |  1   |\n        | None |  1   | None |\n        +------+------+------+\n        [3 rows x 3 columns]\n        <BLANKLINE>\n        "
        from .sframe import SFrame as _SFrame
        if self.dtype not in [dict, array.array, list]:
            raise TypeError('Only SArray of dict/list/array type supports unpack')
        if column_name_prefix is None:
            column_name_prefix = ''
        if not isinstance(column_name_prefix, six.string_types):
            raise TypeError("'column_name_prefix' must be a string")
        if limit is not None:
            if not _is_non_string_iterable(limit):
                raise TypeError("'limit' must be a list")
            name_types = set([type(i) for i in limit])
            if len(name_types) != 1:
                raise TypeError("'limit' contains values that are different types")
            if self.dtype != dict and name_types.pop() != int:
                raise TypeError("'limit' must contain integer values.")
            if len(set(limit)) != len(limit):
                raise ValueError("'limit' contains duplicate values")
        if column_types is not None:
            if not _is_non_string_iterable(column_types):
                raise TypeError('column_types must be a list')
            for column_type in column_types:
                if column_type not in (int, float, str, list, dict, array.array):
                    raise TypeError("column_types contains unsupported types. Supported types are ['float', 'int', 'list', 'dict', 'str', 'array.array']")
            if limit is not None:
                if len(limit) != len(column_types):
                    raise ValueError('limit and column_types do not have the same length')
            elif self.dtype == dict:
                raise ValueError("if 'column_types' is given, 'limit' has to be provided to unpack dict type.")
            else:
                limit = range(len(column_types))
        else:
            head_rows = self.head(100).dropna()
            lengths = [len(i) for i in head_rows]
            if len(lengths) == 0 or max(lengths) == 0:
                raise RuntimeError('Cannot infer number of items from the SArray, SArray may be empty. please explicitly provide column types')
            if self.dtype != dict:
                length = max(lengths)
                if limit is None:
                    limit = range(length)
                else:
                    length = len(limit)
                if self.dtype == array.array:
                    column_types = [float for i in range(length)]
                else:
                    column_types = list()
                    for i in limit:
                        t = [x[i] if x is not None and len(x) > i else None for x in head_rows]
                        column_types.append(infer_type_of_list(t))
        with cython_context():
            if self.dtype == dict and column_types is None:
                limit = limit if limit is not None else []
                return _SFrame(_proxy=self.__proxy__.unpack_dict(column_name_prefix.encode('utf-8'), limit, na_value))
            else:
                return _SFrame(_proxy=self.__proxy__.unpack(column_name_prefix.encode('utf-8'), limit, column_types, na_value))

    def sort(self, ascending=True):
        if False:
            return 10
        '\n        Sort all values in this SArray.\n\n        Sort only works for sarray of type str, int and float, otherwise TypeError\n        will be raised. Creates a new, sorted SArray.\n\n        Parameters\n        ----------\n        ascending: boolean, optional\n           If true, the sarray values are sorted in ascending order, otherwise,\n           descending order.\n\n        Returns\n        -------\n        out: SArray\n\n        Examples\n        --------\n        >>> sa = SArray([3,2,1])\n        >>> sa.sort()\n        dtype: int\n        Rows: 3\n        [1, 2, 3]\n        '
        from .sframe import SFrame as _SFrame
        if self.dtype not in (int, float, str, datetime.datetime):
            raise TypeError('Only sarray with type (int, float, str, datetime.datetime) can be sorted')
        sf = _SFrame()
        sf['a'] = self
        return sf.sort('a', ascending)['a']

    def __check_min_observations(self, min_observations):
        if False:
            i = 10
            return i + 15
        if min_observations is None:
            min_observations = (1 << 64) - 1
        if min_observations < 0:
            raise ValueError('min_observations must be a positive integer')
        return min_observations

    def rolling_mean(self, window_start, window_end, min_observations=None):
        if False:
            i = 10
            return i + 15
        '\n        Calculate a new SArray of the mean of different subsets over this\n        SArray.\n\n        Also known as a "moving average" or "running average". The subset that\n        the mean is calculated over is defined as an inclusive range relative\n        to the position to each value in the SArray, using `window_start` and\n        `window_end`. For a better understanding of this, see the examples\n        below.\n\n        Parameters\n        ----------\n        window_start : int\n            The start of the subset to calculate the mean relative to the\n            current value.\n\n        window_end : int\n            The end of the subset to calculate the mean relative to the current\n            value. Must be greater than `window_start`.\n\n        min_observations : int\n            Minimum number of non-missing observations in window required to\n            calculate the mean (otherwise result is None). None signifies that\n            the entire window must not include a missing value. A negative\n            number throws an error.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import pandas\n        >>> sa = SArray([1,2,3,4,5])\n        >>> series = pandas.Series([1,2,3,4,5])\n\n        A rolling mean with a window including the previous 2 entries including\n        the current:\n        >>> sa.rolling_mean(-2,0)\n        dtype: float\n        Rows: 5\n        [None, None, 2.0, 3.0, 4.0]\n\n        Pandas equivalent:\n        >>> pandas.rolling_mean(series, 3)\n        0   NaN\n        1   NaN\n        2     2\n        3     3\n        4     4\n        dtype: float64\n\n        Same rolling mean operation, but 2 minimum observations:\n        >>> sa.rolling_mean(-2,0,min_observations=2)\n        dtype: float\n        Rows: 5\n        [None, 1.5, 2.0, 3.0, 4.0]\n\n        Pandas equivalent:\n        >>> pandas.rolling_mean(series, 3, min_periods=2)\n        0    NaN\n        1    1.5\n        2    2.0\n        3    3.0\n        4    4.0\n        dtype: float64\n\n        A rolling mean with a size of 3, centered around the current:\n        >>> sa.rolling_mean(-1,1)\n        dtype: float\n        Rows: 5\n        [None, 2.0, 3.0, 4.0, None]\n\n        Pandas equivalent:\n        >>> pandas.rolling_mean(series, 3, center=True)\n        0   NaN\n        1     2\n        2     3\n        3     4\n        4   NaN\n        dtype: float64\n\n        A rolling mean with a window including the current and the 2 entries\n        following:\n        >>> sa.rolling_mean(0,2)\n        dtype: float\n        Rows: 5\n        [2.0, 3.0, 4.0, None, None]\n\n        A rolling mean with a window including the previous 2 entries NOT\n        including the current:\n        >>> sa.rolling_mean(-2,-1)\n        dtype: float\n        Rows: 5\n        [None, None, 1.5, 2.5, 3.5]\n        '
        min_observations = self.__check_min_observations(min_observations)
        agg_op = None
        if self.dtype is array.array:
            agg_op = '__builtin__vector__avg__'
        else:
            agg_op = '__builtin__avg__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_sum(self, window_start, window_end, min_observations=None):
        if False:
            print('Hello World!')
        '\n        Calculate a new SArray of the sum of different subsets over this\n        SArray.\n\n        Also known as a "moving sum" or "running sum". The subset that\n        the sum is calculated over is defined as an inclusive range relative\n        to the position to each value in the SArray, using `window_start` and\n        `window_end`. For a better understanding of this, see the examples\n        below.\n\n        Parameters\n        ----------\n        window_start : int\n            The start of the subset to calculate the sum relative to the\n            current value.\n\n        window_end : int\n            The end of the subset to calculate the sum relative to the current\n            value. Must be greater than `window_start`.\n\n        min_observations : int\n            Minimum number of non-missing observations in window required to\n            calculate the sum (otherwise result is None). None signifies that\n            the entire window must not include a missing value. A negative\n            number throws an error.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import pandas\n        >>> sa = SArray([1,2,3,4,5])\n        >>> series = pandas.Series([1,2,3,4,5])\n\n        A rolling sum with a window including the previous 2 entries including\n        the current:\n        >>> sa.rolling_sum(-2,0)\n        dtype: int\n        Rows: 5\n        [None, None, 6, 9, 12]\n\n        Pandas equivalent:\n        >>> pandas.rolling_sum(series, 3)\n        0   NaN\n        1   NaN\n        2     6\n        3     9\n        4    12\n        dtype: float64\n\n        Same rolling sum operation, but 2 minimum observations:\n        >>> sa.rolling_sum(-2,0,min_observations=2)\n        dtype: int\n        Rows: 5\n        [None, 3, 6, 9, 12]\n\n        Pandas equivalent:\n        >>> pandas.rolling_sum(series, 3, min_periods=2)\n        0    NaN\n        1      3\n        2      6\n        3      9\n        4     12\n        dtype: float64\n\n        A rolling sum with a size of 3, centered around the current:\n        >>> sa.rolling_sum(-1,1)\n        dtype: int\n        Rows: 5\n        [None, 6, 9, 12, None]\n\n        Pandas equivalent:\n        >>> pandas.rolling_sum(series, 3, center=True)\n        0   NaN\n        1     6\n        2     9\n        3    12\n        4   NaN\n        dtype: float64\n\n        A rolling sum with a window including the current and the 2 entries\n        following:\n        >>> sa.rolling_sum(0,2)\n        dtype: int\n        Rows: 5\n        [6, 9, 12, None, None]\n\n        A rolling sum with a window including the previous 2 entries NOT\n        including the current:\n        >>> sa.rolling_sum(-2,-1)\n        dtype: int\n        Rows: 5\n        [None, None, 3, 5, 7]\n        '
        min_observations = self.__check_min_observations(min_observations)
        agg_op = None
        if self.dtype is array.array:
            agg_op = '__builtin__vector__sum__'
        else:
            agg_op = '__builtin__sum__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_max(self, window_start, window_end, min_observations=None):
        if False:
            print('Hello World!')
        '\n        Calculate a new SArray of the maximum value of different subsets over\n        this SArray.\n\n        The subset that the maximum is calculated over is defined as an\n        inclusive range relative to the position to each value in the SArray,\n        using `window_start` and `window_end`. For a better understanding of\n        this, see the examples below.\n\n        Parameters\n        ----------\n        window_start : int\n            The start of the subset to calculate the maximum relative to the\n            current value.\n\n        window_end : int\n            The end of the subset to calculate the maximum relative to the current\n            value. Must be greater than `window_start`.\n\n        min_observations : int\n            Minimum number of non-missing observations in window required to\n            calculate the maximum (otherwise result is None). None signifies that\n            the entire window must not include a missing value. A negative\n            number throws an error.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import pandas\n        >>> sa = SArray([1,2,3,4,5])\n        >>> series = pandas.Series([1,2,3,4,5])\n\n        A rolling max with a window including the previous 2 entries including\n        the current:\n        >>> sa.rolling_max(-2,0)\n        dtype: int\n        Rows: 5\n        [None, None, 3, 4, 5]\n\n        Pandas equivalent:\n        >>> pandas.rolling_max(series, 3)\n        0   NaN\n        1   NaN\n        2     3\n        3     4\n        4     5\n        dtype: float64\n\n        Same rolling max operation, but 2 minimum observations:\n        >>> sa.rolling_max(-2,0,min_observations=2)\n        dtype: int\n        Rows: 5\n        [None, 2, 3, 4, 5]\n\n        Pandas equivalent:\n        >>> pandas.rolling_max(series, 3, min_periods=2)\n        0    NaN\n        1      2\n        2      3\n        3      4\n        4      5\n        dtype: float64\n\n        A rolling max with a size of 3, centered around the current:\n        >>> sa.rolling_max(-1,1)\n        dtype: int\n        Rows: 5\n        [None, 3, 4, 5, None]\n\n        Pandas equivalent:\n        >>> pandas.rolling_max(series, 3, center=True)\n        0   NaN\n        1     3\n        2     4\n        3     5\n        4   NaN\n        dtype: float64\n\n        A rolling max with a window including the current and the 2 entries\n        following:\n        >>> sa.rolling_max(0,2)\n        dtype: int\n        Rows: 5\n        [3, 4, 5, None, None]\n\n        A rolling max with a window including the previous 2 entries NOT\n        including the current:\n        >>> sa.rolling_max(-2,-1)\n        dtype: int\n        Rows: 5\n        [None, None, 2, 3, 4]\n        '
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__max__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_min(self, window_start, window_end, min_observations=None):
        if False:
            while True:
                i = 10
        '\n        Calculate a new SArray of the minimum value of different subsets over\n        this SArray.\n\n        The subset that the minimum is calculated over is defined as an\n        inclusive range relative to the position to each value in the SArray,\n        using `window_start` and `window_end`. For a better understanding of\n        this, see the examples below.\n\n        Parameters\n        ----------\n        window_start : int\n            The start of the subset to calculate the minimum relative to the\n            current value.\n\n        window_end : int\n            The end of the subset to calculate the minimum relative to the current\n            value. Must be greater than `window_start`.\n\n        min_observations : int\n            Minimum number of non-missing observations in window required to\n            calculate the minimum (otherwise result is None). None signifies that\n            the entire window must not include a missing value. A negative\n            number throws an error.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import pandas\n        >>> sa = SArray([1,2,3,4,5])\n        >>> series = pandas.Series([1,2,3,4,5])\n\n        A rolling min with a window including the previous 2 entries including\n        the current:\n        >>> sa.rolling_min(-2,0)\n        dtype: int\n        Rows: 5\n        [None, None, 1, 2, 3]\n\n        Pandas equivalent:\n        >>> pandas.rolling_min(series, 3)\n        0   NaN\n        1   NaN\n        2     1\n        3     2\n        4     3\n        dtype: float64\n\n        Same rolling min operation, but 2 minimum observations:\n        >>> sa.rolling_min(-2,0,min_observations=2)\n        dtype: int\n        Rows: 5\n        [None, 1, 1, 2, 3]\n\n        Pandas equivalent:\n        >>> pandas.rolling_min(series, 3, min_periods=2)\n        0    NaN\n        1      1\n        2      1\n        3      2\n        4      3\n        dtype: float64\n\n        A rolling min with a size of 3, centered around the current:\n        >>> sa.rolling_min(-1,1)\n        dtype: int\n        Rows: 5\n        [None, 1, 2, 3, None]\n\n        Pandas equivalent:\n        >>> pandas.rolling_min(series, 3, center=True)\n        0   NaN\n        1     1\n        2     2\n        3     3\n        4   NaN\n        dtype: float64\n\n        A rolling min with a window including the current and the 2 entries\n        following:\n        >>> sa.rolling_min(0,2)\n        dtype: int\n        Rows: 5\n        [1, 2, 3, None, None]\n\n        A rolling min with a window including the previous 2 entries NOT\n        including the current:\n        >>> sa.rolling_min(-2,-1)\n        dtype: int\n        Rows: 5\n        [None, None, 1, 2, 3]\n        '
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__min__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_var(self, window_start, window_end, min_observations=None):
        if False:
            i = 10
            return i + 15
        '\n        Calculate a new SArray of the variance of different subsets over this\n        SArray.\n\n        The subset that the variance is calculated over is defined as an inclusive\n        range relative to the position to each value in the SArray, using\n        `window_start` and `window_end`. For a better understanding of this,\n        see the examples below.\n\n        Parameters\n        ----------\n        window_start : int\n            The start of the subset to calculate the variance relative to the\n            current value.\n\n        window_end : int\n            The end of the subset to calculate the variance relative to the current\n            value. Must be greater than `window_start`.\n\n        min_observations : int\n            Minimum number of non-missing observations in window required to\n            calculate the variance (otherwise result is None). None signifies that\n            the entire window must not include a missing value. A negative\n            number throws an error.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import pandas\n        >>> sa = SArray([1,2,3,4,5])\n        >>> series = pandas.Series([1,2,3,4,5])\n\n        A rolling variance with a window including the previous 2 entries\n        including the current:\n        >>> sa.rolling_var(-2,0)\n        dtype: float\n        Rows: 5\n        [None, None, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666]\n\n        Pandas equivalent:\n        >>> pandas.rolling_var(series, 3, ddof=0)\n        0         NaN\n        1         NaN\n        2    0.666667\n        3    0.666667\n        4    0.666667\n        dtype: float64\n\n        Same rolling variance operation, but 2 minimum observations:\n        >>> sa.rolling_var(-2,0,min_observations=2)\n        dtype: float\n        Rows: 5\n        [None, 0.25, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666]\n\n        Pandas equivalent:\n        >>> pandas.rolling_var(series, 3, ddof=0, min_periods=2)\n        0         NaN\n        1    0.250000\n        2    0.666667\n        3    0.666667\n        4    0.666667\n        dtype: float64\n\n        A rolling variance with a size of 3, centered around the current:\n        >>> sa.rolling_var(-1,1)\n        dtype: float\n        Rows: 5\n        [None, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, None]\n\n        Pandas equivalent:\n        >>> pandas.rolling_var(series, 3, center=True)\n        0         NaN\n        1    0.666667\n        2    0.666667\n        3    0.666667\n        4         NaN\n        dtype: float64\n\n        A rolling variance with a window including the current and the 2 entries\n        following:\n        >>> sa.rolling_var(0,2)\n        dtype: float\n        Rows: 5\n        [0.6666666666666666, 0.6666666666666666, 0.6666666666666666, None, None]\n\n        A rolling variance with a window including the previous 2 entries NOT\n        including the current:\n        >>> sa.rolling_var(-2,-1)\n        dtype: float\n        Rows: 5\n        [None, None, 0.25, 0.25, 0.25]\n        '
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__var__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_stdv(self, window_start, window_end, min_observations=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate a new SArray of the standard deviation of different subsets\n        over this SArray.\n\n        The subset that the standard deviation is calculated over is defined as\n        an inclusive range relative to the position to each value in the\n        SArray, using `window_start` and `window_end`. For a better\n        understanding of this, see the examples below.\n\n        Parameters\n        ----------\n        window_start : int\n            The start of the subset to calculate the standard deviation\n            relative to the current value.\n\n        window_end : int\n            The end of the subset to calculate the standard deviation relative\n            to the current value. Must be greater than `window_start`.\n\n        min_observations : int\n            Minimum number of non-missing observations in window required to\n            calculate the standard deviation (otherwise result is None). None\n            signifies that the entire window must not include a missing value.\n            A negative number throws an error.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import pandas\n        >>> sa = SArray([1,2,3,4,5])\n        >>> series = pandas.Series([1,2,3,4,5])\n\n        A rolling standard deviation with a window including the previous 2\n        entries including the current:\n        >>> sa.rolling_stdv(-2,0)\n        dtype: float\n        Rows: 5\n        [None, None, 0.816496580927726, 0.816496580927726, 0.816496580927726]\n\n        Pandas equivalent:\n        >>> pandas.rolling_std(series, 3, ddof=0)\n        0         NaN\n        1         NaN\n        2    0.816497\n        3    0.816497\n        4    0.816497\n        dtype: float64\n\n        Same rolling standard deviation operation, but 2 minimum observations:\n        >>> sa.rolling_stdv(-2,0,min_observations=2)\n        dtype: float\n        Rows: 5\n        [None, 0.5, 0.816496580927726, 0.816496580927726, 0.816496580927726]\n\n        Pandas equivalent:\n        >>> pandas.rolling_std(series, 3, ddof=0, min_periods=2)\n        0         NaN\n        1    0.500000\n        2    0.816497\n        3    0.816497\n        4    0.816497\n        dtype: float64\n\n        A rolling standard deviation with a size of 3, centered around the\n        current:\n        >>> sa.rolling_stdv(-1,1)\n        dtype: float\n        Rows: 5\n        [None, 0.816496580927726, 0.816496580927726, 0.816496580927726, None]\n\n        Pandas equivalent:\n        >>> pandas.rolling_std(series, 3, center=True, ddof=0)\n        0         NaN\n        1    0.816497\n        2    0.816497\n        3    0.816497\n        4         NaN\n        dtype: float64\n\n        A rolling standard deviation with a window including the current and\n        the 2 entries following:\n        >>> sa.rolling_stdv(0,2)\n        dtype: float\n        Rows: 5\n        [0.816496580927726, 0.816496580927726, 0.816496580927726, None, None]\n\n        A rolling standard deviation with a window including the previous 2\n        entries NOT including the current:\n        >>> sa.rolling_stdv(-2,-1)\n        dtype: float\n        Rows: 5\n        [None, None, 0.5, 0.5, 0.5]\n        '
        min_observations = self.__check_min_observations(min_observations)
        agg_op = '__builtin__stdv__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, min_observations))

    def rolling_count(self, window_start, window_end):
        if False:
            while True:
                i = 10
        '\n        Count the number of non-NULL values of different subsets over this\n        SArray.\n\n        The subset that the count is executed on is defined as an inclusive\n        range relative to the position to each value in the SArray, using\n        `window_start` and `window_end`. For a better understanding of this,\n        see the examples below.\n\n        Parameters\n        ----------\n        window_start : int\n            The start of the subset to count relative to the current value.\n\n        window_end : int\n            The end of the subset to count relative to the current value. Must\n            be greater than `window_start`.\n\n        Returns\n        -------\n        out : SArray\n\n        Examples\n        --------\n        >>> import pandas\n        >>> sa = SArray([1,2,3,None,5])\n        >>> series = pandas.Series([1,2,3,None,5])\n\n        A rolling count with a window including the previous 2 entries including\n        the current:\n        >>> sa.rolling_count(-2,0)\n        dtype: int\n        Rows: 5\n        [1, 2, 3, 2, 2]\n\n        Pandas equivalent:\n        >>> pandas.rolling_count(series, 3)\n        0     1\n        1     2\n        2     3\n        3     2\n        4     2\n        dtype: float64\n\n        A rolling count with a size of 3, centered around the current:\n        >>> sa.rolling_count(-1,1)\n        dtype: int\n        Rows: 5\n        [2, 3, 2, 2, 1]\n\n        Pandas equivalent:\n        >>> pandas.rolling_count(series, 3, center=True)\n        0    2\n        1    3\n        2    2\n        3    2\n        4    1\n        dtype: float64\n\n        A rolling count with a window including the current and the 2 entries\n        following:\n        >>> sa.rolling_count(0,2)\n        dtype: int\n        Rows: 5\n        [3, 2, 2, 1, 1]\n\n        A rolling count with a window including the previous 2 entries NOT\n        including the current:\n        >>> sa.rolling_count(-2,-1)\n        dtype: int\n        Rows: 5\n        [0, 1, 2, 2, 1]\n        '
        agg_op = '__builtin__nonnull__count__'
        return SArray(_proxy=self.__proxy__.builtin_rolling_apply(agg_op, window_start, window_end, 0))

    def cumulative_sum(self):
        if False:
            i = 10
            return i + 15
        "\n        Return the cumulative sum of the elements in the SArray.\n\n        Returns an SArray where each element in the output corresponds to the\n        sum of all the elements preceding and including it. The SArray is\n        expected to be of numeric type (int, float), or a numeric vector type.\n\n        Returns\n        -------\n        out : sarray[int, float, array.array]\n\n        Notes\n        -----\n         - Missing values are ignored while performing the cumulative\n           aggregate operation.\n         - For SArray's of type array.array, all entries are expected to\n           be of the same size.\n\n        Examples\n        --------\n        >>> sa = SArray([1, 2, 3, 4, 5])\n        >>> sa.cumulative_sum()\n        dtype: int\n        rows: 3\n        [1, 3, 6, 10, 15]\n        "
        agg_op = '__builtin__cum_sum__'
        return SArray(_proxy=self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_mean(self):
        if False:
            return 10
        "\n        Return the cumulative mean of the elements in the SArray.\n\n        Returns an SArray where each element in the output corresponds to the\n        mean value of all the elements preceding and including it. The SArray\n        is expected to be of numeric type (int, float), or a numeric vector\n        type.\n\n        Returns\n        -------\n        out : Sarray[float, array.array]\n\n        Notes\n        -----\n         - Missing values are ignored while performing the cumulative\n           aggregate operation.\n         - For SArray's of type array.array, all entries are expected to\n           be of the same size.\n\n        Examples\n        --------\n        >>> sa = SArray([1, 2, 3, 4, 5])\n        >>> sa.cumulative_mean()\n        dtype: float\n        rows: 3\n        [1, 1.5, 2, 2.5, 3]\n        "
        agg_op = '__builtin__cum_avg__'
        return SArray(_proxy=self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_min(self):
        if False:
            print('Hello World!')
        '\n        Return the cumulative minimum value of the elements in the SArray.\n\n        Returns an SArray where each element in the output corresponds to the\n        minimum value of all the elements preceding and including it. The\n        SArray is expected to be of numeric type (int, float).\n\n        Returns\n        -------\n        out : SArray[int, float]\n\n        Notes\n        -----\n         - Missing values are ignored while performing the cumulative\n           aggregate operation.\n\n        Examples\n        --------\n        >>> sa = SArray([1, 2, 3, 4, 0])\n        >>> sa.cumulative_min()\n        dtype: int\n        rows: 3\n        [1, 1, 1, 1, 0]\n        '
        agg_op = '__builtin__cum_min__'
        return SArray(_proxy=self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_max(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the cumulative maximum value of the elements in the SArray.\n\n        Returns an SArray where each element in the output corresponds to the\n        maximum value of all the elements preceding and including it. The\n        SArray is expected to be of numeric type (int, float).\n\n        Returns\n        -------\n        out : SArray[int, float]\n\n        Notes\n        -----\n         - Missing values are ignored while performing the cumulative\n           aggregate operation.\n\n        Examples\n        --------\n        >>> sa = SArray([1, 0, 3, 4, 2])\n        >>> sa.cumulative_max()\n        dtype: int\n        rows: 3\n        [1, 1, 3, 4, 4]\n        '
        agg_op = '__builtin__cum_max__'
        return SArray(_proxy=self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_std(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the cumulative standard deviation of the elements in the SArray.\n\n        Returns an SArray where each element in the output corresponds to the\n        standard deviation of all the elements preceding and including it. The\n        SArray is expected to be of numeric type, or a numeric vector type.\n\n        Returns\n        -------\n        out : SArray[int, float]\n\n        Notes\n        -----\n         - Missing values are ignored while performing the cumulative\n           aggregate operation.\n\n        Examples\n        --------\n        >>> sa = SArray([1, 2, 3, 4, 0])\n        >>> sa.cumulative_std()\n        dtype: float\n        rows: 3\n        [0.0, 0.5, 0.816496580927726, 1.118033988749895, 1.4142135623730951]\n        '
        agg_op = '__builtin__cum_std__'
        return SArray(_proxy=self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def cumulative_var(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the cumulative variance of the elements in the SArray.\n\n        Returns an SArray where each element in the output corresponds to the\n        variance of all the elements preceding and including it. The SArray is\n        expected to be of numeric type, or a numeric vector type.\n\n        Returns\n        -------\n        out : SArray[int, float]\n\n        Notes\n        -----\n         - Missing values are ignored while performing the cumulative\n           aggregate operation.\n\n        Examples\n        --------\n        >>> sa = SArray([1, 2, 3, 4, 0])\n        >>> sa.cumulative_var()\n        dtype: float\n        rows: 3\n        [0.0, 0.25, 0.6666666666666666, 1.25, 2.0]\n        '
        agg_op = '__builtin__cum_var__'
        return SArray(_proxy=self.__proxy__.builtin_cumulative_aggregate(agg_op))

    def filter_by(self, values, exclude=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Filter an SArray by values inside an iterable object. The result is an SArray that\n        only includes (or excludes) the values in the given ``values`` :class:`~turicreate.SArray`.\n        If ``values`` is not an SArray, we attempt to convert it to one before filtering.\n\n        Parameters\n        ----------\n        values : SArray | list | numpy.ndarray | pandas.Series | str\n        The values to use to filter the SArray. The resulting SArray will\n        only include rows that have one of these values in the given\n        column.\n\n        exclude : bool\n        If True, the result SArray will contain all rows EXCEPT those that\n        have one of the ``values``.\n\n        Returns\n        -------\n        out : SArray\n        The filtered SArray.\n\n        Examples\n        --------\n        >>> sa = SArray(['dog', 'cat', 'cow', 'horse'])\n        >>> sa.filter_by(['cat', 'hamster', 'dog', 'fish', 'bird', 'snake'])\n        dtype: str\n        Rows: 2\n        ['dog', 'cat']\n\n        >>> sa.filter_by(['cat', 'hamster', 'dog', 'fish', 'bird', 'snake'], exclude=True)\n        dtype: str\n        Rows: 2\n        ['horse', 'cow']\n        "
        from .sframe import SFrame as _SFrame
        column_name = 'sarray'
        if not isinstance(values, SArray):
            if not _is_non_string_iterable(values):
                values = [values]
            values = SArray(values)
        value_sf = _SFrame()
        value_sf.add_column(values, column_name, inplace=True)
        given_type = value_sf.column_types()[0]
        existing_type = self.dtype
        sarray_sf = _SFrame()
        sarray_sf.add_column(self, column_name, inplace=True)
        if given_type != existing_type:
            raise TypeError('Type of given values does not match type of the SArray')
        value_sf = value_sf.groupby(column_name, {})
        with cython_context():
            if exclude:
                id_name = 'id'
                value_sf = value_sf.add_row_number(id_name)
                tmp = _SFrame(_proxy=sarray_sf.__proxy__.join(value_sf.__proxy__, 'left', {column_name: column_name}))
                ret_sf = tmp[tmp[id_name] == None]
                return ret_sf[column_name]
            else:
                ret_sf = _SFrame(_proxy=sarray_sf.__proxy__.join(value_sf.__proxy__, 'inner', {column_name: column_name}))
                return ret_sf[column_name]

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a shallow copy of the sarray.\n        '
        return SArray(_proxy=self.__proxy__)

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        '\n        Returns a deep copy of the sarray. As the data in an SArray is\n        immutable, this is identical to __copy__.\n        '
        return SArray(_proxy=self.__proxy__)

    def abs(self):
        if False:
            print('Hello World!')
        '\n        Returns a new SArray containing the absolute value of each element.\n\n        Examples\n        --------\n        >>> tc.SArray([1, -1, -2]).abs()\n        dtype: int\n        Rows: 3\n        [1, 1, 2]\n\n        >>> tc.SArray([-1., -2.]).abs()\n        dtype: float\n        Rows: 2\n        [1.0, 2.0]\n        '
        return abs(self)