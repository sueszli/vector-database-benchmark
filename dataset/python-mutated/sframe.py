"""
This module defines the SFrame class which provides the
ability to create, access and manipulate a remote scalable dataframe object.

SFrame acts similarly to pandas.DataFrame, but the data is completely immutable
and is stored column wise.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .._connect import main as glconnect
from .._cython.cy_flexible_type import infer_type_of_list
from .._cython.context import debug_trace as cython_context
from .._cython.cy_sframe import UnitySFrameProxy
from ..util import _is_non_string_iterable, _make_internal_url
from ..util import _infer_dbapi2_types
from ..util import _get_module_from_object, _pytype_to_printf
from ..visualization import _get_client_app_path
from .sarray import SArray, _create_sequential_sarray
from .. import aggregate
from .image import Image as _Image
from .._deps import pandas, numpy, HAS_PANDAS, HAS_NUMPY
from ..visualization import Plot
import array
from prettytable import PrettyTable
from textwrap import wrap
import datetime
import time
import itertools
import logging as _logging
import numbers
import sys
import six
import csv
from collections import Iterable as _Iterable
__all__ = ['SFrame']
__LOGGER__ = _logging.getLogger(__name__)
FOOTER_STRS = ['Note: Only the head of the SFrame is printed.', 'You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.']
LAZY_FOOTER_STRS = ['Note: Only the head of the SFrame is printed. This SFrame is lazily evaluated.', 'You can use sf.materialize() to force materialization.']
if sys.version_info.major > 2:
    long = int

def load_sframe(filename):
    if False:
        i = 10
        return i + 15
    "\n    Load an SFrame. The filename extension is used to determine the format\n    automatically. This function is particularly useful for SFrames previously\n    saved in binary format. For CSV imports the ``SFrame.read_csv`` function\n    provides greater control. If the SFrame is in binary format, ``filename`` is\n    actually a directory, created when the SFrame is saved.\n\n    Parameters\n    ----------\n    filename : string\n        Location of the file to load. Can be a local path or a remote URL.\n\n    Returns\n    -------\n    out : SFrame\n\n    See Also\n    --------\n    SFrame.save, SFrame.read_csv\n\n    Examples\n    --------\n    >>> sf = turicreate.SFrame({'id':[1,2,3], 'val':['A','B','C']})\n    >>> sf.save('my_sframe')        # 'my_sframe' is a directory\n    >>> sf_loaded = turicreate.load_sframe('my_sframe')\n    "
    sf = SFrame(data=filename)
    return sf

def _get_global_dbapi_info(dbapi_module, conn):
    if False:
        i = 10
        return i + 15
    '\n    Fetches all needed information from the top-level DBAPI module,\n    guessing at the module if it wasn\'t passed as a parameter. Returns a\n    dictionary of all the needed variables. This is put in one place to\n    make sure the error message is clear if the module "guess" is wrong.\n    '
    module_given_msg = 'The DBAPI2 module given ({0}) is missing the global\n' + "variable '{1}'. Please make sure you are supplying a module that\n" + 'conforms to the DBAPI 2.0 standard (PEP 0249).'
    module_not_given_msg = 'Hello! I gave my best effort to find the\n' + 'top-level module that the connection object you gave me came from.\n' + "I found '{0}' which doesn't have the global variable '{1}'.\n" + 'To avoid this confusion, you can pass the module as a parameter using\n' + "the 'dbapi_module' argument to either from_sql or to_sql."
    if dbapi_module is None:
        dbapi_module = _get_module_from_object(conn)
        module_given = False
    else:
        module_given = True
    module_name = dbapi_module.__name__ if hasattr(dbapi_module, '__name__') else None
    needed_vars = ['apilevel', 'paramstyle', 'Error', 'DATETIME', 'NUMBER', 'ROWID']
    ret_dict = {}
    ret_dict['module_name'] = module_name
    for i in needed_vars:
        tmp = None
        try:
            tmp = eval('dbapi_module.' + i)
        except AttributeError as e:
            if i not in ['apilevel', 'paramstyle', 'Error']:
                pass
            elif module_given:
                raise AttributeError(module_given_msg.format(module_name, i))
            else:
                raise AttributeError(module_not_given_msg.format(module_name, i))
        ret_dict[i] = tmp
    try:
        if ret_dict['apilevel'][0:3] != '2.0':
            raise NotImplementedError('Unsupported API version ' + str(ret_dict['apilevel']) + '. Only DBAPI 2.0 is supported.')
    except TypeError as e:
        e.message = "Module's 'apilevel' value is invalid."
        raise e
    acceptable_paramstyles = ['qmark', 'numeric', 'named', 'format', 'pyformat']
    try:
        if ret_dict['paramstyle'] not in acceptable_paramstyles:
            raise TypeError("Module's 'paramstyle' value is invalid.")
    except TypeError as e:
        raise TypeError("Module's 'paramstyle' value is invalid.")
    return ret_dict

def _convert_rows_to_builtin_seq(data):
    if False:
        print('Hello World!')
    if len(data) > 0 and type(data[0]) != list:
        data = [list(row) for row in data]
    return data

def _force_cast_sql_types(data, result_types, force_cast_cols):
    if False:
        return 10
    if len(force_cast_cols) == 0:
        return data
    ret_data = []
    for row in data:
        for idx in force_cast_cols:
            if row[idx] is not None and result_types[idx] != datetime.datetime:
                row[idx] = result_types[idx](row[idx])
        ret_data.append(row)
    return ret_data

class SFrame(object):
    """
    SFrame means scalable data frame. A tabular, column-mutable dataframe object that can
    scale to big data. The data in SFrame is stored column-wise, and is
    stored on persistent storage (e.g. disk) to avoid being constrained by
    memory size.  Each column in an SFrame is a size-immutable
    :class:`~turicreate.SArray`, but SFrames are mutable in that columns can be
    added and subtracted with ease.  An SFrame essentially acts as an ordered
    dict of SArrays.

    Currently, we support constructing an SFrame from the following data
    formats:

    * csv file (comma separated value)
    * sframe directory archive (A directory where an sframe was saved
      previously)
    * general text file (with csv parsing options, See :py:meth:`read_csv()`)
    * a Python dictionary
    * pandas.DataFrame
    * JSON

    and from the following sources:

    * your local file system
    * a network file system mounted locally
    * HDFS
    * Amazon S3
    * HTTP(S).

    Only basic examples of construction are covered here. For more information
    and examples, please see the `User Guide <https://apple.github.io/turicreate/docs/user
    guide/index.html#Working_with_data_Tabular_data>`_.

    Parameters
    ----------
    data : array | pandas.DataFrame | string | dict, optional
        The actual interpretation of this field is dependent on the ``format``
        parameter. If ``data`` is an array or Pandas DataFrame, the contents are
        stored in the SFrame. If ``data`` is a string, it is interpreted as a
        file. Files can be read from local file system or urls (local://,
        hdfs://, s3://, http://).

    format : string, optional
        Format of the data. The default, "auto" will automatically infer the
        input data format. The inference rules are simple: If the data is an
        array or a dataframe, it is associated with 'array' and 'dataframe'
        respectively. If the data is a string, it is interpreted as a file, and
        the file extension is used to infer the file format. The explicit
        options are:

        - "auto"
        - "array"
        - "dict"
        - "sarray"
        - "dataframe"
        - "csv"
        - "tsv"
        - "sframe".

    See Also
    --------
    read_csv:
        Create a new SFrame from a csv file. Preferred for text and CSV formats,
        because it has a lot more options for controlling the parser.

    save : Save an SFrame for later use.

    Notes
    -----
    - When reading from HDFS on Linux we must guess the location of your java
      installation. By default, we will use the location pointed to by the
      JAVA_HOME environment variable.  If this is not set, we check many common
      installation paths. You may use two environment variables to override
      this behavior.  TURI_JAVA_HOME allows you to specify a specific java
      installation and overrides JAVA_HOME.  TURI_LIBJVM_DIRECTORY
      overrides all and expects the exact directory that your preferred
      libjvm.so file is located.  Use this ONLY if you'd like to use a
      non-standard JVM.

    Examples
    --------

    >>> import turicreate
    >>> from turicreate import SFrame

    **Construction**

    Construct an SFrame from a dataframe and transfers the dataframe object
    across the network.

    >>> df = pandas.DataFrame()
    >>> sf = SFrame(data=df)

    Construct an SFrame from a local csv file (only works for local server).

    >>> sf = SFrame(data='~/mydata/foo.csv')

    Construct an SFrame from a csv file on Amazon S3. This requires the
    environment variables: *AWS_ACCESS_KEY_ID* and *AWS_SECRET_ACCESS_KEY* to be
    set before the python session started.

    >>> sf = SFrame(data='s3://mybucket/foo.csv')

    Read from HDFS using a specific java installation (environment variable
    only applies when using Linux)

    >>> import os
    >>> os.environ['TURI_JAVA_HOME'] = '/my/path/to/java'
    >>> from turicreate import SFrame
    >>> sf = SFrame("hdfs://mycluster.example.com:8020/user/myname/coolfile.txt")

    An SFrame can be constructed from a dictionary of values or SArrays:

    >>> sf = tc.SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    Or equivalently:

    >>> ids = SArray([1,2,3])
    >>> vals = SArray(['A','B','C'])
    >>> sf = SFrame({'id':ids,'val':vals})

    It can also be constructed from an array of SArrays in which case column
    names are automatically assigned.

    >>> ids = SArray([1,2,3])
    >>> vals = SArray(['A','B','C'])
    >>> sf = SFrame([ids, vals])
    >>> sf
    Columns:
        X1 int
        X2 str
    Rows: 3
    Data:
       X1  X2
    0  1   A
    1  2   B
    2  3   C

    If the SFrame is constructed from a list of values, an SFrame of a single
    column is constructed.

    >>> sf = SFrame([1,2,3])
    >>> sf
    Columns:
        X1 int
    Rows: 3
    Data:
       X1
    0  1
    1  2
    2  3

    **Parsing**

    The :py:func:`turicreate.SFrame.read_csv()` is quite powerful and, can be
    used to import a variety of row-based formats.

    First, some simple cases:

    >>> !cat ratings.csv
    user_id,movie_id,rating
    10210,1,1
    10213,2,5
    10217,2,2
    10102,1,3
    10109,3,4
    10117,5,2
    10122,2,4
    10114,1,5
    10125,1,1
    >>> tc.SFrame.read_csv('ratings.csv')
    Columns:
      user_id   int
      movie_id  int
      rating    int
    Rows: 9
    Data:
    +---------+----------+--------+
    | user_id | movie_id | rating |
    +---------+----------+--------+
    |  10210  |    1     |   1    |
    |  10213  |    2     |   5    |
    |  10217  |    2     |   2    |
    |  10102  |    1     |   3    |
    |  10109  |    3     |   4    |
    |  10117  |    5     |   2    |
    |  10122  |    2     |   4    |
    |  10114  |    1     |   5    |
    |  10125  |    1     |   1    |
    +---------+----------+--------+
    [9 rows x 3 columns]


    Delimiters can be specified, if "," is not the delimiter, for instance
    space ' ' in this case. Only single character delimiters are supported.

    >>> !cat ratings.csv
    user_id movie_id rating
    10210 1 1
    10213 2 5
    10217 2 2
    10102 1 3
    10109 3 4
    10117 5 2
    10122 2 4
    10114 1 5
    10125 1 1
    >>> tc.SFrame.read_csv('ratings.csv', delimiter=' ')

    By default, "NA" or a missing element are interpreted as missing values.

    >>> !cat ratings2.csv
    user,movie,rating
    "tom",,1
    harry,5,
    jack,2,2
    bill,,
    >>> tc.SFrame.read_csv('ratings2.csv')
    Columns:
      user  str
      movie int
      rating    int
    Rows: 4
    Data:
    +---------+-------+--------+
    |   user  | movie | rating |
    +---------+-------+--------+
    |   tom   |  None |   1    |
    |  harry  |   5   |  None  |
    |   jack  |   2   |   2    |
    | missing |  None |  None  |
    +---------+-------+--------+
    [4 rows x 3 columns]

    Furthermore due to the dictionary types and list types, can handle parsing
    of JSON-like formats.

    >>> !cat ratings3.csv
    business, categories, ratings
    "Restaurant 1", [1 4 9 10], {"funny":5, "cool":2}
    "Restaurant 2", [], {"happy":2, "sad":2}
    "Restaurant 3", [2, 11, 12], {}
    >>> tc.SFrame.read_csv('ratings3.csv')
    Columns:
    business    str
    categories  array
    ratings dict
    Rows: 3
    Data:
    +--------------+--------------------------------+-------------------------+
    |   business   |           categories           |         ratings         |
    +--------------+--------------------------------+-------------------------+
    | Restaurant 1 | array('d', [1.0, 4.0, 9.0, ... | {'funny': 5, 'cool': 2} |
    | Restaurant 2 |           array('d')           |  {'sad': 2, 'happy': 2} |
    | Restaurant 3 | array('d', [2.0, 11.0, 12.0])  |            {}           |
    +--------------+--------------------------------+-------------------------+
    [3 rows x 3 columns]

    The list and dictionary parsers are quite flexible and can absorb a
    variety of purely formatted inputs. Also, note that the list and dictionary
    types are recursive, allowing for arbitrary values to be contained.

    All these are valid lists:

    >>> !cat interesting_lists.csv
    list
    []
    [1,2,3]
    [1;2,3]
    [1 2 3]
    [{a:b}]
    ["c",d, e]
    [[a]]
    >>> tc.SFrame.read_csv('interesting_lists.csv')
    Columns:
      list  list
    Rows: 7
    Data:
    +-----------------+
    |       list      |
    +-----------------+
    |        []       |
    |    [1, 2, 3]    |
    |    [1, 2, 3]    |
    |    [1, 2, 3]    |
    |   [{'a': 'b'}]  |
    | ['c', 'd', 'e'] |
    |     [['a']]     |
    +-----------------+
    [7 rows x 1 columns]

    All these are valid dicts:

    >>> !cat interesting_dicts.csv
    dict
    {"classic":1,"dict":1}
    {space:1 separated:1}
    {emptyvalue:}
    {}
    {:}
    {recursive1:[{a:b}]}
    {:[{:[a]}]}
    >>> tc.SFrame.read_csv('interesting_dicts.csv')
    Columns:
      dict  dict
    Rows: 7
    Data:
    +------------------------------+
    |             dict             |
    +------------------------------+
    |  {'dict': 1, 'classic': 1}   |
    | {'separated': 1, 'space': 1} |
    |     {'emptyvalue': None}     |
    |              {}              |
    |         {None: None}         |
    | {'recursive1': [{'a': 'b'}]} |
    | {None: [{None: array('d')}]} |
    +------------------------------+
    [7 rows x 1 columns]

    **Saving**

    Save and load the sframe in native format.

    >>> sf.save('mysframedir')
    >>> sf2 = turicreate.load_sframe('mysframedir')

    **Column Manipulation**

    An SFrame is composed of a collection of columns of SArrays, and individual
    SArrays can be extracted easily. For instance given an SFrame:

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    The "id" column can be extracted using:

    >>> sf["id"]
    dtype: int
    Rows: 3
    [1, 2, 3]

    And can be deleted using:

    >>> del sf["id"]

    Multiple columns can be selected by passing a list of column names:

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C'],'val2':[5,6,7]})
    >>> sf
    Columns:
        id   int
        val  str
        val2 int
    Rows: 3
    Data:
       id  val val2
    0  1   A   5
    1  2   B   6
    2  3   C   7
    >>> sf2 = sf[['id','val']]
    >>> sf2
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    You can also select columns using types or a list of types:

    >>> sf2 = sf[int]
    >>> sf2
    Columns:
        id   int
        val2 int
    Rows: 3
    Data:
       id  val2
    0  1   5
    1  2   6
    2  3   7

    Or a mix of types and names:

    >>> sf2 = sf[['id', str]]
    >>> sf2
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C


    The same mechanism can be used to re-order columns:

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C
    >>> sf[['val','id']]
    >>> sf
    Columns:
        val str
        id  int
    Rows: 3
    Data:
       val id
    0  A   1
    1  B   2
    2  C   3

    **Element Access and Slicing**

    SFrames can be accessed by integer keys just like a regular python list.
    Such operations may not be fast on large datasets so looping over an SFrame
    should be avoided.

    >>> sf = SFrame({'id':[1,2,3],'val':['A','B','C']})
    >>> sf[0]
    {'id': 1, 'val': 'A'}
    >>> sf[2]
    {'id': 3, 'val': 'C'}
    >>> sf[5]
    IndexError: SFrame index out of range

    Negative indices can be used to access elements from the tail of the array

    >>> sf[-1] # returns the last element
    {'id': 3, 'val': 'C'}
    >>> sf[-2] # returns the second to last element
    {'id': 2, 'val': 'B'}

    The SFrame also supports the full range of python slicing operators:

    >>> sf[1000:] # Returns an SFrame containing rows 1000 to the end
    >>> sf[:1000] # Returns an SFrame containing rows 0 to row 999 inclusive
    >>> sf[0:1000:2] # Returns an SFrame containing rows 0 to row 1000 in steps of 2
    >>> sf[-100:] # Returns an SFrame containing last 100 rows
    >>> sf[-100:len(sf):2] # Returns an SFrame containing last 100 rows in steps of 2

    **Logical Filter**

    An SFrame can be filtered using

    >>> sframe[binary_filter]

    where sframe is an SFrame and binary_filter is an SArray of the same length.
    The result is a new SFrame which contains only rows of the SFrame where its
    matching row in the binary_filter is non zero.

    This permits the use of boolean operators that can be used to perform
    logical filtering operations. For instance, given an SFrame

    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C

    >>> sf[(sf['id'] >= 1) & (sf['id'] <= 2)]
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B

    See :class:`~turicreate.SArray` for more details on the use of the logical
    filter.

    This can also be used more generally to provide filtering capability which
    is otherwise not expressible with simple boolean functions. For instance:

    >>> sf[sf['id'].apply(lambda x: math.log(x) <= 1)]
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B

    Or alternatively:

    >>> sf[sf.apply(lambda x: math.log(x['id']) <= 1)]

            Create an SFrame from a Python dictionary.

    >>> from turicreate import SFrame
    >>> sf = SFrame({'id':[1,2,3], 'val':['A','B','C']})
    >>> sf
    Columns:
        id  int
        val str
    Rows: 3
    Data:
       id  val
    0  1   A
    1  2   B
    2  3   C
    """
    __slots__ = ['_proxy', '_cache']

    def __init__(self, data=None, format='auto', _proxy=None):
        if False:
            while True:
                i = 10
        "__init__(data=list(), format='auto')\n        Construct a new SFrame from a url or a pandas.DataFrame.\n        "
        if _proxy:
            self.__proxy__ = _proxy
        else:
            self.__proxy__ = UnitySFrameProxy()
            _format = None
            if six.PY2 and isinstance(data, unicode):
                data = data.encode('utf-8')
            if format == 'auto':
                if HAS_PANDAS and isinstance(data, pandas.DataFrame):
                    _format = 'dataframe'
                elif isinstance(data, str) or (sys.version_info.major < 3 and isinstance(data, unicode)):
                    if data.endswith(('.csv', '.csv.gz')):
                        _format = 'csv'
                    elif data.endswith(('.tsv', '.tsv.gz')):
                        _format = 'tsv'
                    elif data.endswith(('.txt', '.txt.gz')):
                        print('Assuming file is csv. For other delimiters, ' + 'please use `SFrame.read_csv`.')
                        _format = 'csv'
                    else:
                        _format = 'sframe'
                elif type(data) == SArray:
                    _format = 'sarray'
                elif isinstance(data, SFrame):
                    _format = 'sframe_obj'
                elif isinstance(data, dict):
                    _format = 'dict'
                elif _is_non_string_iterable(data):
                    _format = 'array'
                elif data is None:
                    _format = 'empty'
                else:
                    raise ValueError('Cannot infer input type for data ' + str(data))
            else:
                _format = format
            with cython_context():
                if _format == 'dataframe':
                    for c in data.columns.values:
                        self.add_column(SArray(data[c].values), str(c), inplace=True)
                elif _format == 'sframe_obj':
                    for col in data.column_names():
                        self.__proxy__.add_column(data[col].__proxy__, col)
                elif _format == 'sarray':
                    self.__proxy__.add_column(data.__proxy__, '')
                elif _format == 'array':
                    if len(data) > 0:
                        unique_types = set([type(x) for x in data if x is not None])
                        if len(unique_types) == 1 and SArray in unique_types:
                            for arr in data:
                                self.add_column(arr, inplace=True)
                        elif SArray in unique_types:
                            raise ValueError('Cannot create SFrame from mix of regular values and SArrays')
                        else:
                            self.__proxy__.add_column(SArray(data).__proxy__, '')
                elif _format == 'dict':
                    if len(set((len(value) for value in data.values()))) > 1:
                        raise RuntimeError('All column should be of the same length')
                    sarray_keys = sorted((key for (key, value) in six.iteritems(data) if isinstance(value, SArray)))
                    self.__proxy__.load_from_dataframe({key: value for (key, value) in six.iteritems(data) if not isinstance(value, SArray)})
                    for key in sarray_keys:
                        self.__proxy__.add_column(data[key].__proxy__, key)
                elif _format == 'csv':
                    url = data
                    tmpsf = SFrame.read_csv(url, delimiter=',', header=True)
                    self.__proxy__ = tmpsf.__proxy__
                elif _format == 'tsv':
                    url = data
                    tmpsf = SFrame.read_csv(url, delimiter='\t', header=True)
                    self.__proxy__ = tmpsf.__proxy__
                elif _format == 'sframe':
                    url = _make_internal_url(data)
                    self.__proxy__.load_from_sframe_index(url)
                elif _format == 'empty':
                    pass
                else:
                    raise ValueError('Unknown input type: ' + format)

    @staticmethod
    def _infer_column_types_from_lines(first_rows):
        if False:
            for i in range(10):
                print('nop')
        if len(first_rows.column_names()) < 1:
            print('Insufficient number of columns to perform type inference')
            raise RuntimeError('Insufficient columns ')
        if len(first_rows) < 1:
            print('Insufficient number of rows to perform type inference')
            raise RuntimeError('Insufficient rows')
        all_column_values_transposed = [list(first_rows[col]) for col in first_rows.column_names()]
        all_column_values = [list(x) for x in list(zip(*all_column_values_transposed))]
        all_column_type_hints = [[type(t) for t in vals] for vals in all_column_values]
        if len(set((len(x) for x in all_column_type_hints))) != 1:
            print('Unable to infer column types. Defaulting to str')
            return str
        column_type_hints = all_column_type_hints[0]
        for i in range(1, len(all_column_type_hints)):
            currow = all_column_type_hints[i]
            for j in range(len(column_type_hints)):
                d = set([currow[j], column_type_hints[j]])
                if len(d) == 1:
                    continue
                if (long in d or int in d) and float in d:
                    column_type_hints[j] = float
                elif array.array in d and list in d:
                    column_type_hints[j] = list
                elif type(None) in d:
                    if currow[j] != type(None):
                        column_type_hints[j] = currow[j]
                else:
                    column_type_hints[j] = str
        for i in range(len(column_type_hints)):
            if column_type_hints[i] == type(None):
                column_type_hints[i] = str
        return column_type_hints

    @classmethod
    def _read_csv_impl(cls, url, delimiter=',', header=True, error_bad_lines=False, comment_char='', escape_char='\\', double_quote=True, quote_char='"', skip_initial_space=True, column_type_hints=None, na_values=['NA'], line_terminator='\n', usecols=[], nrows=None, skiprows=0, verbose=True, store_errors=True, nrows_to_infer=100, true_values=[], false_values=[], _only_raw_string_substitutions=False, **kwargs):
        if False:
            return 10
        '\n        Constructs an SFrame from a CSV file or a path to multiple CSVs, and\n        returns a pair containing the SFrame and optionally\n        (if store_errors=True) a dict of filenames to SArrays\n        indicating for each file, what are the incorrectly parsed lines\n        encountered.\n\n        Parameters\n        ----------\n        store_errors : bool\n            If true, the output errors dict will be filled.\n\n        See `read_csv` for the rest of the parameters.\n        '
        if 'sep' in kwargs:
            delimiter = kwargs['sep']
            del kwargs['sep']
        if 'quotechar' in kwargs:
            quote_char = kwargs['quotechar']
            del kwargs['quotechar']
        if 'doublequote' in kwargs:
            double_quote = kwargs['doublequote']
            del kwargs['doublequote']
        if 'comment' in kwargs:
            comment_char = kwargs['comment']
            del kwargs['comment']
            if comment_char is None:
                comment_char = ''
        if 'lineterminator' in kwargs:
            line_terminator = kwargs['lineterminator']
            del kwargs['lineterminator']
        if len(kwargs) > 0:
            raise TypeError('Unexpected keyword arguments ' + str(kwargs.keys()))
        parsing_config = dict()
        parsing_config['delimiter'] = delimiter
        parsing_config['use_header'] = header
        parsing_config['continue_on_failure'] = not error_bad_lines
        parsing_config['comment_char'] = comment_char
        parsing_config['escape_char'] = '\x00' if escape_char is None else escape_char
        parsing_config['use_escape_char'] = escape_char is None
        parsing_config['double_quote'] = double_quote
        parsing_config['quote_char'] = quote_char
        parsing_config['skip_initial_space'] = skip_initial_space
        parsing_config['store_errors'] = store_errors
        parsing_config['line_terminator'] = line_terminator
        parsing_config['output_columns'] = usecols
        parsing_config['skip_rows'] = skiprows
        parsing_config['true_values'] = true_values
        parsing_config['false_values'] = false_values
        parsing_config['only_raw_string_substitutions'] = _only_raw_string_substitutions
        if type(na_values) is str:
            na_values = [na_values]
        if na_values is not None and len(na_values) > 0:
            parsing_config['na_values'] = na_values
        if nrows is not None:
            parsing_config['row_limit'] = nrows
        proxy = UnitySFrameProxy()
        internal_url = _make_internal_url(url)
        column_type_inference_was_used = False
        if column_type_hints is None:
            try:
                first_rows = SFrame.read_csv(url, nrows=nrows_to_infer, column_type_hints=type(None), header=header, delimiter=delimiter, comment_char=comment_char, escape_char=escape_char, double_quote=double_quote, quote_char=quote_char, skip_initial_space=skip_initial_space, na_values=na_values, line_terminator=line_terminator, usecols=usecols, skiprows=skiprows, verbose=verbose, true_values=true_values, false_values=false_values, _only_raw_string_substitutions=_only_raw_string_substitutions)
                column_type_hints = SFrame._infer_column_types_from_lines(first_rows)
                typelist = '[' + ','.join((t.__name__ for t in column_type_hints)) + ']'
                if verbose:
                    print('------------------------------------------------------')
                    print('Inferred types from first %d line(s) of file as ' % nrows_to_infer)
                    print('column_type_hints=' + typelist)
                    print('If parsing fails due to incorrect types, you can correct')
                    print('the inferred type list above and pass it to read_csv in')
                    print('the column_type_hints argument')
                    print('------------------------------------------------------')
                column_type_inference_was_used = True
            except RuntimeError as e:
                if type(e) == RuntimeError and ('cancel' in str(e.args[0]) or 'Cancel' in str(e.args[0])):
                    raise e
                column_type_hints = str
                if verbose:
                    print('Could not detect types. Using str for each column.')
        if type(column_type_hints) is type:
            type_hints = {'__all_columns__': column_type_hints}
        elif type(column_type_hints) is list:
            type_hints = dict(list(zip(['__X%d__' % i for i in range(len(column_type_hints))], column_type_hints)))
        elif type(column_type_hints) is dict:
            try:
                first_rows = SFrame.read_csv(url, nrows=nrows_to_infer, column_type_hints=type(None), header=header, delimiter=delimiter, comment_char=comment_char, escape_char=escape_char, double_quote=double_quote, quote_char=quote_char, skip_initial_space=skip_initial_space, na_values=na_values, line_terminator=line_terminator, usecols=usecols, skiprows=skiprows, verbose=verbose, true_values=true_values, false_values=false_values, _only_raw_string_substitutions=_only_raw_string_substitutions)
                inferred_types = SFrame._infer_column_types_from_lines(first_rows)
                inferred_types = dict(list(zip(first_rows.column_names(), inferred_types)))
                for key in column_type_hints:
                    inferred_types[key] = column_type_hints[key]
                column_type_hints = inferred_types
            except RuntimeError as e:
                if type(e) == RuntimeError and ('cancel' in str(e) or 'Cancel' in str(e)):
                    raise e
                if verbose:
                    print('Could not detect types. Using str for all unspecified columns.')
            type_hints = column_type_hints
        else:
            raise TypeError('Invalid type for column_type_hints. Must be a dictionary, list or a single type.')
        try:
            if not verbose:
                glconnect.get_server().set_log_progress(False)
            with cython_context():
                errors = proxy.load_from_csvs(internal_url, parsing_config, type_hints)
        except Exception as e:
            if type(e) == RuntimeError and 'CSV parsing cancelled' in str(e.args[0]):
                raise e
            if column_type_inference_was_used:
                if verbose:
                    print('Unable to parse the file with automatic type inference.')
                    print('Defaulting to column_type_hints=str')
                type_hints = {'__all_columns__': str}
                try:
                    with cython_context():
                        errors = proxy.load_from_csvs(internal_url, parsing_config, type_hints)
                except:
                    glconnect.get_server().set_log_progress(True)
                    raise
            else:
                glconnect.get_server().set_log_progress(True)
                raise
        glconnect.get_server().set_log_progress(True)
        return (cls(_proxy=proxy), {f: SArray(_proxy=es) for (f, es) in errors.items()})

    @classmethod
    def read_csv_with_errors(cls, url, delimiter=',', header=True, comment_char='', escape_char='\\', double_quote=True, quote_char='"', skip_initial_space=True, column_type_hints=None, na_values=['NA'], line_terminator='\n', usecols=[], nrows=None, skiprows=0, verbose=True, nrows_to_infer=100, true_values=[], false_values=[], _only_raw_string_substitutions=False, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Constructs an SFrame from a CSV file or a path to multiple CSVs, and\n        returns a pair containing the SFrame and a dict of filenames to SArrays\n        indicating for each file, what are the incorrectly parsed lines\n        encountered.\n\n        Parameters\n        ----------\n        url : string\n            Location of the CSV file or directory to load. If URL is a directory\n            or a "glob" pattern, all matching files will be loaded.\n\n        delimiter : string, optional\n            This describes the delimiter used for parsing csv files.\n\n        header : bool, optional\n            If true, uses the first row as the column names. Otherwise use the\n            default column names: \'X1, X2, ...\'.\n\n        comment_char : string, optional\n            The character which denotes that the\n            remainder of the line is a comment.\n\n        escape_char : string, optional\n            Character which begins a C escape sequence. Defaults to backslash(\\)\n            Set to None to disable.\n\n        double_quote : bool, optional\n            If True, two consecutive quotes in a string are parsed to a single\n            quote.\n\n        quote_char : string, optional\n            Character sequence that indicates a quote.\n\n        skip_initial_space : bool, optional\n            Ignore extra spaces at the start of a field\n\n        column_type_hints : None, type, list[type], dict[string, type], optional\n            This provides type hints for each column. By default, this method\n            attempts to detect the type of each column automatically.\n\n            Supported types are int, float, str, list, dict, and array.array.\n\n            * If a single type is provided, the type will be\n              applied to all columns. For instance, column_type_hints=float\n              will force all columns to be parsed as float.\n            * If a list of types is provided, the types applies\n              to each column in order, e.g.[int, float, str]\n              will parse the first column as int, second as float and third as\n              string.\n            * If a dictionary of column name to type is provided,\n              each type value in the dictionary is applied to the key it\n              belongs to.\n              For instance {\'user\':int} will hint that the column called "user"\n              should be parsed as an integer, and the rest will be type inferred.\n\n        na_values : str | list of str, optional\n            A string or list of strings to be interpreted as missing values.\n\n        true_values : str | list of str, optional\n            A string or list of strings to be interpreted as 1\n\n        false_values : str | list of str, optional\n            A string or list of strings to be interpreted as 0\n\n        line_terminator : str, optional\n            A string to be interpreted as the line terminator. Defaults to "\\n"\n            which will also correctly match Mac, Linux and Windows line endings\n            ("\\r", "\\n" and "\\r\\n" respectively)\n\n        usecols : list of str, optional\n            A subset of column names to output. If unspecified (default),\n            all columns will be read. This can provide performance gains if the\n            number of columns are large. If the input file has no headers,\n            usecols=[\'X1\',\'X3\'] will read columns 1 and 3.\n\n        nrows : int, optional\n            If set, only this many rows will be read from the file.\n\n        skiprows : int, optional\n            If set, this number of rows at the start of the file are skipped.\n\n        verbose : bool, optional\n            If True, print the progress.\n\n        Returns\n        -------\n        out : tuple\n            The first element is the SFrame with good data. The second element\n            is a dictionary of filenames to SArrays indicating for each file,\n            what are the incorrectly parsed lines encountered.\n\n        See Also\n        --------\n        read_csv, SFrame\n\n        Examples\n        --------\n        >>> bad_url = \'https://static.turi.com/datasets/bad_csv_example.csv\'\n        >>> (sf, bad_lines) = turicreate.SFrame.read_csv_with_errors(bad_url)\n        >>> sf\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        +---------+----------+--------+\n        |  25904  |   1663   |   3    |\n        |  25907  |   1663   |   3    |\n        |  25923  |   1663   |   3    |\n        |  25924  |   1663   |   3    |\n        |  25928  |   1663   |   2    |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [98 rows x 3 columns]\n\n        >>> bad_lines\n        {\'https://static.turi.com/datasets/bad_csv_example.csv\': dtype: str\n         Rows: 1\n         [\'x,y,z,a,b,c\']}\n       '
        return cls._read_csv_impl(url, delimiter=delimiter, header=header, error_bad_lines=False, comment_char=comment_char, escape_char=escape_char, double_quote=double_quote, quote_char=quote_char, skip_initial_space=skip_initial_space, column_type_hints=column_type_hints, na_values=na_values, line_terminator=line_terminator, usecols=usecols, nrows=nrows, verbose=verbose, skiprows=skiprows, store_errors=True, nrows_to_infer=nrows_to_infer, true_values=true_values, false_values=false_values, _only_raw_string_substitutions=_only_raw_string_substitutions, **kwargs)

    @classmethod
    def read_csv(cls, url, delimiter=',', header=True, error_bad_lines=False, comment_char='', escape_char='\\', double_quote=True, quote_char='"', skip_initial_space=True, column_type_hints=None, na_values=['NA'], line_terminator='\n', usecols=[], nrows=None, skiprows=0, verbose=True, nrows_to_infer=100, true_values=[], false_values=[], _only_raw_string_substitutions=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructs an SFrame from a CSV file or a path to multiple CSVs.\n\n        Parameters\n        ----------\n        url : string\n            Location of the CSV file or directory to load. If URL is a directory\n            or a "glob" pattern, all matching files will be loaded.\n\n        delimiter : string, optional\n            This describes the delimiter used for parsing csv files.\n\n        header : bool, optional\n            If true, uses the first row as the column names. Otherwise use the\n            default column names : \'X1, X2, ...\'.\n\n        error_bad_lines : bool\n            If true, will fail upon encountering a bad line. If false, will\n            continue parsing skipping lines which fail to parse correctly.\n            A sample of the first 10 encountered bad lines will be printed.\n\n        comment_char : string, optional\n            The character which denotes that the remainder of the line is a\n            comment.\n\n        escape_char : string, optional\n            Character which begins a C escape sequence. Defaults to backslash(\\)\n            Set to None to disable.\n\n        double_quote : bool, optional\n            If True, two consecutive quotes in a string are parsed to a single\n            quote.\n\n        quote_char : string, optional\n            Character sequence that indicates a quote.\n\n        skip_initial_space : bool, optional\n            Ignore extra spaces at the start of a field\n\n        column_type_hints : None, type, list[type], dict[string, type], optional\n            This provides type hints for each column. By default, this method\n            attempts to detect the type of each column automatically.\n\n            Supported types are int, float, str, list, dict, and array.array.\n\n            * If a single type is provided, the type will be\n              applied to all columns. For instance, column_type_hints=float\n              will force all columns to be parsed as float.\n            * If a list of types is provided, the types applies\n              to each column in order, e.g.[int, float, str]\n              will parse the first column as int, second as float and third as\n              string.\n            * If a dictionary of column name to type is provided,\n              each type value in the dictionary is applied to the key it\n              belongs to.\n              For instance {\'user\':int} will hint that the column called "user"\n              should be parsed as an integer, and the rest will be type inferred.\n\n        na_values : str | list of str, optional\n            A string or list of strings to be interpreted as missing values.\n\n        true_values : str | list of str, optional\n            A string or list of strings to be interpreted as 1\n\n        false_values : str | list of str, optional\n            A string or list of strings to be interpreted as 0\n\n\n        line_terminator : str, optional\n            A string to be interpreted as the line terminator. Defaults to "\n"\n            which will also correctly match Mac, Linux and Windows line endings\n            ("\\r", "\\n" and "\\r\\n" respectively)\n\n        usecols : list of str, optional\n            A subset of column names to output. If unspecified (default),\n            all columns will be read. This can provide performance gains if the\n            number of columns are large. If the input file has no headers,\n            usecols=[\'X1\',\'X3\'] will read columns 1 and 3.\n\n        nrows : int, optional\n            If set, only this many rows will be read from the file.\n\n        skiprows : int, optional\n            If set, this number of rows at the start of the file are skipped.\n\n        verbose : bool, optional\n            If True, print the progress.\n\n        nrows_to_infer : integer\n            The number of rows used to infer column types.\n\n        Returns\n        -------\n        out : SFrame\n\n        See Also\n        --------\n        read_csv_with_errors, SFrame\n\n        Examples\n        --------\n\n        Read a regular csv file, with all default options, automatically\n        determine types:\n\n        >>> url = \'https://static.turi.com/datasets/rating_data_example.csv\'\n        >>> sf = turicreate.SFrame.read_csv(url)\n        >>> sf\n        Columns:\n          user_id int\n          movie_id  int\n          rating  int\n        Rows: 10000\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        +---------+----------+--------+\n        |  25904  |   1663   |   3    |\n        |  25907  |   1663   |   3    |\n        |  25923  |   1663   |   3    |\n        |  25924  |   1663   |   3    |\n        |  25928  |   1663   |   2    |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [10000 rows x 3 columns]\n\n        Read only the first 100 lines of the csv file:\n\n        >>> sf = turicreate.SFrame.read_csv(url, nrows=100)\n        >>> sf\n        Columns:\n          user_id int\n          movie_id  int\n          rating  int\n        Rows: 100\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        +---------+----------+--------+\n        |  25904  |   1663   |   3    |\n        |  25907  |   1663   |   3    |\n        |  25923  |   1663   |   3    |\n        |  25924  |   1663   |   3    |\n        |  25928  |   1663   |   2    |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [100 rows x 3 columns]\n\n        Read all columns as str type\n\n        >>> sf = turicreate.SFrame.read_csv(url, column_type_hints=str)\n        >>> sf\n        Columns:\n          user_id  str\n          movie_id  str\n          rating  str\n        Rows: 10000\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        +---------+----------+--------+\n        |  25904  |   1663   |   3    |\n        |  25907  |   1663   |   3    |\n        |  25923  |   1663   |   3    |\n        |  25924  |   1663   |   3    |\n        |  25928  |   1663   |   2    |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [10000 rows x 3 columns]\n\n        Specify types for a subset of columns and leave the rest to be str.\n\n        >>> sf = turicreate.SFrame.read_csv(url,\n        ...                               column_type_hints={\n        ...                               \'user_id\':int, \'rating\':float\n        ...                               })\n        >>> sf\n        Columns:\n          user_id str\n          movie_id  str\n          rating  float\n        Rows: 10000\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        +---------+----------+--------+\n        |  25904  |   1663   |  3.0   |\n        |  25907  |   1663   |  3.0   |\n        |  25923  |   1663   |  3.0   |\n        |  25924  |   1663   |  3.0   |\n        |  25928  |   1663   |  2.0   |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [10000 rows x 3 columns]\n\n        Not treat first line as header:\n\n        >>> sf = turicreate.SFrame.read_csv(url, header=False)\n        >>> sf\n        Columns:\n          X1  str\n          X2  str\n          X3  str\n        Rows: 10001\n        +---------+----------+--------+\n        |    X1   |    X2    |   X3   |\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        |  25904  |   1663   |   3    |\n        |  25907  |   1663   |   3    |\n        |  25923  |   1663   |   3    |\n        |  25924  |   1663   |   3    |\n        |  25928  |   1663   |   2    |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [10001 rows x 3 columns]\n\n        Treat \'3\' as missing value:\n\n        >>> sf = turicreate.SFrame.read_csv(url, na_values=[\'3\'], column_type_hints=str)\n        >>> sf\n        Columns:\n          user_id str\n          movie_id  str\n          rating  str\n        Rows: 10000\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        +---------+----------+--------+\n        |  25904  |   1663   |  None  |\n        |  25907  |   1663   |  None  |\n        |  25923  |   1663   |  None  |\n        |  25924  |   1663   |  None  |\n        |  25928  |   1663   |   2    |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [10000 rows x 3 columns]\n\n        Throw error on parse failure:\n\n        >>> bad_url = \'https://static.turi.com/datasets/bad_csv_example.csv\'\n        >>> sf = turicreate.SFrame.read_csv(bad_url, error_bad_lines=True)\n        RuntimeError: Runtime Exception. Unable to parse line "x,y,z,a,b,c"\n        Set error_bad_lines=False to skip bad lines\n        '
        return cls._read_csv_impl(url, delimiter=delimiter, header=header, error_bad_lines=error_bad_lines, comment_char=comment_char, escape_char=escape_char, double_quote=double_quote, quote_char=quote_char, skip_initial_space=skip_initial_space, column_type_hints=column_type_hints, na_values=na_values, line_terminator=line_terminator, usecols=usecols, nrows=nrows, skiprows=skiprows, verbose=verbose, store_errors=False, nrows_to_infer=nrows_to_infer, true_values=true_values, false_values=false_values, _only_raw_string_substitutions=_only_raw_string_substitutions, **kwargs)[0]

    @classmethod
    def read_json(cls, url, orient='records'):
        if False:
            return 10
        '\n        Reads a JSON file representing a table into an SFrame.\n\n        Parameters\n        ----------\n        url : string\n            Location of the CSV file or directory to load. If URL is a directory\n            or a "glob" pattern, all matching files will be loaded.\n\n        orient : string, optional. Either "records" or "lines"\n            If orient="records" the file is expected to contain a single JSON\n            array, where each array element is a dictionary. If orient="lines",\n            the file is expected to contain a JSON element per line.\n\n        Examples\n        --------\n        The orient parameter describes the expected input format of the JSON\n        file.\n\n        If orient="records", the JSON file is expected to contain a single\n        JSON Array where each array element is a dictionary describing the row.\n        For instance:\n\n        >>> !cat input.json\n        [{\'a\':1,\'b\':1}, {\'a\':2,\'b\':2}, {\'a\':3,\'b\':3}]\n        >>> SFrame.read_json(\'input.json\', orient=\'records\')\n        Columns:\n                a\tint\n                b\tint\n        Rows: 3\n        Data:\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | 1 |\n        | 2 | 2 |\n        | 3 | 3 |\n        +---+---+\n\n        If orient="lines", the JSON file is expected to contain a JSON element\n        per line. If each line contains a dictionary, it is automatically\n        unpacked.\n\n        >>> !cat input.json\n        {\'a\':1,\'b\':1}\n        {\'a\':2,\'b\':2}\n        {\'a\':3,\'b\':3}\n        >>> g = SFrame.read_json(\'input.json\', orient=\'lines\')\n        Columns:\n                a\tint\n                b\tint\n        Rows: 3\n        Data:\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | 1 |\n        | 2 | 2 |\n        | 3 | 3 |\n        +---+---+\n\n        If the lines are not dictionaries, the original format is maintained.\n\n        >>> !cat input.json\n        [\'a\',\'b\',\'c\']\n        [\'d\',\'e\',\'f\']\n        [\'g\',\'h\',\'i\']\n        [1,2,3]\n        >>> g = SFrame.read_json(\'input.json\', orient=\'lines\')\n        Columns:\n                X1\tlist\n        Rows: 3\n        Data:\n        +-----------+\n        |     X1    |\n        +-----------+\n        | [a, b, c] |\n        | [d, e, f] |\n        | [g, h, i] |\n        +-----------+\n        [3 rows x 1 columns]\n        '
        if orient == 'records':
            g = SArray.read_json(url)
            if len(g) == 0:
                return SFrame()
            if g.dtype != dict:
                raise RuntimeError('Invalid input JSON format. Expected list of dictionaries')
            g = SFrame({'X1': g})
            return g.unpack('X1', '')
        elif orient == 'lines':
            g = cls.read_csv(url, header=False, na_values=['null'], true_values=['true'], false_values=['false'], _only_raw_string_substitutions=True)
            if g.num_rows() == 0:
                return SFrame()
            if g.num_columns() != 1:
                raise RuntimeError('Input JSON not of expected format')
            if g['X1'].dtype == dict:
                return g.unpack('X1', '')
            else:
                return g
        else:
            raise ValueError('Invalid value for orient parameter (' + str(orient) + ')')

    @classmethod
    def from_sql(cls, conn, sql_statement, params=None, type_inference_rows=100, dbapi_module=None, column_type_hints=None, cursor_arraysize=128):
        if False:
            i = 10
            return i + 15
        '\n        Convert the result of a SQL database query to an SFrame.\n\n        Parameters\n        ----------\n        conn : dbapi2.Connection\n          A DBAPI2 connection object. Any connection object originating from\n          the \'connect\' method of a DBAPI2-compliant package can be used.\n\n        sql_statement : str\n          The query to be sent to the database through the given connection.\n          No checks are performed on the `sql_statement`. Any side effects from\n          the query will be reflected on the database.  If no result rows are\n          returned, an empty SFrame is created.\n\n        params : iterable | dict, optional\n          Parameters to substitute for any parameter markers in the\n          `sql_statement`. Be aware that the style of parameters may vary\n          between different DBAPI2 packages.\n\n        type_inference_rows : int, optional\n          The maximum number of rows to use for determining the column types of\n          the SFrame. These rows are held in Python until all column types are\n          determined or the maximum is reached.\n\n        dbapi_module : module | package, optional\n          The top-level DBAPI2 module/package that constructed the given\n          connection object. By default, a best guess of which module the\n          connection came from is made. In the event that this guess is wrong,\n          this will need to be specified.\n\n        column_type_hints : dict | list | type, optional\n          Specifies the types of the output SFrame. If a dict is given, it must\n          have result column names as keys, but need not have all of the result\n          column names. If a list is given, the length of the list must match\n          the number of result columns. If a single type is given, all columns\n          in the output SFrame will be this type. If the result type is\n          incompatible with the types given in this argument, a casting error\n          will occur.\n\n        cursor_arraysize : int, optional\n          The number of rows to fetch from the database at one time.\n\n        Returns\n        -------\n        out : SFrame\n\n        Examples\n        --------\n        >>> import sqlite3\n\n        >>> conn = sqlite3.connect(\'example.db\')\n\n        >>> turicreate.SFrame.from_sql(conn, "SELECT * FROM foo")\n        Columns:\n                a       int\n                b       int\n        Rows: 1\n        Data:\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | 2 |\n        +---+---+\n        [1 rows x 2 columns]\n        '
        mod_info = _get_global_dbapi_info(dbapi_module, conn)
        from .sframe_builder import SFrameBuilder
        c = conn.cursor()
        try:
            if params is None:
                c.execute(sql_statement)
            else:
                c.execute(sql_statement, params)
        except mod_info['Error'] as e:
            if hasattr(conn, 'rollback'):
                conn.rollback()
            raise e
        c.arraysize = cursor_arraysize
        result_desc = c.description
        result_names = [i[0] for i in result_desc]
        result_types = [None for i in result_desc]
        cols_to_force_cast = set()
        temp_vals = []
        col_name_to_num = {result_names[i]: i for i in range(len(result_names))}
        if column_type_hints is not None:
            if type(column_type_hints) is dict:
                for (k, v) in column_type_hints.items():
                    col_num = col_name_to_num[k]
                    cols_to_force_cast.add(col_num)
                    result_types[col_num] = v
            elif type(column_type_hints) is list:
                if len(column_type_hints) != len(result_names):
                    __LOGGER__.warn('If column_type_hints is specified as a ' + 'list, it must be of the same size as the result ' + "set's number of columns. Ignoring (use dict instead).")
                else:
                    result_types = column_type_hints
                    cols_to_force_cast.update(range(len(result_desc)))
            elif type(column_type_hints) is type:
                result_types = [column_type_hints for i in result_desc]
                cols_to_force_cast.update(range(len(result_desc)))
        hintable_types = [int, float, str]
        if not all([i in hintable_types or i is None for i in result_types]):
            raise TypeError('Only ' + str(hintable_types) + ' can be provided as type hints!')
        if not all(result_types):
            try:
                row = c.fetchone()
            except mod_info['Error'] as e:
                if hasattr(conn, 'rollback'):
                    conn.rollback()
                raise e
            while row is not None:
                temp_vals.append(row)
                val_count = 0
                for val in row:
                    if result_types[val_count] is None and val is not None:
                        result_types[val_count] = type(val)
                    val_count += 1
                if all(result_types) or len(temp_vals) >= type_inference_rows:
                    break
                row = c.fetchone()
        if not all(result_types):
            missing_val_cols = [i for (i, v) in enumerate(result_types) if v is None]
            cols_to_force_cast.update(missing_val_cols)
            inferred_types = _infer_dbapi2_types(c, mod_info)
            cnt = 0
            for i in result_types:
                if i is None:
                    result_types[cnt] = inferred_types[cnt]
                cnt += 1
        sb = SFrameBuilder(result_types, column_names=result_names)
        unsupported_cols = [i for (i, v) in enumerate(sb.column_types()) if v is type(None)]
        if len(unsupported_cols) > 0:
            cols_to_force_cast.update(unsupported_cols)
            for i in unsupported_cols:
                result_types[i] = str
            sb = SFrameBuilder(result_types, column_names=result_names)
        temp_vals = _convert_rows_to_builtin_seq(temp_vals)
        sb.append_multiple(_force_cast_sql_types(temp_vals, result_types, cols_to_force_cast))
        rows = c.fetchmany()
        while len(rows) > 0:
            rows = _convert_rows_to_builtin_seq(rows)
            sb.append_multiple(_force_cast_sql_types(rows, result_types, cols_to_force_cast))
            rows = c.fetchmany()
        cls = sb.close()
        try:
            c.close()
        except mod_info['Error'] as e:
            if hasattr(conn, 'rollback'):
                conn.rollback()
            raise e
        return cls

    def to_sql(self, conn, table_name, dbapi_module=None, use_python_type_specifiers=False, use_exact_column_names=True):
        if False:
            return 10
        "\n        Convert an SFrame to a single table in a SQL database.\n\n        This function does not attempt to create the table or check if a table\n        named `table_name` exists in the database. It simply assumes that\n        `table_name` exists in the database and appends to it.\n\n        `to_sql` can be thought of as a convenience wrapper around\n        parameterized SQL insert statements.\n\n        Parameters\n        ----------\n        conn : dbapi2.Connection\n          A DBAPI2 connection object. Any connection object originating from\n          the 'connect' method of a DBAPI2-compliant package can be used.\n\n        table_name : str\n          The name of the table to append the data in this SFrame.\n\n        dbapi_module : module | package, optional\n          The top-level DBAPI2 module/package that constructed the given\n          connection object. By default, a best guess of which module the\n          connection came from is made. In the event that this guess is wrong,\n          this will need to be specified.\n\n        use_python_type_specifiers : bool, optional\n          If the DBAPI2 module's parameter marker style is 'format' or\n          'pyformat', attempt to use accurate type specifiers for each value\n          ('s' for string, 'd' for integer, etc.). Many DBAPI2 modules simply\n          use 's' for all types if they use these parameter markers, so this is\n          False by default.\n\n        use_exact_column_names : bool, optional\n          Specify the column names of the SFrame when inserting its contents\n          into the DB. If the specified table does not have the exact same\n          column names as the SFrame, inserting the data will fail. If False,\n          the columns in the SFrame are inserted in order without care of the\n          schema of the DB table. True by default.\n        "
        mod_info = _get_global_dbapi_info(dbapi_module, conn)
        c = conn.cursor()
        col_info = list(zip(self.column_names(), self.column_types()))
        if not use_python_type_specifiers:
            _pytype_to_printf = lambda x: 's'
        sql_param = {'qmark': lambda name, col_num, col_type: '?', 'numeric': lambda name, col_num, col_type: ':' + str(col_num + 1), 'named': lambda name, col_num, col_type: ':' + str(name), 'format': lambda name, col_num, col_type: '%' + _pytype_to_printf(col_type), 'pyformat': lambda name, col_num, col_type: '%(' + str(name) + ')' + _pytype_to_printf(col_type)}
        get_sql_param = sql_param[mod_info['paramstyle']]
        ins_str = 'INSERT INTO ' + str(table_name)
        value_str = ' VALUES ('
        col_str = ' ('
        count = 0
        for i in col_info:
            col_str += i[0]
            value_str += get_sql_param(i[0], count, i[1])
            if count < len(col_info) - 1:
                col_str += ','
                value_str += ','
            count += 1
        col_str += ')'
        value_str += ')'
        if use_exact_column_names:
            ins_str += col_str
        ins_str += value_str
        if mod_info['paramstyle'] == 'named' or mod_info['paramstyle'] == 'pyformat':
            prepare_sf_row = lambda x: x
        else:
            col_names = self.column_names()
            prepare_sf_row = lambda x: [x[i] for i in col_names]
        for i in self:
            try:
                c.execute(ins_str, prepare_sf_row(i))
            except mod_info['Error'] as e:
                if hasattr(conn, 'rollback'):
                    conn.rollback()
                raise e
        conn.commit()
        c.close()

    def __hash__(self):
        if False:
            print('Hello World!')
        '\n        Because we override `__eq__` we need to implement this function in Python 3.\n        Just make it match default behavior in Python 2.\n        '
        return id(self) // 16

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return append one frames to other\n        '
        self = self.append(other)
        return self

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a string description of the frame\n        '
        ret = self.__get_column_description__()
        (is_empty, data_str) = self.__str_impl__()
        if is_empty:
            data_str = '\t[]'
        if self.__has_size__():
            ret = ret + 'Rows: ' + str(len(self)) + '\n\n'
        else:
            ret = ret + 'Rows: Unknown' + '\n\n'
        ret = ret + 'Data:\n'
        ret = ret + data_str
        return ret

    def __get_column_description__(self):
        if False:
            for i in range(10):
                print('nop')
        colnames = self.column_names()
        coltypes = self.column_types()
        ret = 'Columns:\n'
        if len(colnames) > 0:
            for i in range(len(colnames)):
                ret = ret + '\t' + colnames[i] + '\t' + coltypes[i].__name__ + '\n'
            ret = ret + '\n'
        else:
            ret = ret + '\tNone\n\n'
        return ret

    def __get_pretty_tables__(self, wrap_text=False, max_row_width=80, max_column_width=30, max_columns=20, max_rows_to_display=60):
        if False:
            return 10
        '\n        Returns a list of pretty print tables representing the current SFrame.\n        If the number of columns is larger than max_columns, the last pretty\n        table will contain an extra column of "...".\n        Parameters\n        ----------\n        wrap_text : bool, optional\n        max_row_width : int, optional\n            Max number of characters per table.\n        max_column_width : int, optional\n            Max number of characters per column.\n        max_columns : int, optional\n            Max number of columns per table.\n        max_rows_to_display : int, optional\n            Max number of rows to display.\n        Returns\n        -------\n        out : list[PrettyTable]\n        '
        if len(self) <= max_rows_to_display:
            headsf = self.__copy__()
        else:
            headsf = self.head(max_rows_to_display)
        if headsf.shape == (0, 0):
            return [PrettyTable()]
        for col in headsf.column_names():
            if headsf[col].dtype is array.array:
                headsf[col] = headsf[col].astype(list)

        def _value_to_str(value):
            if False:
                for i in range(10):
                    print('nop')
            if type(value) is array.array:
                return str(list(value))
            elif type(value) is numpy.ndarray:
                return str(value).replace('\n', ' ')
            elif type(value) is list:
                return '[' + ', '.join((_value_to_str(x) for x in value)) + ']'
            else:
                return str(value)

        def _escape_space(s):
            if False:
                for i in range(10):
                    print('nop')
            if sys.version_info.major == 3:
                return ''.join([ch.encode('unicode_escape').decode() if ch.isspace() else ch for ch in s])
            return ''.join([ch.encode('string_escape') if ch.isspace() else ch for ch in s])

        def _truncate_respect_unicode(s, max_length):
            if False:
                while True:
                    i = 10
            if len(s) <= max_length:
                return s
            elif sys.version_info.major < 3:
                u = unicode(s, 'utf-8', errors='replace')
                return u[:max_length].encode('utf-8')
            else:
                return s[:max_length]

        def _truncate_str(s, wrap_str=False):
            if False:
                while True:
                    i = 10
            '\n            Truncate and optionally wrap the input string as unicode, replace\n            unconvertible character with a diamond ?.\n            '
            s = _escape_space(s)
            if len(s) <= max_column_width:
                if sys.version_info.major < 3:
                    return unicode(s, 'utf-8', errors='replace')
                else:
                    return s
            else:
                ret = ''
                if wrap_str:
                    wrapped_lines = wrap(s, max_column_width)
                    if len(wrapped_lines) == 1:
                        return wrapped_lines[0]
                    last_line = wrapped_lines[1]
                    if len(last_line) >= max_column_width:
                        last_line = _truncate_respect_unicode(last_line, max_column_width - 4)
                    ret = wrapped_lines[0] + '\n' + last_line + ' ...'
                else:
                    ret = _truncate_respect_unicode(s, max_column_width - 4) + '...'
                if sys.version_info.major < 3:
                    return unicode(ret, 'utf-8', errors='replace')
                else:
                    return ret
        columns = self.column_names()[:max_columns]
        columns.reverse()
        num_column_of_last_table = 0
        row_of_tables = []
        while len(columns) > 0:
            tbl = PrettyTable()
            table_width = 0
            num_column_of_last_table = 0
            while len(columns) > 0:
                col = columns.pop()
                if len(headsf) > 0:
                    col_width = min(max_column_width, max((len(str(x)) for x in headsf[col])))
                else:
                    col_width = max_column_width
                if table_width + col_width < max_row_width:
                    header = _truncate_str(col, wrap_text)
                    tbl.add_column(header, [_truncate_str(_value_to_str(x), wrap_text) for x in headsf[col]])
                    table_width = str(tbl).find('\n')
                    num_column_of_last_table += 1
                else:
                    columns.append(col)
                    break
            tbl.align = 'c'
            row_of_tables.append(tbl)
        if self.num_columns() > max_columns:
            row_of_tables[-1].add_column('...', ['...'] * len(headsf))
            num_column_of_last_table += 1
        if self.__has_size__() and self.num_rows() > headsf.num_rows():
            row_of_tables[-1].add_row(['...'] * num_column_of_last_table)
        return row_of_tables

    def print_rows(self, num_rows=10, num_columns=40, max_column_width=30, max_row_width=80, output_file=None):
        if False:
            i = 10
            return i + 15
        '\n        Print the first M rows and N columns of the SFrame in human readable\n        format.\n\n        Parameters\n        ----------\n        num_rows : int, optional\n            Number of rows to print.\n\n        num_columns : int, optional\n            Number of columns to print.\n\n        max_column_width : int, optional\n            Maximum width of a column. Columns use fewer characters if possible.\n\n        max_row_width : int, optional\n            Maximum width of a printed row. Columns beyond this width wrap to a\n            new line. `max_row_width` is automatically reset to be the\n            larger of itself and `max_column_width`.\n\n        output_file: file, optional\n            The stream or file that receives the output. By default the output\n            goes to sys.stdout, but it can also be redirected to a file or a\n            string (using an object of type StringIO).\n\n        See Also\n        --------\n        head, tail\n        '
        if output_file is None:
            output_file = sys.stdout
        max_row_width = max(max_row_width, max_column_width + 1)
        printed_sf = self._imagecols_to_stringcols(num_rows)
        row_of_tables = printed_sf.__get_pretty_tables__(wrap_text=False, max_rows_to_display=num_rows, max_columns=num_columns, max_column_width=max_column_width, max_row_width=max_row_width)
        footer = '[%d rows x %d columns]\n' % self.shape
        print('\n'.join([str(tb) for tb in row_of_tables]) + '\n' + footer, file=output_file)

    def _imagecols_to_stringcols(self, num_rows=10):
        if False:
            return 10
        types = self.column_types()
        names = self.column_names()
        image_column_names = [names[i] for i in range(len(names)) if types[i] == _Image]
        printed_sf = self.__copy__()
        if len(image_column_names) > 0:
            for t in names:
                if t in image_column_names:
                    printed_sf[t] = self[t].astype(str)
        return printed_sf.head(num_rows)

    def drop_duplicates(self, subset):
        if False:
            return 10
        '\n        Returns an SFrame with duplicate rows removed.\n\n        Parameters\n        ----------\n        subset : column label or sequence of labels\n            Use only these columns for identifying duplicates.\n\n        Examples\n        --------\n        >>> import turicreate as tc\n        >>> sf = tc.SFrame({\'A\': [\'a\', \'b\', \'a\', \'C\'], \'B\': [\'b\', \'a\', \'b\', \'D\'], \'C\': [1, 2, 1, 8]})\n        >>> sf.drop_duplicates(subset=["A","B"])\n        Columns:\n\t        A\tstr\n\t        B\tstr\n\t        C\tint\n        Rows: 3\n        Data:\n        +---+---+---+\n        | A | B | C |\n        +---+---+---+\n        | b | a | 2 |\n        | C | D | 8 |\n        | a | b | 1 |\n        +---+---+---+\n        [3 rows x 3 columns]\n\n        '
        result = all((elem in self.column_names() for elem in subset))
        if result:
            return self.groupby(subset, {col: aggregate.SELECT_ONE(col) for col in self.column_names() if col not in subset})
        else:
            raise TypeError('Not all subset columns in SFrame')

    def __str_impl__(self, num_rows=10, footer=True):
        if False:
            return 10
        '\n        Returns a string containing the first num_rows elements of the frame, along\n        with a description of the frame.\n        '
        MAX_ROWS_TO_DISPLAY = num_rows
        printed_sf = self._imagecols_to_stringcols(MAX_ROWS_TO_DISPLAY)
        row_of_tables = printed_sf.__get_pretty_tables__(wrap_text=False, max_rows_to_display=MAX_ROWS_TO_DISPLAY)
        is_empty = len(printed_sf) == 0
        if not footer:
            return (is_empty, '\n'.join([str(tb) for tb in row_of_tables]))
        if self.__has_size__():
            footer = '[%d rows x %d columns]\n' % self.shape
            if self.num_rows() > MAX_ROWS_TO_DISPLAY:
                footer += '\n'.join(FOOTER_STRS)
        else:
            footer = '[? rows x %d columns]\n' % self.num_columns()
            footer += '\n'.join(LAZY_FOOTER_STRS)
        return (is_empty, '\n'.join([str(tb) for tb in row_of_tables]) + '\n' + footer)

    def __str__(self, num_rows=10, footer=True):
        if False:
            i = 10
            return i + 15
        '\n        Returns a string containing the first 10 elements of the frame, along\n        with a description of the frame.\n        '
        return self.__str_impl__(num_rows, footer)[1]

    def _repr_html_(self):
        if False:
            while True:
                i = 10
        MAX_ROWS_TO_DISPLAY = 10
        printed_sf = self._imagecols_to_stringcols(MAX_ROWS_TO_DISPLAY)
        row_of_tables = printed_sf.__get_pretty_tables__(wrap_text=True, max_row_width=120, max_columns=40, max_column_width=25, max_rows_to_display=MAX_ROWS_TO_DISPLAY)
        if self.__has_size__():
            footer = '[%d rows x %d columns]<br/>' % self.shape
            if self.num_rows() > MAX_ROWS_TO_DISPLAY:
                footer += '<br/>'.join(FOOTER_STRS)
        else:
            footer = '[? rows x %d columns]<br/>' % self.num_columns()
            footer += '<br/>'.join(LAZY_FOOTER_STRS)
        begin = '<div style="max-height:1000px;max-width:1500px;overflow:auto;">'
        end = '\n</div>'
        return begin + '\n'.join([tb.get_html_string(format=True) for tb in row_of_tables]) + '\n' + footer + end

    def __nonzero__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns true if the frame is not empty.\n        '
        return self.num_rows() != 0

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of rows of the sframe.\n        '
        return self.num_rows()

    def __copy__(self):
        if False:
            return 10
        '\n        Returns a shallow copy of the sframe.\n        '
        return self.select_columns(self.column_names())

    def __deepcopy__(self, memo):
        if False:
            return 10
        '\n        Returns a deep copy of the sframe. As the data in an SFrame is\n        immutable, this is identical to __copy__.\n        '
        return self.__copy__()

    def copy(self):
        if False:
            return 10
        '\n        Returns a shallow copy of the sframe.\n        '
        return self.__copy__()

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _row_selector(self, other):
        if False:
            while True:
                i = 10
        '\n        Where other is an SArray of identical length as the current Frame,\n        this returns a selection of a subset of rows in the current SFrame\n        where the corresponding row in the selector is non-zero.\n        '
        if type(other) is SArray:
            if self.__has_size__() and other.__has_size__() and (len(other) != len(self)):
                raise IndexError('Cannot perform logical indexing on arrays of different length.')
            with cython_context():
                return SFrame(_proxy=self.__proxy__.logical_filter(other.__proxy__))

    @property
    def dtype(self):
        if False:
            while True:
                i = 10
        '\n        The type of each column.\n\n        Returns\n        -------\n        out : list[type]\n            Column types of the SFrame.\n\n        See Also\n        --------\n        column_types\n        '
        return self.column_types()

    def num_rows(self):
        if False:
            i = 10
            return i + 15
        '\n        The number of rows in this SFrame.\n\n        Returns\n        -------\n        out : int\n            Number of rows in the SFrame.\n\n        See Also\n        --------\n        num_columns\n        '
        return self.__proxy__.num_rows()

    def num_columns(self):
        if False:
            while True:
                i = 10
        '\n        The number of columns in this SFrame.\n\n        Returns\n        -------\n        out : int\n            Number of columns in the SFrame.\n\n        See Also\n        --------\n        num_rows\n        '
        return self.__proxy__.num_columns()

    def column_names(self):
        if False:
            print('Hello World!')
        '\n        The name of each column in the SFrame.\n\n        Returns\n        -------\n        out : list[string]\n            Column names of the SFrame.\n\n        See Also\n        --------\n        rename\n        '
        return self.__proxy__.column_names()

    def column_types(self):
        if False:
            while True:
                i = 10
        '\n        The type of each column in the SFrame.\n\n        Returns\n        -------\n        out : list[type]\n            Column types of the SFrame.\n\n        See Also\n        --------\n        dtype\n        '
        return self.__proxy__.dtype()

    def head(self, n=10):
        if False:
            i = 10
            return i + 15
        '\n        The first n rows of the SFrame.\n\n        Parameters\n        ----------\n        n : int, optional\n            The number of rows to fetch.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame which contains the first n rows of the current SFrame\n\n        See Also\n        --------\n        tail, print_rows\n        '
        return SFrame(_proxy=self.__proxy__.head(n))

    def to_dataframe(self):
        if False:
            return 10
        '\n        Convert this SFrame to pandas.DataFrame.\n\n        This operation will construct a pandas.DataFrame in memory. Care must\n        be taken when size of the returned object is big.\n\n        Returns\n        -------\n        out : pandas.DataFrame\n            The dataframe which contains all rows of SFrame\n        '
        from ..toolkits.image_classifier._evaluation import _image_resize
        assert HAS_PANDAS, 'pandas is not installed.'
        df = pandas.DataFrame()
        for i in range(self.num_columns()):
            column_name = self.column_names()[i]
            if self.column_types()[i] == _Image:
                df[column_name] = [_image_resize(x[column_name])._to_pil_image() for x in self.select_columns([column_name])]
            else:
                df[column_name] = list(self[column_name])
            if len(df[column_name]) == 0:
                column_type = self.column_types()[i]
                if column_type in (array.array, type(None)):
                    column_type = 'object'
                df[column_name] = df[column_name].astype(column_type)
        return df

    def to_numpy(self):
        if False:
            return 10
        '\n        Converts this SFrame to a numpy array\n\n        This operation will construct a numpy array in memory. Care must\n        be taken when size of the returned object is big.\n\n        Returns\n        -------\n        out : numpy.ndarray\n            A Numpy Array containing all the values of the SFrame\n\n        '
        assert HAS_NUMPY, 'numpy is not installed.'
        import numpy
        return numpy.transpose(numpy.asarray([self[x] for x in self.column_names()]))

    def tail(self, n=10):
        if False:
            i = 10
            return i + 15
        '\n        The last n rows of the SFrame.\n\n        Parameters\n        ----------\n        n : int, optional\n            The number of rows to fetch.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame which contains the last n rows of the current SFrame\n\n        See Also\n        --------\n        head, print_rows\n        '
        return SFrame(_proxy=self.__proxy__.tail(n))

    def apply(self, fn, dtype=None):
        if False:
            return 10
        "\n        Transform each row to an :class:`~turicreate.SArray` according to a\n        specified function. Returns a new SArray of ``dtype`` where each element\n        in this SArray is transformed by `fn(x)` where `x` is a single row in\n        the sframe represented as a dictionary.  The ``fn`` should return\n        exactly one value which can be cast into type ``dtype``. If ``dtype`` is\n        not specified, the first 100 rows of the SFrame are used to make a guess\n        of the target data type.\n\n        Parameters\n        ----------\n        fn : function\n            The function to transform each row of the SFrame. The return\n            type should be convertible to `dtype` if `dtype` is not None.\n            This can also be a toolkit extension function which is compiled\n            as a native shared library using SDK.\n\n        dtype : dtype, optional\n            The dtype of the new SArray. If None, the first 100\n            elements of the array are used to guess the target\n            data type.\n\n        Returns\n        -------\n        out : SArray\n            The SArray transformed by fn.  Each element of the SArray is of\n            type ``dtype``\n\n        Examples\n        --------\n        Concatenate strings from several columns:\n\n        >>> sf = turicreate.SFrame({'user_id': [1, 2, 3], 'movie_id': [3, 3, 6],\n                                  'rating': [4, 5, 1]})\n        >>> sf.apply(lambda x: str(x['user_id']) + str(x['movie_id']) + str(x['rating']))\n        dtype: str\n        Rows: 3\n        ['134', '235', '361']\n        "
        assert callable(fn), 'Input must be callable'
        test_sf = self[:10]
        dryrun = [fn(row) for row in test_sf]
        if dtype is None:
            dtype = SArray(dryrun).dtype
        seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        nativefn = None
        try:
            from .. import extensions as extensions
            nativefn = extensions._build_native_function_call(fn)
        except:
            pass
        if nativefn is not None:
            with cython_context():
                return SArray(_proxy=self.__proxy__.transform_native(nativefn, dtype, seed))
        with cython_context():
            return SArray(_proxy=self.__proxy__.transform(fn, dtype, seed))

    def flat_map(self, column_names, fn, column_types='auto', seed=None):
        if False:
            print('Hello World!')
        "\n        Map each row of the SFrame to multiple rows in a new SFrame via a\n        function.\n\n        The output of `fn` must have type List[List[...]].  Each inner list\n        will be a single row in the new output, and the collection of these\n        rows within the outer list make up the data for the output SFrame.\n        All rows must have the same length and the same order of types to\n        make sure the result columns are homogeneously typed.  For example, if\n        the first element emitted into in the outer list by `fn` is\n        [43, 2.3, 'string'], then all other elements emitted into the outer\n        list must be a list with three elements, where the first is an int,\n        second is a float, and third is a string.  If column_types is not\n        specified, the first 10 rows of the SFrame are used to determine the\n        column types of the returned sframe.\n\n        Parameters\n        ----------\n        column_names : list[str]\n            The column names for the returned SFrame.\n\n        fn : function\n            The function that maps each of the sframe row into multiple rows,\n            returning List[List[...]].  All outputted rows must have the same\n            length and order of types.\n\n        column_types : list[type], optional\n            The column types of the output SFrame. Default value will be\n            automatically inferred by running `fn` on the first 10 rows of the\n            input. If the types cannot be inferred from the first 10 rows, an\n            error is raised.\n\n        seed : int, optional\n            Used as the seed if a random number generator is included in `fn`.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame containing the results of the flat_map of the\n            original SFrame.\n\n        Examples\n        ---------\n        Repeat each row according to the value in the 'number' column.\n\n        >>> sf = turicreate.SFrame({'letter': ['a', 'b', 'c'],\n        ...                       'number': [1, 2, 3]})\n        >>> sf.flat_map(['number', 'letter'],\n        ...             lambda x: [list(x.values()) for i in range(x['number'])])\n        +--------+--------+\n        | number | letter |\n        +--------+--------+\n        |   1    |   a    |\n        |   2    |   b    |\n        |   2    |   b    |\n        |   3    |   c    |\n        |   3    |   c    |\n        |   3    |   c    |\n        +--------+--------+\n        [6 rows x 2 columns]\n        "
        assert callable(fn), 'Input must be callable'
        if seed is None:
            seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        if column_types == 'auto':
            types = set()
            sample = self[0:10]
            results = [fn(row) for row in sample]
            for rows in results:
                if type(rows) is not list:
                    raise TypeError('Output type of the lambda function must be a list of lists')
                for row in rows:
                    if type(row) is not list:
                        raise TypeError('Output type of the lambda function must be a list of lists')
                    types.add(tuple([type(v) for v in row]))
            if len(types) == 0:
                raise TypeError('Could not infer output column types from the first ten rows ' + "of the SFrame. Please use the 'column_types' parameter to " + 'set the types.')
            if len(types) > 1:
                raise TypeError('Mapped rows must have the same length and types')
            column_types = list(types.pop())
        assert type(column_types) is list, "'column_types' must be a list."
        assert len(column_types) == len(column_names), 'Number of output columns must match the size of column names'
        with cython_context():
            return SFrame(_proxy=self.__proxy__.flat_map(fn, column_names, column_types, seed))

    def sample(self, fraction, seed=None, exact=False):
        if False:
            print('Hello World!')
        "\n        Sample a fraction of the current SFrame's rows.\n\n        Parameters\n        ----------\n        fraction : float\n            Fraction of the rows to fetch. Must be between 0 and 1.\n            if exact is False (default), the number of rows returned is\n            approximately the fraction times the number of rows.\n\n        seed : int, optional\n            Seed for the random number generator used to sample.\n\n        exact: bool, optional\n            Defaults to False. If exact=True, an exact fraction is returned,\n            but at a performance penalty.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame containing sampled rows of the current SFrame.\n\n        Examples\n        --------\n        Suppose we have an SFrame with 6,145 rows.\n\n        >>> import random\n        >>> sf = SFrame({'id': range(0, 6145)})\n\n        Retrieve about 30% of the SFrame rows with repeatable results by\n        setting the random seed.\n\n        >>> len(sf.sample(.3, seed=5))\n        1783\n        "
        if seed is None:
            seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        if fraction > 1 or fraction < 0:
            raise ValueError('Invalid sampling rate: ' + str(fraction))
        if self.num_rows() == 0 or self.num_columns() == 0:
            return self
        else:
            with cython_context():
                return SFrame(_proxy=self.__proxy__.sample(fraction, seed, exact))

    def shuffle(self):
        if False:
            print('Hello World!')
        '\n        Randomly shuffles the rows of the SFrame.\n\n        Returns\n        -------\n        out : [SFrame]\n            An SFrame with all the same rows but with the rows in a random order.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({"nums": [1, 2, 3, 4],\n                                    "letters": ["a", "b", "c", "d"]})\n        >>> shuffled_sf = sf.shuffle()\n        >>> print(shuffled_sf)\n        +---------+------+\n        | letters | nums |\n        +---------+------+\n        |    d    |  4   |\n        |    c    |  3   |\n        |    a    |  1   |\n        |    b    |  2   |\n        +---------+------+\n        [4 rows x 2 columns]\n        '
        return SFrame(_proxy=self.__proxy__.shuffle())

    def random_split(self, fraction, seed=None, exact=False):
        if False:
            print('Hello World!')
        "\n        Randomly split the rows of an SFrame into two SFrames. The first SFrame\n        contains *M* rows, sampled uniformly (without replacement) from the\n        original SFrame. *M* is approximately the fraction times the original\n        number of rows. The second SFrame contains the remaining rows of the\n        original SFrame.\n\n        An exact fraction partition can be optionally obtained by setting\n        exact=True.\n\n        Parameters\n        ----------\n        fraction : float\n            Fraction of the rows to fetch. Must be between 0 and 1.\n            if exact is False (default), the number of rows returned is\n            approximately the fraction times the number of rows.\n\n        seed : int, optional\n            Seed for the random number generator used to split.\n\n        exact: bool, optional\n            Defaults to False. If exact=True, an exact fraction is returned,\n            but at a performance penalty.\n\n        Returns\n        -------\n        out : tuple [SFrame]\n            Two new SFrames.\n\n        Examples\n        --------\n        Suppose we have an SFrame with 1,024 rows and we want to randomly split\n        it into training and testing datasets with about a 90%/10% split.\n\n        >>> sf = turicreate.SFrame({'id': range(1024)})\n        >>> sf_train, sf_test = sf.random_split(.9, seed=5)\n        >>> print(len(sf_train), len(sf_test))\n        922 102\n        "
        if fraction > 1 or fraction < 0:
            raise ValueError('Invalid sampling rate: ' + str(fraction))
        if self.num_rows() == 0 or self.num_columns() == 0:
            return (SFrame(), SFrame())
        if seed is None:
            seed = abs(hash('%0.20f' % time.time())) % 2 ** 31
        try:
            seed = int(seed)
        except ValueError:
            raise ValueError("The 'seed' parameter must be of type int.")
        with cython_context():
            proxy_pair = self.__proxy__.random_split(fraction, seed, exact)
            return (SFrame(data=[], _proxy=proxy_pair[0]), SFrame(data=[], _proxy=proxy_pair[1]))

    def topk(self, column_name, k=10, reverse=False):
        if False:
            while True:
                i = 10
        "\n        Get top k rows according to the given column. Result is according to and\n        sorted by `column_name` in the given order (default is descending).\n        When `k` is small, `topk` is more efficient than `sort`.\n\n        Parameters\n        ----------\n        column_name : string\n            The column to sort on\n\n        k : int, optional\n            The number of rows to return\n\n        reverse : bool, optional\n            If True, return the top k rows in ascending order, otherwise, in\n            descending order.\n\n        Returns\n        -------\n        out : SFrame\n            an SFrame containing the top k rows sorted by column_name.\n\n        See Also\n        --------\n        sort\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': range(1000)})\n        >>> sf['value'] = -sf['id']\n        >>> sf.topk('id', k=3)\n        +--------+--------+\n        |   id   |  value |\n        +--------+--------+\n        |   999  |  -999  |\n        |   998  |  -998  |\n        |   997  |  -997  |\n        +--------+--------+\n        [3 rows x 2 columns]\n\n        >>> sf.topk('value', k=3)\n        +--------+--------+\n        |   id   |  value |\n        +--------+--------+\n        |   1    |  -1    |\n        |   2    |  -2    |\n        |   3    |  -3    |\n        +--------+--------+\n        [3 rows x 2 columns]\n        "
        if type(column_name) is not str:
            raise TypeError('column_name must be a string')
        sf = self[self[column_name].is_topk(k, reverse)]
        return sf.sort(column_name, ascending=reverse)

    def save(self, filename, format=None):
        if False:
            i = 10
            return i + 15
        "\n        Save the SFrame to a file system for later use.\n\n        Parameters\n        ----------\n        filename : string\n            The location to save the SFrame. Either a local directory or a\n            remote URL. If the format is 'binary', a directory will be created\n            at the location which will contain the sframe.\n\n        format : {'binary', 'csv', 'json'}, optional\n            Format in which to save the SFrame. Binary saved SFrames can be\n            loaded much faster and without any format conversion losses. If not\n            given, will try to infer the format from filename given. If file\n            name ends with 'csv' or '.csv.gz', then save as 'csv' format,\n            otherwise save as 'binary' format.\n            See export_csv for more csv saving options.\n\n        See Also\n        --------\n        load_sframe, SFrame\n\n        Examples\n        --------\n        >>> # Save the sframe into binary format\n        >>> sf.save('data/training_data_sframe')\n\n        >>> # Save the sframe into csv format\n        >>> sf.save('data/training_data.csv', format='csv')\n        "
        if format is None:
            if filename.endswith(('.csv', '.csv.gz')):
                format = 'csv'
            elif filename.endswith('.json'):
                format = 'json'
            else:
                format = 'binary'
        elif format == 'csv':
            if not filename.endswith(('.csv', '.csv.gz')):
                filename = filename + '.csv'
        elif format != 'binary' and format != 'json':
            raise ValueError("Invalid format: {}. Supported formats are 'csv' and 'binary' and 'json'".format(format))
        url = _make_internal_url(filename)
        with cython_context():
            if format == 'binary':
                self.__proxy__.save(url)
            elif format == 'csv':
                assert filename.endswith(('.csv', '.csv.gz'))
                self.__proxy__.save_as_csv(url, {})
            elif format == 'json':
                self.export_json(url)
            else:
                raise ValueError('Unsupported format: {}'.format(format))

    def export_csv(self, filename, delimiter=',', line_terminator='\n', header=True, quote_level=csv.QUOTE_NONNUMERIC, double_quote=True, escape_char='\\', quote_char='"', na_rep='', file_header='', file_footer='', line_prefix='', _no_prefix_on_first_value=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Writes an SFrame to a CSV file.\n\n        Parameters\n        ----------\n        filename : string\n            The location to save the CSV.\n\n        delimiter : string, optional\n            This describes the delimiter used for writing csv files.\n\n        line_terminator: string, optional\n            The newline character\n\n        header : bool, optional\n            If true, the column names are emitted as a header.\n\n        quote_level: csv.QUOTE_ALL | csv.QUOTE_NONE | csv.QUOTE_NONNUMERIC, optional\n            The quoting level. If csv.QUOTE_ALL, every field is quoted.\n            if csv.quote_NONE, no field is quoted. If csv.QUOTE_NONNUMERIC, only\n            non-numeric fileds are quoted. csv.QUOTE_MINIMAL is interpreted as\n            csv.QUOTE_NONNUMERIC.\n\n        double_quote : bool, optional\n            If True, quotes are escaped as two consecutive quotes\n\n        escape_char : string, optional\n            Character which begins a C escape sequence\n\n        quote_char: string, optional\n            Character used to quote fields\n\n        na_rep: string, optional\n            The value used to denote a missing value.\n\n        file_header: string, optional\n            A string printed to the start of the file\n\n        file_footer: string, optional\n            A string printed to the end of the file\n\n        line_prefix: string, optional\n            A string printed at the start of each value line\n        '
        if 'sep' in kwargs:
            delimiter = kwargs['sep']
            del kwargs['sep']
        if 'quotechar' in kwargs:
            quote_char = kwargs['quotechar']
            del kwargs['quotechar']
        if 'doublequote' in kwargs:
            double_quote = kwargs['doublequote']
            del kwargs['doublequote']
        if 'lineterminator' in kwargs:
            line_terminator = kwargs['lineterminator']
            del kwargs['lineterminator']
        if len(kwargs) > 0:
            raise TypeError('Unexpected keyword arguments ' + str(list(kwargs.keys())))
        write_csv_options = {}
        write_csv_options['delimiter'] = delimiter
        write_csv_options['escape_char'] = escape_char
        write_csv_options['double_quote'] = double_quote
        write_csv_options['quote_char'] = quote_char
        if quote_level == csv.QUOTE_MINIMAL:
            write_csv_options['quote_level'] = 0
        elif quote_level == csv.QUOTE_ALL:
            write_csv_options['quote_level'] = 1
        elif quote_level == csv.QUOTE_NONNUMERIC:
            write_csv_options['quote_level'] = 2
        elif quote_level == csv.QUOTE_NONE:
            write_csv_options['quote_level'] = 3
        write_csv_options['header'] = header
        write_csv_options['line_terminator'] = line_terminator
        write_csv_options['na_value'] = na_rep
        write_csv_options['file_header'] = file_header
        write_csv_options['file_footer'] = file_footer
        write_csv_options['line_prefix'] = line_prefix
        write_csv_options['_no_prefix_on_first_value'] = _no_prefix_on_first_value
        url = _make_internal_url(filename)
        self.__proxy__.save_as_csv(url, write_csv_options)

    def export_json(self, filename, orient='records'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Writes an SFrame to a JSON file.\n\n        Parameters\n        ----------\n        filename : string\n            The location to save the JSON file.\n\n        orient : string, optional. Either "records" or "lines"\n            If orient="records" the file is saved as a single JSON array.\n            If orient="lines", the file is saves as a JSON value per line.\n\n        Examples\n        --------\n        The orient parameter describes the expected input format of the JSON\n        file.\n\n        If orient="records", the output will be a single JSON Array where\n        each array element is a dictionary describing the row.\n\n        >>> g\n        Columns:\n                a\tint\n                b\tint\n        Rows: 3\n        Data:\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | 1 |\n        | 2 | 2 |\n        | 3 | 3 |\n        +---+---+\n        >>> g.export(\'output.json\', orient=\'records\')\n        >>> !cat output.json\n        [\n        {\'a\':1,\'b\':1},\n        {\'a\':2,\'b\':2},\n        {\'a\':3,\'b\':3},\n        ]\n\n        If orient="rows", each row will be emitted as a JSON dictionary to\n        each file line.\n\n        >>> g\n        Columns:\n                a\tint\n                b\tint\n        Rows: 3\n        Data:\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | 1 |\n        | 2 | 2 |\n        | 3 | 3 |\n        +---+---+\n        >>> g.export(\'output.json\', orient=\'rows\')\n        >>> !cat output.json\n        {\'a\':1,\'b\':1}\n        {\'a\':2,\'b\':2}\n        {\'a\':3,\'b\':3}\n        '
        if orient == 'records':
            self.pack_columns(dtype=dict).export_csv(filename, file_header='[', file_footer=']', header=False, double_quote=False, quote_level=csv.QUOTE_NONE, line_prefix=',', _no_prefix_on_first_value=True)
        elif orient == 'lines':
            self.pack_columns(dtype=dict).export_csv(filename, header=False, double_quote=False, quote_level=csv.QUOTE_NONE)
        else:
            raise ValueError('Invalid value for orient parameter (' + str(orient) + ')')

    def _save_reference(self, filename):
        if False:
            i = 10
            return i + 15
        "\n        Performs an incomplete save of an existing SFrame into a directory.\n        This saved SFrame may reference SFrames in other locations in the same\n        filesystem for certain resources.\n\n        Parameters\n        ----------\n        filename : string\n            The location to save the SFrame. Either a local directory or a\n            remote URL.\n\n        See Also\n        --------\n        load_sframe, SFrame\n\n        Examples\n        --------\n        >>> # Save the sframe into binary format\n        >>> sf.save_reference('data/training_data_sframe')\n        "
        url = _make_internal_url(filename)
        with cython_context():
            self.__proxy__.save_reference(url)

    def select_column(self, column_name):
        if False:
            print('Hello World!')
        "\n        Get a reference to the :class:`~turicreate.SArray` that corresponds with\n        the given column_name. Throws an exception if the column_name is\n        something other than a string or if the column name is not found.\n\n        Parameters\n        ----------\n        column_name: str\n            The column name.\n\n        Returns\n        -------\n        out : SArray\n            The SArray that is referred by ``column_name``.\n\n        See Also\n        --------\n        select_columns\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'user_id': [1,2,3],\n        ...                       'user_name': ['alice', 'bob', 'charlie']})\n        >>> # This line is equivalent to `sa = sf['user_name']`\n        >>> sa = sf.select_column('user_name')\n        >>> sa\n        dtype: str\n        Rows: 3\n        ['alice', 'bob', 'charlie']\n        "
        if not isinstance(column_name, str):
            raise TypeError('Invalid column_nametype: must be str')
        with cython_context():
            return SArray(data=[], _proxy=self.__proxy__.select_column(column_name))

    def select_columns(self, column_names):
        if False:
            return 10
        "\n        Selects all columns where the name of the column or the type of column\n        is included in the column_names. An exception is raised if duplicate columns\n        are selected i.e. sf.select_columns(['a','a']), or non-existent columns\n        are selected.\n\n        Throws an exception for all other input types.\n\n        Parameters\n        ----------\n        column_names: list[str or type]\n            The list of column names or a list of types.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that is made up of the columns referred to in\n            ``column_names`` from the current SFrame.\n\n        See Also\n        --------\n        select_column\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'user_id': [1,2,3],\n        ...                       'user_name': ['alice', 'bob', 'charlie'],\n        ...                       'zipcode': [98101, 98102, 98103]\n        ...                      })\n        >>> # This line is equivalent to `sf2 = sf[['user_id', 'zipcode']]`\n        >>> sf2 = sf.select_columns(['user_id', 'zipcode'])\n        >>> sf2\n        +---------+---------+\n        | user_id | zipcode |\n        +---------+---------+\n        |    1    |  98101  |\n        |    2    |  98102  |\n        |    3    |  98103  |\n        +---------+---------+\n        [3 rows x 2 columns]\n        "
        if not _is_non_string_iterable(column_names):
            raise TypeError('column_names must be an iterable')
        if not all([isinstance(x, six.string_types) or isinstance(x, type) or isinstance(x, bytes) for x in column_names]):
            raise TypeError('Invalid key type: must be str, unicode, bytes or type')
        requested_str_columns = [s for s in column_names if isinstance(s, six.string_types)]
        from collections import Counter
        column_names_counter = Counter(column_names)
        if len(column_names) != len(column_names_counter):
            for key in column_names_counter:
                if column_names_counter[key] > 1:
                    raise ValueError("There are duplicate keys in key list: '" + key + "'")
        colnames_and_types = list(zip(self.column_names(), self.column_types()))
        selected_columns = requested_str_columns
        typelist = [s for s in column_names if isinstance(s, type)]
        for i in colnames_and_types:
            if i[1] in typelist and i[0] not in selected_columns:
                selected_columns += [i[0]]
        selected_columns = selected_columns
        with cython_context():
            return SFrame(data=[], _proxy=self.__proxy__.select_columns(selected_columns))

    def add_column(self, data, column_name='', inplace=False):
        if False:
            while True:
                i = 10
        "\n        Returns an SFrame with a new column. The number of elements in the data\n        given must match the length of every other column of the SFrame.\n        If no name is given, a default name is chosen.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        data : SArray\n            The 'column' of data to add.\n\n        column_name : string, optional\n            The name of the column. If no name is given, a default name is\n            chosen.\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n\n        Returns\n        -------\n        out : SFrame\n            The current SFrame.\n\n        See Also\n        --------\n        add_columns\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})\n        >>> sa = turicreate.SArray(['cat', 'dog', 'fossa'])\n        >>> # This line is equivalent to `sf['species'] = sa`\n        >>> res = sf.add_column(sa, 'species')\n        >>> res\n        +----+-----+---------+\n        | id | val | species |\n        +----+-----+---------+\n        | 1  |  A  |   cat   |\n        | 2  |  B  |   dog   |\n        | 3  |  C  |  fossa  |\n        +----+-----+---------+\n        [3 rows x 3 columns]\n        "
        if not isinstance(data, SArray):
            if isinstance(data, _Iterable):
                data = SArray(data)
            elif self.num_columns() == 0:
                data = SArray([data])
            else:
                data = SArray.from_const(data, self.num_rows())
        if not isinstance(column_name, str):
            raise TypeError('Invalid column name: must be str')
        if inplace:
            ret = self
        else:
            ret = self.copy()
        with cython_context():
            ret.__proxy__.add_column(data.__proxy__, column_name)
        ret._cache = None
        return ret

    def add_columns(self, data, column_names=None, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns an SFrame with multiple columns added. The number of\n        elements in all columns must match the length of every other column of\n        the SFrame.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        data : list[SArray] or SFrame\n            The columns to add.\n\n        column_names: list of string, optional\n            A list of column names. All names must be specified. ``column_names`` is\n            ignored if data is an SFrame.\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n\n        Returns\n        -------\n        out : SFrame\n            The current SFrame.\n\n        See Also\n        --------\n        add_column\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})\n        >>> sf2 = turicreate.SFrame({'species': ['cat', 'dog', 'fossa'],\n        ...                        'age': [3, 5, 9]})\n        >>> res = sf.add_columns(sf2)\n        >>> res\n        +----+-----+-----+---------+\n        | id | val | age | species |\n        +----+-----+-----+---------+\n        | 1  |  A  |  3  |   cat   |\n        | 2  |  B  |  5  |   dog   |\n        | 3  |  C  |  9  |  fossa  |\n        +----+-----+-----+---------+\n        [3 rows x 4 columns]\n        "
        datalist = data
        if isinstance(data, SFrame):
            other = data
            datalist = [other.select_column(name) for name in other.column_names()]
            column_names = other.column_names()
            my_columns = set(self.column_names())
            for name in column_names:
                if name in my_columns:
                    raise ValueError("Column '" + name + "' already exists in current SFrame")
        else:
            if not _is_non_string_iterable(datalist):
                raise TypeError('datalist must be an iterable')
            if not _is_non_string_iterable(column_names):
                raise TypeError('column_names must be an iterable')
            if not all([isinstance(x, SArray) for x in datalist]):
                raise TypeError('Must give column as SArray')
            if not all([isinstance(x, str) for x in column_names]):
                raise TypeError('Invalid column name in list : must all be str')
        if inplace:
            ret = self
        else:
            ret = self.copy()
        with cython_context():
            ret.__proxy__.add_columns([x.__proxy__ for x in datalist], column_names)
        ret._cache = None
        return ret

    def remove_column(self, column_name, inplace=False):
        if False:
            return 10
        "\n        Returns an SFrame with a column removed.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        column_name : string\n            The name of the column to remove.\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n\n        Returns\n        -------\n        out : SFrame\n            The SFrame with given column removed.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})\n        >>> # This is equivalent to `del sf['val']`\n        >>> res = sf.remove_column('val')\n        >>> res\n        +----+\n        | id |\n        +----+\n        | 1  |\n        | 2  |\n        | 3  |\n        +----+\n        [3 rows x 1 columns]\n        "
        column_name = str(column_name)
        if column_name not in self.column_names():
            raise KeyError('Cannot find column %s' % column_name)
        colid = self.column_names().index(column_name)
        if inplace:
            ret = self
        else:
            ret = self.copy()
        with cython_context():
            ret.__proxy__.remove_column(colid)
        ret._cache = None
        return ret

    def remove_columns(self, column_names, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns an SFrame with one or more columns removed.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        column_names : list or iterable\n            A list or iterable of column names.\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n\n        Returns\n        -------\n        out : SFrame\n            The SFrame with given columns removed.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val1': ['A', 'B', 'C'], 'val2' : [10, 11, 12]})\n        >>> res = sf.remove_columns(['val1', 'val2'])\n        >>> res\n        +----+\n        | id |\n        +----+\n        | 1  |\n        | 2  |\n        | 3  |\n        +----+\n        [3 rows x 1 columns]\n        "
        column_names = list(column_names)
        existing_columns = dict(((k, i) for (i, k) in enumerate(self.column_names())))
        for name in column_names:
            if name not in existing_columns:
                raise KeyError('Cannot find column %s' % name)
        deletion_indices = sorted((existing_columns[name] for name in column_names))
        if inplace:
            ret = self
        else:
            ret = self.copy()
        for colid in reversed(deletion_indices):
            with cython_context():
                ret.__proxy__.remove_column(colid)
        ret._cache = None
        return ret

    def swap_columns(self, column_name_1, column_name_2, inplace=False):
        if False:
            while True:
                i = 10
        "\n        Returns an SFrame with two column positions swapped.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        column_name_1 : string\n            Name of column to swap\n\n        column_name_2 : string\n            Name of other column to swap\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n\n        Returns\n        -------\n        out : SFrame\n            The SFrame with swapped columns.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})\n        >>> res = sf.swap_columns('id', 'val')\n        >>> res\n        +-----+-----+\n        | val | id  |\n        +-----+-----+\n        |  A  |  1  |\n        |  B  |  2  |\n        |  C  |  3  |\n        +----+-----+\n        [3 rows x 2 columns]\n        "
        colnames = self.column_names()
        colid_1 = colnames.index(column_name_1)
        colid_2 = colnames.index(column_name_2)
        if inplace:
            ret = self
        else:
            ret = self.copy()
        with cython_context():
            ret.__proxy__.swap_columns(colid_1, colid_2)
        ret._cache = None
        return ret

    def rename(self, names, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns an SFrame with columns renamed. ``names`` is expected to be a\n        dict specifying the old and new names. This changes the names of the\n        columns given as the keys and replaces them with the names given as the\n        values.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        names : dict [string, string]\n            Dictionary of [old_name, new_name]\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n\n        Returns\n        -------\n        out : SFrame\n            The current SFrame.\n\n        See Also\n        --------\n        column_names\n\n        Examples\n        --------\n        >>> sf = SFrame({'X1': ['Alice','Bob'],\n        ...              'X2': ['123 Fake Street','456 Fake Street']})\n        >>> res = sf.rename({'X1': 'name', 'X2':'address'})\n        >>> res\n        +-------+-----------------+\n        |  name |     address     |\n        +-------+-----------------+\n        | Alice | 123 Fake Street |\n        |  Bob  | 456 Fake Street |\n        +-------+-----------------+\n        [2 rows x 2 columns]\n        "
        if type(names) is not dict:
            raise TypeError('names must be a dictionary: oldname -> newname')
        all_columns = set(self.column_names())
        for k in names:
            if not k in all_columns:
                raise ValueError('Cannot find column %s in the SFrame' % k)
        if inplace:
            ret = self
        else:
            ret = self.copy()
        with cython_context():
            for k in names:
                colid = ret.column_names().index(k)
                ret.__proxy__.set_column_name(colid, names[k])
        ret._cache = None
        return ret

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        "\n        This method does things based on the type of `key`.\n\n        If `key` is:\n            * str\n                selects column with name 'key'\n            * type\n                selects all columns with types matching the type\n            * list of str or type\n                selects all columns with names or type in the list\n            * SArray\n                Performs a logical filter.  Expects given SArray to be the same\n                length as all columns in current SFrame.  Every row\n                corresponding with an entry in the given SArray that is\n                equivalent to False is filtered from the result.\n            * int\n                Returns a single row of the SFrame (the `key`th one) as a dictionary.\n            * slice\n                Returns an SFrame including only the sliced rows.\n        "
        if type(key) is SArray:
            return self._row_selector(key)
        elif isinstance(key, six.string_types):
            if six.PY2 and type(key) == unicode:
                key = key.encode('utf-8')
            return self.select_column(key)
        elif type(key) is type:
            return self.select_columns([key])
        elif _is_non_string_iterable(key):
            return self.select_columns(key)
        elif isinstance(key, numbers.Integral):
            sf_len = len(self)
            if key < 0:
                key = sf_len + key
            if key >= sf_len:
                raise IndexError('SFrame index out of range')
            if not hasattr(self, '_cache') or self._cache is None:
                self._cache = {}
            try:
                (lb, ub, value_list) = self._cache['getitem_cache']
                if lb <= key < ub:
                    return value_list[int(key - lb)]
            except KeyError:
                pass
            if not 'getitem_cache_blocksize' in self._cache:
                block_size = 8 * 1024 // sum((2 if dt in [int, long, float] else 8 for dt in self.column_types()))
                block_size = max(16, block_size)
                self._cache['getitem_cache_blocksize'] = block_size
            else:
                block_size = self._cache['getitem_cache_blocksize']
            block_num = int(key // block_size)
            lb = block_num * block_size
            ub = min(sf_len, lb + block_size)
            val_list = list(SFrame(_proxy=self.__proxy__.copy_range(lb, 1, ub)))
            self._cache['getitem_cache'] = (lb, ub, val_list)
            return val_list[int(key - lb)]
        elif type(key) is slice:
            start = key.start
            stop = key.stop
            step = key.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)
            if step is None:
                step = 1
            if start < 0:
                start = len(self) + start
            if stop < 0:
                stop = len(self) + stop
            return SFrame(_proxy=self.__proxy__.copy_range(start, step, stop))
        else:
            raise TypeError('Invalid index type: must be SArray, list, int, or str')

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        '\n        A wrapper around add_column(s).  Key can be either a list or a str.  If\n        value is an SArray, it is added to the SFrame as a column.  If it is a\n        constant value (int, str, or float), then a column is created where\n        every entry is equal to the constant value.  Existing columns can also\n        be replaced using this wrapper.\n        '
        if type(key) is list:
            self.add_columns(value, key, inplace=True)
        elif type(key) is str:
            sa_value = None
            if type(value) is SArray:
                sa_value = value
            elif _is_non_string_iterable(value):
                sa_value = SArray(value)
            else:
                sa_value = SArray.from_const(value, self.num_rows())
            if not key in self.column_names():
                with cython_context():
                    self.add_column(sa_value, key, inplace=True)
            else:
                single_column = self.num_columns() == 1
                if single_column:
                    tmpname = key
                    saved_column = self.select_column(key)
                    self.remove_column(key, inplace=True)
                else:
                    tmpname = '__' + '-'.join(self.column_names())
                try:
                    self.add_column(sa_value, tmpname, inplace=True)
                except Exception:
                    if single_column:
                        self.add_column(saved_column, key, inplace=True)
                    raise
                if not single_column:
                    self.swap_columns(key, tmpname, inplace=True)
                    self.remove_column(key, inplace=True)
                    self.rename({tmpname: key}, inplace=True)
        else:
            raise TypeError('Cannot set column with key type ' + str(type(key)))

    def __delitem__(self, key):
        if False:
            return 10
        '\n        Wrapper around remove_column.\n        '
        self.remove_column(key, inplace=True)

    def materialize(self):
        if False:
            i = 10
            return i + 15
        '\n        For an SFrame that is lazily evaluated, force the persistence of the\n        SFrame to disk, committing all lazy evaluated operations.\n        '
        with cython_context():
            self.__proxy__.materialize()

    def is_materialized(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns whether or not the SFrame has been materialized.\n        '
        return self.__is_materialized__()

    def __is_materialized__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns whether or not the SFrame has been materialized.\n        '
        return self.__proxy__.is_materialized()

    def __has_size__(self):
        if False:
            while True:
                i = 10
        '\n        Returns whether or not the size of the SFrame is known.\n        '
        return self.__proxy__.has_size()

    def __query_plan_str__(self):
        if False:
            print('Hello World!')
        '\n        Returns the query plan as a dot graph string\n        '
        return self.__proxy__.query_plan_string()

    def __iter__(self):
        if False:
            while True:
                i = 10
        '\n        Provides an iterator to the rows of the SFrame.\n        '

        def generator():
            if False:
                i = 10
                return i + 15
            elems_at_a_time = 262144
            self.__proxy__.begin_iterator()
            ret = self.__proxy__.iterator_get_next(elems_at_a_time)
            column_names = self.column_names()
            while True:
                for j in ret:
                    yield dict(list(zip(column_names, j)))
                if len(ret) == elems_at_a_time:
                    ret = self.__proxy__.iterator_get_next(elems_at_a_time)
                else:
                    break
        return generator()

    def append(self, other):
        if False:
            for i in range(10):
                print('nop')
        "\n        Add the rows of an SFrame to the end of this SFrame.\n\n        Both SFrames must have the same set of columns with the same column\n        names and column types.\n\n        Parameters\n        ----------\n        other : SFrame\n            Another SFrame whose rows are appended to the current SFrame.\n\n        Returns\n        -------\n        out : SFrame\n            The result SFrame from the append operation.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': [4, 6, 8], 'val': ['D', 'F', 'H']})\n        >>> sf2 = turicreate.SFrame({'id': [1, 2, 3], 'val': ['A', 'B', 'C']})\n        >>> sf = sf.append(sf2)\n        >>> sf\n        +----+-----+\n        | id | val |\n        +----+-----+\n        | 4  |  D  |\n        | 6  |  F  |\n        | 8  |  H  |\n        | 1  |  A  |\n        | 2  |  B  |\n        | 3  |  C  |\n        +----+-----+\n        [6 rows x 2 columns]\n        "
        if type(other) is not SFrame:
            raise RuntimeError('SFrame append can only work with SFrame')
        with cython_context():
            return SFrame(_proxy=self.__proxy__.append(other.__proxy__))

    def groupby(self, key_column_names, operations, *args):
        if False:
            return 10
        '\n        Perform a group on the key_column_names followed by aggregations on the\n        columns listed in operations.\n\n        The operations parameter is a dictionary that indicates which\n        aggregation operators to use and which columns to use them on. The\n        available operators are SUM, MAX, MIN, COUNT, AVG, VAR, STDV, CONCAT,\n        SELECT_ONE, ARGMIN, ARGMAX, and QUANTILE. For convenience, aggregators\n        MEAN, STD, and VARIANCE are available as synonyms for AVG, STDV, and\n        VAR. See :mod:`~turicreate.aggregate` for more detail on the aggregators.\n\n        Parameters\n        ----------\n        key_column_names : string | list[string]\n            Column(s) to group by. Key columns can be of any type other than\n            dictionary.\n\n        operations : dict, list\n            Dictionary of columns and aggregation operations. Each key is a\n            output column name and each value is an aggregator. This can also\n            be a list of aggregators, in which case column names will be\n            automatically assigned.\n\n        *args\n            All other remaining arguments will be interpreted in the same\n            way as the operations argument.\n\n        Returns\n        -------\n        out_sf : SFrame\n            A new SFrame, with a column for each groupby column and each\n            aggregation operation.\n\n        See Also\n        --------\n        aggregate\n\n        Notes\n        -----\n        * Numeric aggregators (such as sum, mean, stdev etc.) follow the skip\n        None policy i.e they will omit all missing values from the aggregation.\n        As an example, `sum([None, 5, 10]) = 15` because the `None` value is\n        skipped.\n        * Aggregators have a default value when no values (after skipping all\n        `None` values) are present. Default values are `None` for [\'ARGMAX\',\n        \'ARGMIN\', \'AVG\', \'STD\', \'MEAN\', \'MIN\', \'MAX\'],  `0` for [\'COUNT\'\n        \'COUNT_DISTINCT\', \'DISTINCT\'] `[]` for \'CONCAT\', \'QUANTILE\',\n        \'DISTINCT\', and `{}` for \'FREQ_COUNT\'.\n\n        Examples\n        --------\n        Suppose we have an SFrame with movie ratings by many users.\n\n        >>> import turicreate.aggregate as agg\n        >>> url = \'https://static.turi.com/datasets/rating_data_example.csv\'\n        >>> sf = turicreate.SFrame.read_csv(url)\n        >>> sf\n        +---------+----------+--------+\n        | user_id | movie_id | rating |\n        +---------+----------+--------+\n        |  25904  |   1663   |   3    |\n        |  25907  |   1663   |   3    |\n        |  25923  |   1663   |   3    |\n        |  25924  |   1663   |   3    |\n        |  25928  |   1663   |   2    |\n        |  25933  |   1663   |   4    |\n        |  25934  |   1663   |   4    |\n        |  25935  |   1663   |   4    |\n        |  25936  |   1663   |   5    |\n        |  25937  |   1663   |   2    |\n        |   ...   |   ...    |  ...   |\n        +---------+----------+--------+\n        [10000 rows x 3 columns]\n\n        Compute the number of occurrences of each user.\n\n        >>> user_count = sf.groupby(key_column_names=\'user_id\',\n        ...                         operations={\'count\': agg.COUNT()})\n        >>> user_count\n        +---------+-------+\n        | user_id | count |\n        +---------+-------+\n        |  62361  |   1   |\n        |  30727  |   1   |\n        |  40111  |   1   |\n        |  50513  |   1   |\n        |  35140  |   1   |\n        |  42352  |   1   |\n        |  29667  |   1   |\n        |  46242  |   1   |\n        |  58310  |   1   |\n        |  64614  |   1   |\n        |   ...   |  ...  |\n        +---------+-------+\n        [9852 rows x 2 columns]\n\n        Compute the mean and standard deviation of ratings per user.\n\n        >>> user_rating_stats = sf.groupby(key_column_names=\'user_id\',\n        ...                                operations={\n        ...                                    \'mean_rating\': agg.MEAN(\'rating\'),\n        ...                                    \'std_rating\': agg.STD(\'rating\')\n        ...                                })\n        >>> user_rating_stats\n        +---------+-------------+------------+\n        | user_id | mean_rating | std_rating |\n        +---------+-------------+------------+\n        |  62361  |     5.0     |    0.0     |\n        |  30727  |     4.0     |    0.0     |\n        |  40111  |     2.0     |    0.0     |\n        |  50513  |     4.0     |    0.0     |\n        |  35140  |     4.0     |    0.0     |\n        |  42352  |     5.0     |    0.0     |\n        |  29667  |     4.0     |    0.0     |\n        |  46242  |     5.0     |    0.0     |\n        |  58310  |     2.0     |    0.0     |\n        |  64614  |     2.0     |    0.0     |\n        |   ...   |     ...     |    ...     |\n        +---------+-------------+------------+\n        [9852 rows x 3 columns]\n\n        Compute the movie with the minimum rating per user.\n\n        >>> chosen_movies = sf.groupby(key_column_names=\'user_id\',\n        ...                            operations={\n        ...                                \'worst_movies\': agg.ARGMIN(\'rating\',\'movie_id\')\n        ...                            })\n        >>> chosen_movies\n        +---------+-------------+\n        | user_id | worst_movies |\n        +---------+-------------+\n        |  62361  |     1663    |\n        |  30727  |     1663    |\n        |  40111  |     1663    |\n        |  50513  |     1663    |\n        |  35140  |     1663    |\n        |  42352  |     1663    |\n        |  29667  |     1663    |\n        |  46242  |     1663    |\n        |  58310  |     1663    |\n        |  64614  |     1663    |\n        |   ...   |     ...     |\n        +---------+-------------+\n        [9852 rows x 2 columns]\n\n        Compute the movie with the max rating per user and also the movie with\n        the maximum imdb-ranking per user.\n\n        >>> sf[\'imdb-ranking\'] = sf[\'rating\'] * 10\n        >>> chosen_movies = sf.groupby(key_column_names=\'user_id\',\n        ...         operations={(\'max_rating_movie\',\'max_imdb_ranking_movie\'): agg.ARGMAX((\'rating\',\'imdb-ranking\'),\'movie_id\')})\n        >>> chosen_movies\n        +---------+------------------+------------------------+\n        | user_id | max_rating_movie | max_imdb_ranking_movie |\n        +---------+------------------+------------------------+\n        |  62361  |       1663       |          16630         |\n        |  30727  |       1663       |          16630         |\n        |  40111  |       1663       |          16630         |\n        |  50513  |       1663       |          16630         |\n        |  35140  |       1663       |          16630         |\n        |  42352  |       1663       |          16630         |\n        |  29667  |       1663       |          16630         |\n        |  46242  |       1663       |          16630         |\n        |  58310  |       1663       |          16630         |\n        |  64614  |       1663       |          16630         |\n        |   ...   |       ...        |          ...           |\n        +---------+------------------+------------------------+\n        [9852 rows x 3 columns]\n\n        Compute the movie with the max rating per user.\n\n        >>> chosen_movies = sf.groupby(key_column_names=\'user_id\',\n                    operations={\'best_movies\': agg.ARGMAX(\'rating\',\'movie\')})\n\n        Compute the movie with the max rating per user and also the movie with the maximum imdb-ranking per user.\n\n        >>> chosen_movies = sf.groupby(key_column_names=\'user_id\',\n                   operations={(\'max_rating_movie\',\'max_imdb_ranking_movie\'): agg.ARGMAX((\'rating\',\'imdb-ranking\'),\'movie\')})\n\n        Compute the count, mean, and standard deviation of ratings per (user,\n        time), automatically assigning output column names.\n\n        >>> sf[\'time\'] = sf.apply(lambda x: (x[\'user_id\'] + x[\'movie_id\']) % 11 + 2000)\n        >>> user_rating_stats = sf.groupby([\'user_id\', \'time\'],\n        ...                                [agg.COUNT(),\n        ...                                 agg.AVG(\'rating\'),\n        ...                                 agg.STDV(\'rating\')])\n        >>> user_rating_stats\n        +------+---------+-------+---------------+----------------+\n        | time | user_id | Count | Avg of rating | Stdv of rating |\n        +------+---------+-------+---------------+----------------+\n        | 2006 |  61285  |   1   |      4.0      |      0.0       |\n        | 2000 |  36078  |   1   |      4.0      |      0.0       |\n        | 2003 |  47158  |   1   |      3.0      |      0.0       |\n        | 2007 |  34446  |   1   |      3.0      |      0.0       |\n        | 2010 |  47990  |   1   |      3.0      |      0.0       |\n        | 2003 |  42120  |   1   |      5.0      |      0.0       |\n        | 2007 |  44940  |   1   |      4.0      |      0.0       |\n        | 2008 |  58240  |   1   |      4.0      |      0.0       |\n        | 2002 |   102   |   1   |      1.0      |      0.0       |\n        | 2009 |  52708  |   1   |      3.0      |      0.0       |\n        | ...  |   ...   |  ...  |      ...      |      ...       |\n        +------+---------+-------+---------------+----------------+\n        [10000 rows x 5 columns]\n\n\n        The groupby function can take a variable length list of aggregation\n        specifiers so if we want the count and the 0.25 and 0.75 quantiles of\n        ratings:\n\n        >>> user_rating_stats = sf.groupby([\'user_id\', \'time\'], agg.COUNT(),\n        ...                                {\'rating_quantiles\': agg.QUANTILE(\'rating\',[0.25, 0.75])})\n        >>> user_rating_stats\n        +------+---------+-------+------------------------+\n        | time | user_id | Count |    rating_quantiles    |\n        +------+---------+-------+------------------------+\n        | 2006 |  61285  |   1   | array(\'d\', [4.0, 4.0]) |\n        | 2000 |  36078  |   1   | array(\'d\', [4.0, 4.0]) |\n        | 2003 |  47158  |   1   | array(\'d\', [3.0, 3.0]) |\n        | 2007 |  34446  |   1   | array(\'d\', [3.0, 3.0]) |\n        | 2010 |  47990  |   1   | array(\'d\', [3.0, 3.0]) |\n        | 2003 |  42120  |   1   | array(\'d\', [5.0, 5.0]) |\n        | 2007 |  44940  |   1   | array(\'d\', [4.0, 4.0]) |\n        | 2008 |  58240  |   1   | array(\'d\', [4.0, 4.0]) |\n        | 2002 |   102   |   1   | array(\'d\', [1.0, 1.0]) |\n        | 2009 |  52708  |   1   | array(\'d\', [3.0, 3.0]) |\n        | ...  |   ...   |  ...  |          ...           |\n        +------+---------+-------+------------------------+\n        [10000 rows x 4 columns]\n\n        To put all items a user rated into one list value by their star rating:\n\n        >>> user_rating_stats = sf.groupby(["user_id", "rating"],\n        ...                                {"rated_movie_ids":agg.CONCAT("movie_id")})\n        >>> user_rating_stats\n        +--------+---------+----------------------+\n        | rating | user_id |     rated_movie_ids  |\n        +--------+---------+----------------------+\n        |   3    |  31434  | array(\'d\', [1663.0]) |\n        |   5    |  25944  | array(\'d\', [1663.0]) |\n        |   4    |  38827  | array(\'d\', [1663.0]) |\n        |   4    |  51437  | array(\'d\', [1663.0]) |\n        |   4    |  42549  | array(\'d\', [1663.0]) |\n        |   4    |  49532  | array(\'d\', [1663.0]) |\n        |   3    |  26124  | array(\'d\', [1663.0]) |\n        |   4    |  46336  | array(\'d\', [1663.0]) |\n        |   4    |  52133  | array(\'d\', [1663.0]) |\n        |   5    |  62361  | array(\'d\', [1663.0]) |\n        |  ...   |   ...   |         ...          |\n        +--------+---------+----------------------+\n        [9952 rows x 3 columns]\n\n        To put all items and rating of a given user together into a dictionary\n        value:\n\n        >>> user_rating_stats = sf.groupby("user_id",\n        ...                                {"movie_rating":agg.CONCAT("movie_id", "rating")})\n        >>> user_rating_stats\n        +---------+--------------+\n        | user_id | movie_rating |\n        +---------+--------------+\n        |  62361  |  {1663: 5}   |\n        |  30727  |  {1663: 4}   |\n        |  40111  |  {1663: 2}   |\n        |  50513  |  {1663: 4}   |\n        |  35140  |  {1663: 4}   |\n        |  42352  |  {1663: 5}   |\n        |  29667  |  {1663: 4}   |\n        |  46242  |  {1663: 5}   |\n        |  58310  |  {1663: 2}   |\n        |  64614  |  {1663: 2}   |\n        |   ...   |     ...      |\n        +---------+--------------+\n        [9852 rows x 2 columns]\n        '
        if isinstance(key_column_names, str):
            key_column_names = [key_column_names]
        my_column_names = self.column_names()
        key_columns_array = []
        for column in key_column_names:
            if not isinstance(column, str):
                raise TypeError('Column name must be a string')
            if column not in my_column_names:
                raise KeyError('Column "' + column + '" does not exist in SFrame')
            if self[column].dtype == dict:
                raise TypeError('Cannot group on a dictionary column.')
            key_columns_array.append(column)
        group_output_columns = []
        group_columns = []
        group_ops = []
        all_ops = [operations] + list(args)
        for op_entry in all_ops:
            operation = op_entry
            if not (isinstance(operation, list) or isinstance(operation, dict)):
                operation = [operation]
            if isinstance(operation, dict):
                for key in operation:
                    val = operation[key]
                    if type(val) is tuple:
                        (op, column) = val
                        if op == '__builtin__avg__' and self[column[0]].dtype in [array.array, numpy.ndarray]:
                            op = '__builtin__vector__avg__'
                        if op == '__builtin__sum__' and self[column[0]].dtype in [array.array, numpy.ndarray]:
                            op = '__builtin__vector__sum__'
                        if (op == '__builtin__argmax__' or op == '__builtin__argmin__') and (type(column[0]) is tuple) != (type(key) is tuple):
                            raise TypeError('Output column(s) and aggregate column(s) for aggregate operation should be either all tuple or all string.')
                        if (op == '__builtin__argmax__' or op == '__builtin__argmin__') and type(column[0]) is tuple:
                            for (col, output) in zip(column[0], key):
                                group_columns = group_columns + [[col, column[1]]]
                                group_ops = group_ops + [op]
                                group_output_columns = group_output_columns + [output]
                        else:
                            group_columns = group_columns + [column]
                            group_ops = group_ops + [op]
                            group_output_columns = group_output_columns + [key]
                        if op == '__builtin__concat__dict__':
                            key_column = column[0]
                            key_column_type = self.select_column(key_column).dtype
                            if not key_column_type in (int, float, str):
                                raise TypeError('CONCAT key column must be int, float or str type')
                    elif val == aggregate.COUNT:
                        group_output_columns = group_output_columns + [key]
                        val = aggregate.COUNT()
                        (op, column) = val
                        group_columns = group_columns + [column]
                        group_ops = group_ops + [op]
                    else:
                        raise TypeError('Unexpected type in aggregator definition of output column: ' + key)
            elif isinstance(operation, list):
                for val in operation:
                    if type(val) is tuple:
                        (op, column) = val
                        if op == '__builtin__avg__' and self[column[0]].dtype in [array.array, numpy.ndarray]:
                            op = '__builtin__vector__avg__'
                        if op == '__builtin__sum__' and self[column[0]].dtype in [array.array, numpy.ndarray]:
                            op = '__builtin__vector__sum__'
                        if (op == '__builtin__argmax__' or op == '__builtin__argmin__') and type(column[0]) is tuple:
                            for col in column[0]:
                                group_columns = group_columns + [[col, column[1]]]
                                group_ops = group_ops + [op]
                                group_output_columns = group_output_columns + ['']
                        else:
                            group_columns = group_columns + [column]
                            group_ops = group_ops + [op]
                            group_output_columns = group_output_columns + ['']
                        if op == '__builtin__concat__dict__':
                            key_column = column[0]
                            key_column_type = self.select_column(key_column).dtype
                            if not key_column_type in (int, float, str):
                                raise TypeError('CONCAT key column must be int, float or str type')
                    elif val == aggregate.COUNT:
                        group_output_columns = group_output_columns + ['']
                        val = aggregate.COUNT()
                        (op, column) = val
                        group_columns = group_columns + [column]
                        group_ops = group_ops + [op]
                    else:
                        raise TypeError('Unexpected type in aggregator definition.')
        for (cols, op) in zip(group_columns, group_ops):
            for col in cols:
                if not isinstance(col, str):
                    raise TypeError('Column name must be a string')
            if not isinstance(op, str):
                raise TypeError('Operation type not recognized.')
            if op is not aggregate.COUNT()[0]:
                for col in cols:
                    if col not in my_column_names:
                        raise KeyError('Column ' + col + ' does not exist in SFrame')
        with cython_context():
            return SFrame(_proxy=self.__proxy__.groupby_aggregate(key_columns_array, group_columns, group_output_columns, group_ops))

    def join(self, right, on=None, how='inner', alter_name=None):
        if False:
            print('Hello World!')
        "\n        Merge two SFrames. Merges the current (left) SFrame with the given\n        (right) SFrame using a SQL-style equi-join operation by columns.\n\n        Parameters\n        ----------\n        right : SFrame\n            The SFrame to join.\n\n        on : None | str | list | dict, optional\n            The column name(s) representing the set of join keys.  Each row that\n            has the same value in this set of columns will be merged together.\n\n            * If 'None' is given, join will use all columns that have the same\n              name as the set of join keys.\n\n            * If a str is given, this is interpreted as a join using one column,\n              where both SFrames have the same column name.\n\n            * If a list is given, this is interpreted as a join using one or\n              more column names, where each column name given exists in both\n              SFrames.\n\n            * If a dict is given, each dict key is taken as a column name in the\n              left SFrame, and each dict value is taken as the column name in\n              right SFrame that will be joined together. e.g.\n              {'left_col_name':'right_col_name'}.\n\n        how : {'left', 'right', 'outer', 'inner'}, optional\n            The type of join to perform.  'inner' is default.\n\n            * inner: Equivalent to a SQL inner join.  Result consists of the\n              rows from the two frames whose join key values match exactly,\n              merged together into one SFrame.\n\n            * left: Equivalent to a SQL left outer join. Result is the union\n              between the result of an inner join and the rest of the rows from\n              the left SFrame, merged with missing values.\n\n            * right: Equivalent to a SQL right outer join.  Result is the union\n              between the result of an inner join and the rest of the rows from\n              the right SFrame, merged with missing values.\n\n            * outer: Equivalent to a SQL full outer join. Result is\n              the union between the result of a left outer join and a right\n              outer join.\n\n        alter_name : None | dict\n            user provided names to resolve column name conflict when merging two sframe.\n\n            * 'None', then default conflict resolution will be used. For example, if 'X' is\n            defined in the sframe on the left side of join, and there's an column also called\n            'X' in the sframe on the right, 'X.1' will be used as the new column name when\n            appending the column 'X' from the right sframe, in order to avoid column name collision.\n\n            * if a dict is given, the dict key should be obtained from column names from the right\n            sframe. The dict value should be user preferred column name to resolve the name collision\n            instead of resolving by the default behavior. In general, dict key should not be any value\n            from the right sframe column names. If dict value will cause potential name confict\n            after an attempt to resolve, exception will be thrown.\n\n        Returns\n        -------\n        out : SFrame\n\n        Examples\n        --------\n        >>> animals = turicreate.SFrame({'id': [1, 2, 3, 4],\n        ...                           'name': ['dog', 'cat', 'sheep', 'cow']})\n        >>> sounds = turicreate.SFrame({'id': [1, 3, 4, 5],\n        ...                          'sound': ['woof', 'baa', 'moo', 'oink']})\n        >>> animals.join(sounds, how='inner')\n        +----+-------+-------+\n        | id |  name | sound |\n        +----+-------+-------+\n        | 1  |  dog  |  woof |\n        | 3  | sheep |  baa  |\n        | 4  |  cow  |  moo  |\n        +----+-------+-------+\n        [3 rows x 3 columns]\n\n        >>> animals.join(sounds, on='id', how='left')\n        +----+-------+-------+\n        | id |  name | sound |\n        +----+-------+-------+\n        | 1  |  dog  |  woof |\n        | 3  | sheep |  baa  |\n        | 4  |  cow  |  moo  |\n        | 2  |  cat  |  None |\n        +----+-------+-------+\n        [4 rows x 3 columns]\n\n        >>> animals.join(sounds, on=['id'], how='right')\n        +----+-------+-------+\n        | id |  name | sound |\n        +----+-------+-------+\n        | 1  |  dog  |  woof |\n        | 3  | sheep |  baa  |\n        | 4  |  cow  |  moo  |\n        | 5  |  None |  oink |\n        +----+-------+-------+\n        [4 rows x 3 columns]\n\n        >>> animals.join(sounds, on={'id':'id'}, how='outer')\n        +----+-------+-------+\n        | id |  name | sound |\n        +----+-------+-------+\n        | 1  |  dog  |  woof |\n        | 3  | sheep |  baa  |\n        | 4  |  cow  |  moo  |\n        | 5  |  None |  oink |\n        | 2  |  cat  |  None |\n        +----+-------+-------+\n        [5 rows x 3 columns]\n        "
        available_join_types = ['left', 'right', 'outer', 'inner']
        if not isinstance(right, SFrame):
            raise TypeError('Can only join two SFrames')
        if how not in available_join_types:
            raise ValueError('Invalid join type')
        if self.num_columns() <= 0 or right.num_columns() <= 0:
            raise ValueError('Cannot join an SFrame with no columns.')
        join_keys = dict()
        if on is None:
            left_names = self.column_names()
            right_names = right.column_names()
            common_columns = [name for name in left_names if name in right_names]
            for name in common_columns:
                join_keys[name] = name
        elif type(on) is str:
            join_keys[on] = on
        elif type(on) is list:
            for name in on:
                if type(name) is not str:
                    raise TypeError('Join keys must each be a str.')
                join_keys[name] = name
        elif type(on) is dict:
            join_keys = on
        else:
            raise TypeError('Must pass a str, list, or dict of join keys')
        with cython_context():
            if alter_name is None:
                return SFrame(_proxy=self.__proxy__.join(right.__proxy__, how, join_keys))
            if type(alter_name) is dict:
                left_names = self.column_names()
                right_names = right.column_names()
                for (k, v) in alter_name.items():
                    if k not in right_names or k in join_keys:
                        raise KeyError('Redundant key %s for collision resolution' % k)
                    if k == v:
                        raise ValueError('Key %s should not be equal to value' % k)
                    if v in left_names or v in right_names:
                        raise ValueError('Value %s will cause further collision' % v)
                return SFrame(_proxy=self.__proxy__.join_with_custom_name(right.__proxy__, how, join_keys, alter_name))

    def filter_by(self, values, column_name, exclude=False):
        if False:
            i = 10
            return i + 15
        "\n        Filter an SFrame by values inside an iterable object. Result is an\n        SFrame that only includes (or excludes) the rows that have a column\n        with the given ``column_name`` which holds one of the values in the\n        given ``values`` :class:`~turicreate.SArray`. If ``values`` is not an\n        SArray, we attempt to convert it to one before filtering.\n\n        Parameters\n        ----------\n        values : SArray | list | numpy.ndarray | pandas.Series | str | map\n        | generator | filter | None | range\n            The values to use to filter the SFrame.  The resulting SFrame will\n            only include rows that have one of these values in the given\n            column.\n\n        column_name : str\n            The column of the SFrame to match with the given `values`.\n\n        exclude : bool\n            If True, the result SFrame will contain all rows EXCEPT those that\n            have one of ``values`` in ``column_name``.\n\n        Returns\n        -------\n        out : SFrame\n            The filtered SFrame.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id': [1, 2, 3, 4],\n        ...                      'animal_type': ['dog', 'cat', 'cow', 'horse'],\n        ...                      'name': ['bob', 'jim', 'jimbob', 'bobjim']})\n        >>> household_pets = ['cat', 'hamster', 'dog', 'fish', 'bird', 'snake']\n        >>> sf.filter_by(household_pets, 'animal_type')\n        +-------------+----+------+\n        | animal_type | id | name |\n        +-------------+----+------+\n        |     dog     | 1  | bob  |\n        |     cat     | 2  | jim  |\n        +-------------+----+------+\n        [2 rows x 3 columns]\n        >>> sf.filter_by(household_pets, 'animal_type', exclude=True)\n        +-------------+----+--------+\n        | animal_type | id |  name  |\n        +-------------+----+--------+\n        |    horse    | 4  | bobjim |\n        |     cow     | 3  | jimbob |\n        +-------------+----+--------+\n        [2 rows x 3 columns]\n\n        >>> sf.filter_by(None, 'name', exclude=True)\n        +-------------+----+--------+\n        | animal_type | id |  name  |\n        +-------------+----+--------+\n        |     dog     | 1  |  bob   |\n        |     cat     | 2  |  jim   |\n        |     cow     | 3  | jimbob |\n        |    horse    | 4  | bobjim |\n        +-------------+----+--------+\n        [4 rows x 3 columns]\n\n        >>> sf.filter_by(filter(lambda x : len(x) > 3, sf['name']), 'name', exclude=True)\n        +-------------+----+--------+\n        | animal_type | id |  name  |\n        +-------------+----+--------+\n        |     dog     | 1  |  bob   |\n        |     cat     | 2  |  jim   |\n        +-------------+----+--------+\n        [2 rows x 3 columns]\n\n        >>> sf.filter_by(range(3), 'id', exclude=True)\n        +-------------+----+--------+\n        | animal_type | id |  name  |\n        +-------------+----+--------+\n        |     cow     | 3  | jimbob |\n        |    horse    | 4  | bobjim |\n        +-------------+----+--------+\n        [2 rows x 3 columns]\n        "
        if type(column_name) is not str:
            raise TypeError('Must pass a str as column_name')
        existing_columns = self.column_names()
        if column_name not in existing_columns:
            raise KeyError("Column '" + column_name + "' not in SFrame.")
        existing_type = self[column_name].dtype
        if type(values) is not SArray:
            if not _is_non_string_iterable(values):
                values = [values]
            elif SArray._is_iterable_required_to_listify(values):
                values = list(values)
            if all((val is None for val in values)):
                values = SArray(values, existing_type)
            else:
                values = SArray(values)
        value_sf = SFrame()
        value_sf.add_column(values, column_name, inplace=True)
        given_type = value_sf.column_types()[0]
        if given_type != existing_type:
            raise TypeError(("Type of given values ({0}) does not match type of column '" + column_name + "' ({1}) in SFrame.").format(given_type, existing_type))
        value_sf = value_sf.groupby(column_name, {})
        with cython_context():
            if exclude:
                id_name = 'id'
                while id_name in existing_columns:
                    id_name += '1'
                value_sf = value_sf.add_row_number(id_name)
                tmp = SFrame(_proxy=self.__proxy__.join(value_sf.__proxy__, 'left', {column_name: column_name}))
                ret_sf = tmp[tmp[id_name] == None]
                del ret_sf[id_name]
                return ret_sf
            else:
                return SFrame(_proxy=self.__proxy__.join(value_sf.__proxy__, 'inner', {column_name: column_name}))

    def explore(self, title=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Explore the SFrame in an interactive GUI. Opens a new app window.\n\n        Parameters\n        ----------\n        title : str\n            The plot title to show for the resulting visualization. Defaults to None.\n            If the title is None, a default title will be provided.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n        Suppose \'sf\' is an SFrame, we can view it using:\n\n        >>> sf.explore()\n\n        To override the default plot title and axis labels:\n\n        >>> sf.explore(title="My Plot Title")\n        '
        import sys
        if sys.platform != 'darwin' and sys.platform != 'linux2' and (sys.platform != 'linux'):
            raise NotImplementedError('Visualization is currently supported only on macOS and Linux.')
        from ..visualization._plot import _target, display_table_in_notebook, _ensure_web_server
        if _target == 'none':
            return
        if title is None:
            title = ''
        if _target == 'browser':
            _ensure_web_server()
            import webbrowser
            import turicreate as tc
            url = tc.extensions.get_url_for_table(self, title)
            webbrowser.open_new_tab(url)
            return
        try:
            if _target == 'auto' and (get_ipython().__class__.__name__ == 'ZMQInteractiveShell' or get_ipython().__class__.__name__ == 'Shell'):
                display_table_in_notebook(self, title)
                return
        except NameError:
            pass
        path_to_client = _get_client_app_path()
        self.__proxy__.explore(path_to_client, title)

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Visualize a summary of each column in an SFrame. Opens a new app window.\n\n        Notes\n        -----\n        - The plot will render either inline in a Jupyter Notebook, in a web\n          browser, or in a native GUI window, depending on the value provided in\n          `turicreate.visualization.set_target` (defaults to 'auto').\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n        Suppose 'sf' is an SFrame, we can view it using:\n\n        >>> sf.show()\n        "
        returned_plot = self.plot()
        returned_plot.show()

    def plot(self):
        if False:
            print('Hello World!')
        "\n        Create a Plot object that contains a summary of each column\n        in an SFrame.\n\n        Returns\n        -------\n        out : Plot\n        A :class: Plot object that is the columnwise summary of the sframe.\n\n        Examples\n        --------\n        Suppose 'sf' is an SFrame, we can make a plot object as:\n\n        >>> plt = sf.plot()\n\n        We can then visualize the plot using:\n\n        >>> plt.show()\n        "
        return Plot(_proxy=self.__proxy__.plot())

    def pack_columns(self, column_names=None, column_name_prefix=None, dtype=list, fill_na=None, remove_prefix=True, new_column_name=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pack columns of the current SFrame into one single column. The result\n        is a new SFrame with the unaffected columns from the original SFrame\n        plus the newly created column.\n\n        The list of columns that are packed is chosen through either the\n        ``column_names`` or ``column_name_prefix`` parameter. Only one of the parameters\n        is allowed to be provided. ``columns_names`` explicitly specifies the list of\n        columns to pack, while ``column_name_prefix`` specifies that all columns that\n        have the given prefix are to be packed.\n\n        The type of the resulting column is decided by the ``dtype`` parameter.\n        Allowed values for ``dtype`` are dict, array.array and list:\n\n         - *dict*: pack to a dictionary SArray where column name becomes\n           dictionary key and column value becomes dictionary value\n\n         - *array.array*: pack all values from the packing columns into an array\n\n         - *list*: pack all values from the packing columns into a list.\n\n        Parameters\n        ----------\n        column_names : list[str], optional\n            A list of column names to be packed.  If omitted and\n            `column_name_prefix` is not specified, all columns from current SFrame\n            are packed.  This parameter is mutually exclusive with the\n            `column_name_prefix` parameter.\n\n        column_name_prefix : str, optional\n            Pack all columns with the given `column_name_prefix`.\n            This parameter is mutually exclusive with the `columns_names` parameter.\n\n        dtype : dict | array.array | list, optional\n            The resulting packed column type. If not provided, dtype is list.\n\n        fill_na : value, optional\n            Value to fill into packed column if missing value is encountered.\n            If packing to dictionary, `fill_na` is only applicable to dictionary\n            values; missing keys are not replaced.\n\n        remove_prefix : bool, optional\n            If True and `column_name_prefix` is specified, the dictionary key will\n            be constructed by removing the prefix from the column name.\n            This option is only applicable when packing to dict type.\n\n        new_column_name : str, optional\n            Packed column name.  If not given and `column_name_prefix` is given,\n            then the prefix will be used as the new column name, otherwise name\n            is generated automatically.\n\n        Returns\n        -------\n        out : SFrame\n            An SFrame that contains columns that are not packed, plus the newly\n            packed column.\n\n        See Also\n        --------\n        unpack\n\n        Notes\n        -----\n        - If packing to dictionary, missing key is always dropped. Missing\n          values are dropped if fill_na is not provided, otherwise, missing\n          value is replaced by \'fill_na\'. If packing to list or array, missing\n          values will be kept. If \'fill_na\' is provided, the missing value is\n          replaced with \'fill_na\' value.\n\n        Examples\n        --------\n        Suppose \'sf\' is an an SFrame that maintains business category\n        information:\n\n        >>> sf = turicreate.SFrame({\'business\': range(1, 5),\n        ...                       \'category.retail\': [1, None, 1, None],\n        ...                       \'category.food\': [1, 1, None, None],\n        ...                       \'category.service\': [None, 1, 1, None],\n        ...                       \'category.shop\': [1, 1, None, 1]})\n        >>> sf\n        +----------+-----------------+---------------+------------------+---------------+\n        | business | category.retail | category.food | category.service | category.shop |\n        +----------+-----------------+---------------+------------------+---------------+\n        |    1     |        1        |       1       |       None       |       1       |\n        |    2     |       None      |       1       |        1         |       1       |\n        |    3     |        1        |      None     |        1         |      None     |\n        |    4     |       None      |       1       |       None       |       1       |\n        +----------+-----------------+---------------+------------------+---------------+\n        [4 rows x 5 columns]\n\n        To pack all category columns into a list:\n\n        >>> sf.pack_columns(column_name_prefix=\'category\')\n        +----------+-----------------------+\n        | business |        category       |\n        +----------+-----------------------+\n        |    1     |    [1, 1, None, 1]    |\n        |    2     |    [1, None, 1, 1]    |\n        |    3     |   [None, 1, 1, None]  |\n        |    4     | [None, None, None, 1] |\n        +----------+-----------------------+\n        [4 rows x 2 columns]\n\n        To pack all category columns into a dictionary, with new column name:\n\n        >>> sf.pack_columns(column_name_prefix=\'category\', dtype=dict,\n        ...                 new_column_name=\'new name\')\n        +----------+-------------------------------+\n        | business |            new name           |\n        +----------+-------------------------------+\n        |    1     | {\'food\': 1, \'shop\': 1, \'re... |\n        |    2     | {\'food\': 1, \'shop\': 1, \'se... |\n        |    3     |  {\'retail\': 1, \'service\': 1}  |\n        |    4     |          {\'shop\': 1}          |\n        +----------+-------------------------------+\n        [4 rows x 2 columns]\n\n        To keep column prefix in the resulting dict key:\n\n        >>> sf.pack_columns(column_name_prefix=\'category\', dtype=dict,\n                            remove_prefix=False)\n        +----------+-------------------------------+\n        | business |            category           |\n        +----------+-------------------------------+\n        |    1     | {\'category.retail\': 1, \'ca... |\n        |    2     | {\'category.food\': 1, \'cate... |\n        |    3     | {\'category.retail\': 1, \'ca... |\n        |    4     |      {\'category.shop\': 1}     |\n        +----------+-------------------------------+\n        [4 rows x 2 columns]\n\n        To explicitly pack a set of columns:\n\n        >>> sf.pack_columns(column_names = [\'business\', \'category.retail\',\n                                       \'category.food\', \'category.service\',\n                                       \'category.shop\'])\n        +-----------------------+\n        |           X1          |\n        +-----------------------+\n        |   [1, 1, 1, None, 1]  |\n        |   [2, None, 1, 1, 1]  |\n        | [3, 1, None, 1, None] |\n        | [4, None, 1, None, 1] |\n        +-----------------------+\n        [4 rows x 1 columns]\n\n        To pack all columns with name starting with \'category\' into an array\n        type, and with missing value replaced with 0:\n\n        >>> import array\n        >>> sf.pack_columns(column_name_prefix="category", dtype=array.array,\n        ...                 fill_na=0)\n        +----------+----------------------+\n        | business |       category       |\n        +----------+----------------------+\n        |    1     | [1.0, 1.0, 0.0, 1.0] |\n        |    2     | [1.0, 0.0, 1.0, 1.0] |\n        |    3     | [0.0, 1.0, 1.0, 0.0] |\n        |    4     | [0.0, 0.0, 0.0, 1.0] |\n        +----------+----------------------+\n        [4 rows x 2 columns]\n        '
        if column_names is not None and column_name_prefix is not None:
            raise ValueError("'column_names' and 'column_name_prefix' parameter cannot be given at the same time.")
        if new_column_name is None and column_name_prefix is not None:
            new_column_name = column_name_prefix
        if column_name_prefix is not None:
            if type(column_name_prefix) != str:
                raise TypeError("'column_name_prefix' must be a string")
            column_names = [name for name in self.column_names() if name.startswith(column_name_prefix)]
            if len(column_names) == 0:
                raise ValueError("There is no column starts with prefix '" + column_name_prefix + "'")
        elif column_names is None:
            column_names = self.column_names()
        else:
            if not _is_non_string_iterable(column_names):
                raise TypeError('column_names must be an iterable type')
            column_name_set = set(self.column_names())
            for column in column_names:
                if column not in column_name_set:
                    raise ValueError("Current SFrame has no column called '" + str(column) + "'.")
            if len(set(column_names)) != len(column_names):
                raise ValueError('There is duplicate column names in column_names parameter')
        if dtype not in (dict, list, array.array):
            raise ValueError('Resulting dtype has to be one of dict/array.array/list type')
        if dtype == array.array:
            if fill_na is not None and type(fill_na) not in (int, float):
                raise ValueError('fill_na value for array needs to be numeric type')
            for column in column_names:
                if self[column].dtype not in (int, float):
                    raise TypeError("Column '" + column + "' type is not numeric, cannot pack into array type")
        if dtype == dict and column_name_prefix is not None and (remove_prefix == True):
            size_prefix = len(column_name_prefix)
            first_char = set([c[size_prefix:size_prefix + 1] for c in column_names])
            if len(first_char) == 1 and first_char.pop() in ['.', '-', '_']:
                dict_keys = [name[size_prefix + 1:] for name in column_names]
            else:
                dict_keys = [name[size_prefix:] for name in column_names]
        else:
            dict_keys = column_names
        rest_columns = [name for name in self.column_names() if name not in column_names]
        if new_column_name is not None:
            if type(new_column_name) != str:
                raise TypeError("'new_column_name' has to be a string")
            if new_column_name in rest_columns:
                raise KeyError('Current SFrame already contains a column name ' + new_column_name)
        else:
            new_column_name = ''
        ret_sa = None
        with cython_context():
            ret_sa = SArray(_proxy=self.__proxy__.pack_columns(column_names, dict_keys, dtype, fill_na))
        new_sf = self.select_columns(rest_columns)
        new_sf.add_column(ret_sa, new_column_name, inplace=True)
        return new_sf

    def split_datetime(self, column_name, column_name_prefix=None, limit=None, timezone=False):
        if False:
            while True:
                i = 10
        "\n        Splits a datetime column of SFrame to multiple columns, with each value in a\n        separate column. Returns a new SFrame with the expanded column replaced with\n        a list of new columns. The expanded column must be of datetime type.\n\n        For more details regarding name generation and\n        other, refer to :py:func:`turicreate.SArray.split_datetime()`\n\n        Parameters\n        ----------\n        column_name : str\n            Name of the unpacked column.\n\n        column_name_prefix : str, optional\n            If provided, expanded column names would start with the given prefix.\n            If not provided, the default value is the name of the expanded column.\n\n        limit: list[str], optional\n            Limits the set of datetime elements to expand.\n            Possible values are 'year','month','day','hour','minute','second',\n            'weekday', 'isoweekday', 'tmweekday', and 'us'.\n            If not provided, only ['year','month','day','hour','minute','second']\n            are expanded.\n\n        timezone : bool, optional\n            A boolean parameter that determines whether to show the timezone\n            column or not. Defaults to False.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that contains rest of columns from original SFrame with\n            the given column replaced with a collection of expanded columns.\n\n        Examples\n        ---------\n\n        >>> sf\n        Columns:\n            id   int\n            submission  datetime\n        Rows: 2\n        Data:\n            +----+-------------------------------------------------+\n            | id |               submission                        |\n            +----+-------------------------------------------------+\n            | 1  | datetime(2011, 1, 21, 7, 17, 21, tzinfo=GMT(+1))|\n            | 2  | datetime(2011, 1, 21, 5, 43, 21, tzinfo=GMT(+1))|\n            +----+-------------------------------------------------+\n\n        >>> sf.split_datetime('submission',limit=['hour','minute'])\n        Columns:\n            id  int\n            submission.hour int\n            submission.minute int\n        Rows: 2\n        Data:\n        +----+-----------------+-------------------+\n        | id | submission.hour | submission.minute |\n        +----+-----------------+-------------------+\n        | 1  |        7        |        17         |\n        | 2  |        5        |        43         |\n        +----+-----------------+-------------------+\n        "
        if column_name not in self.column_names():
            raise KeyError("column '" + column_name + "' does not exist in current SFrame")
        if column_name_prefix is None:
            column_name_prefix = column_name
        new_sf = self[column_name].split_datetime(column_name_prefix, limit, timezone)
        rest_columns = [name for name in self.column_names() if name != column_name]
        new_names = new_sf.column_names()
        while set(new_names).intersection(rest_columns):
            new_names = [name + '.1' for name in new_names]
        new_sf.rename(dict(list(zip(new_sf.column_names(), new_names))), inplace=True)
        ret_sf = self.select_columns(rest_columns)
        ret_sf.add_columns(new_sf, inplace=True)
        return ret_sf

    def unpack(self, column_name=None, column_name_prefix=None, column_types=None, na_value=None, limit=None):
        if False:
            print('Hello World!')
        '\n        Expand one column of this SFrame to multiple columns with each value in\n        a separate column. Returns a new SFrame with the unpacked column\n        replaced with a list of new columns.  The column must be of\n        list/array/dict type.\n\n        For more details regarding name generation, missing value handling and\n        other, refer to the SArray version of\n        :py:func:`~turicreate.SArray.unpack()`.\n\n        Parameters\n        ----------\n        column_name : str, optional\n            Name of the unpacked column, if provided. If not provided\n            and only one column is present then the column is unpacked.\n            In case of multiple columns, name must be provided to know\n            which column to be unpacked.\n\n\n        column_name_prefix : str, optional\n            If provided, unpacked column names would start with the given\n            prefix. If not provided, default value is the name of the unpacked\n            column.\n\n        column_types : [type], optional\n            Column types for the unpacked columns.\n            If not provided, column types are automatically inferred from first\n            100 rows. For array type, default column types are float.  If\n            provided, column_types also restricts how many columns to unpack.\n\n        na_value : flexible_type, optional\n            If provided, convert all values that are equal to "na_value" to\n            missing value (None).\n\n        limit : list[str] | list[int], optional\n            Control unpacking only a subset of list/array/dict value. For\n            dictionary SArray, `limit` is a list of dictionary keys to restrict.\n            For list/array SArray, `limit` is a list of integers that are\n            indexes into the list/array value.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that contains rest of columns from original SFrame with\n            the given column replaced with a collection of unpacked columns.\n\n        See Also\n        --------\n        pack_columns, SArray.unpack\n\n        Examples\n        ---------\n        >>> sf = turicreate.SFrame({\'id\': [1,2,3],\n        ...                      \'wc\': [{\'a\': 1}, {\'b\': 2}, {\'a\': 1, \'b\': 2}]})\n        +----+------------------+\n        | id |        wc        |\n        +----+------------------+\n        | 1  |     {\'a\': 1}     |\n        | 2  |     {\'b\': 2}     |\n        | 3  | {\'a\': 1, \'b\': 2} |\n        +----+------------------+\n        [3 rows x 2 columns]\n\n        >>> sf.unpack(\'wc\')\n        +----+------+------+\n        | id | wc.a | wc.b |\n        +----+------+------+\n        | 1  |  1   | None |\n        | 2  | None |  2   |\n        | 3  |  1   |  2   |\n        +----+------+------+\n        [3 rows x 3 columns]\n\n        To not have prefix in the generated column name:\n\n        >>> sf.unpack(\'wc\', column_name_prefix="")\n        +----+------+------+\n        | id |  a   |  b   |\n        +----+------+------+\n        | 1  |  1   | None |\n        | 2  | None |  2   |\n        | 3  |  1   |  2   |\n        +----+------+------+\n        [3 rows x 3 columns]\n\n        To limit subset of keys to unpack:\n\n        >>> sf.unpack(\'wc\', limit=[\'b\'])\n        +----+------+\n        | id | wc.b |\n        +----+------+\n        | 1  | None |\n        | 2  |  2   |\n        | 3  |  2   |\n        +----+------+\n        [3 rows x 3 columns]\n\n        To unpack an array column:\n\n        >>> import array\n        >>> sf = turicreate.SFrame({\'id\': [1,2,3],\n        ...                       \'friends\': [array.array(\'d\', [1.0, 2.0, 3.0]),\n        ...                                   array.array(\'d\', [2.0, 3.0, 4.0]),\n        ...                                   array.array(\'d\', [3.0, 4.0, 5.0])]})\n        >>> sf\n        +-----------------+----+\n        |     friends     | id |\n        +-----------------+----+\n        | [1.0, 2.0, 3.0] | 1  |\n        | [2.0, 3.0, 4.0] | 2  |\n        | [3.0, 4.0, 5.0] | 3  |\n        +-----------------+----+\n        [3 rows x 2 columns]\n\n        >>> sf.unpack(\'friends\')\n        +----+-----------+-----------+-----------+\n        | id | friends.0 | friends.1 | friends.2 |\n        +----+-----------+-----------+-----------+\n        | 1  |    1.0    |    2.0    |    3.0    |\n        | 2  |    2.0    |    3.0    |    4.0    |\n        | 3  |    3.0    |    4.0    |    5.0    |\n        +----+-----------+-----------+-----------+\n        [3 rows x 4 columns]\n\n        >>> sf = turicreate.SFrame([{\'a\':1,\'b\':2,\'c\':3},{\'a\':4,\'b\':5,\'c\':6}])\n        >>> sf.unpack()\n        +---+---+---+\n        | a | b | c |\n        +---+---+---+\n        | 1 | 2 | 3 |\n        | 4 | 5 | 6 |\n        +---+---+---+\n        [2 rows x 3 columns]\n\n\n\n        '
        if column_name is None:
            if self.num_columns() == 0:
                raise RuntimeError('No column exists in the current SFrame')
            for t in range(self.num_columns()):
                column_type = self.column_types()[t]
                if column_type == dict or column_type == list or column_type == array.array:
                    if column_name is None:
                        column_name = self.column_names()[t]
                    else:
                        raise RuntimeError('Column name needed to unpack')
            if column_name is None:
                raise RuntimeError('No columns can be unpacked')
            elif column_name_prefix is None:
                column_name_prefix = ''
        elif column_name not in self.column_names():
            raise KeyError("Column '" + column_name + "' does not exist in current SFrame")
        if column_name_prefix is None:
            column_name_prefix = column_name
        new_sf = self[column_name].unpack(column_name_prefix, column_types, na_value, limit)
        rest_columns = [name for name in self.column_names() if name != column_name]
        new_names = new_sf.column_names()
        while set(new_names).intersection(rest_columns):
            new_names = [name + '.1' for name in new_names]
        new_sf.rename(dict(list(zip(new_sf.column_names(), new_names))), inplace=True)
        ret_sf = self.select_columns(rest_columns)
        ret_sf.add_columns(new_sf, inplace=True)
        return ret_sf

    def stack(self, column_name, new_column_name=None, drop_na=False, new_column_type=None):
        if False:
            print('Hello World!')
        '\n        Convert a "wide" column of an SFrame to one or two "tall" columns by\n        stacking all values.\n\n        The stack works only for columns of dict, list, or array type.  If the\n        column is dict type, two new columns are created as a result of\n        stacking: one column holds the key and another column holds the value.\n        The rest of the columns are repeated for each key/value pair.\n\n        If the column is array or list type, one new column is created as a\n        result of stacking. With each row holds one element of the array or list\n        value, and the rest columns from the same original row repeated.\n\n        The returned SFrame includes the newly created column(s) and all\n        columns other than the one that is stacked.\n\n        Parameters\n        --------------\n        column_name : str\n            The column to stack. This column must be of dict/list/array type\n\n        new_column_name : str | list of str, optional\n            The new column name(s). If original column is list/array type,\n            new_column_name must a string. If original column is dict type,\n            new_column_name must be a list of two strings. If not given, column\n            names are generated automatically.\n\n        drop_na : boolean, optional\n            If True, missing values and empty list/array/dict are all dropped\n            from the resulting column(s). If False, missing values are\n            maintained in stacked column(s).\n\n        new_column_type : type | list of types, optional\n            The new column types. If original column is a list/array type\n            new_column_type must be a single type, or a list of one type. If\n            original column is of dict type, new_column_type must be a list of\n            two types. If not provided, the types are automatically inferred\n            from the first 100 values of the SFrame.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that contains newly stacked column(s) plus columns in\n            original SFrame other than the stacked column.\n\n        See Also\n        --------\n        unstack\n\n        Examples\n        ---------\n        Suppose \'sf\' is an SFrame that contains a column of dict type:\n\n        >>> sf = turicreate.SFrame({\'topic\':[1,2,3,4],\n        ...                       \'words\': [{\'a\':3, \'cat\':2},\n        ...                                 {\'a\':1, \'the\':2},\n        ...                                 {\'the\':1, \'dog\':3},\n        ...                                 {}]\n        ...                      })\n        +-------+----------------------+\n        | topic |        words         |\n        +-------+----------------------+\n        |   1   |  {\'a\': 3, \'cat\': 2}  |\n        |   2   |  {\'a\': 1, \'the\': 2}  |\n        |   3   | {\'the\': 1, \'dog\': 3} |\n        |   4   |          {}          |\n        +-------+----------------------+\n        [4 rows x 2 columns]\n\n        Stack would stack all keys in one column and all values in another\n        column:\n\n        >>> sf.stack(\'words\', new_column_name=[\'word\', \'count\'])\n        +-------+------+-------+\n        | topic | word | count |\n        +-------+------+-------+\n        |   1   |  a   |   3   |\n        |   1   | cat  |   2   |\n        |   2   |  a   |   1   |\n        |   2   | the  |   2   |\n        |   3   | the  |   1   |\n        |   3   | dog  |   3   |\n        |   4   | None |  None |\n        +-------+------+-------+\n        [7 rows x 3 columns]\n\n        Observe that since topic 4 had no words, an empty row is inserted.\n        To drop that row, set drop_na=True in the parameters to stack.\n\n        Suppose \'sf\' is an SFrame that contains a user and his/her friends,\n        where \'friends\' columns is an array type. Stack on \'friends\' column\n        would create a user/friend list for each user/friend pair:\n\n        >>> sf = turicreate.SFrame({\'topic\':[1,2,3],\n        ...                       \'friends\':[[2,3,4], [5,6],\n        ...                                  [4,5,10,None]]\n        ...                      })\n        >>> sf\n        +-------+------------------+\n        | topic |     friends      |\n        +-------+------------------+\n        |  1    |     [2, 3, 4]    |\n        |  2    |      [5, 6]      |\n        |  3    | [4, 5, 10, None] |\n        +----- -+------------------+\n        [3 rows x 2 columns]\n\n        >>> sf.stack(\'friends\', new_column_name=\'friend\')\n        +-------+--------+\n        | topic | friend |\n        +-------+--------+\n        |   1   |   2    |\n        |   1   |   3    |\n        |   1   |   4    |\n        |   2   |   5    |\n        |   2   |   6    |\n        |   3   |   4    |\n        |   3   |   5    |\n        |   3   |   10   |\n        |   3   |  None  |\n        +-------+--------+\n        [9 rows x 2 columns]\n\n        '
        column_name = str(column_name)
        if column_name not in self.column_names():
            raise ValueError("Cannot find column '" + str(column_name) + "' in the SFrame.")
        stack_column_type = self[column_name].dtype
        if stack_column_type not in [dict, array.array, list]:
            raise TypeError('Stack is only supported for column of dict/list/array type.')
        if new_column_type is not None:
            if type(new_column_type) is type:
                new_column_type = [new_column_type]
            if stack_column_type in [list, array.array] and len(new_column_type) != 1:
                raise ValueError('Expecting a single column type to unpack list or array columns')
            if stack_column_type in [dict] and len(new_column_type) != 2:
                raise ValueError('Expecting two column types to unpack a dict column')
        if new_column_name is not None:
            if stack_column_type == dict:
                if type(new_column_name) is not list:
                    raise TypeError('new_column_name has to be a list to stack dict type')
                elif len(new_column_name) != 2:
                    raise TypeError('new_column_name must have length of two')
            else:
                if type(new_column_name) != str:
                    raise TypeError('new_column_name has to be a str')
                new_column_name = [new_column_name]
            for name in new_column_name:
                if name in self.column_names() and name != column_name:
                    raise ValueError("Column with name '" + name + "' already exists, pick a new column name")
        elif stack_column_type == dict:
            new_column_name = ['', '']
        else:
            new_column_name = ['']
        head_row = SArray(self[column_name].head(100)).dropna()
        if len(head_row) == 0:
            raise ValueError('Cannot infer column type because there is not enough rows to infer value')
        if new_column_type is None:
            if stack_column_type == dict:
                keys = []
                values = []
                for row in head_row:
                    for val in row:
                        keys.append(val)
                        if val is not None:
                            values.append(row[val])
                new_column_type = [infer_type_of_list(keys), infer_type_of_list(values)]
            else:
                values = [v for v in itertools.chain.from_iterable(head_row)]
                new_column_type = [infer_type_of_list(values)]
        with cython_context():
            return SFrame(_proxy=self.__proxy__.stack(column_name, new_column_name, new_column_type, drop_na))

    def unstack(self, column_names, new_column_name=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Concatenate values from one or two columns into one column, grouping by\n        all other columns. The resulting column could be of type list, array or\n        dictionary.  If ``column_names`` is a numeric column, the result will be of\n        array.array type.  If ``column_names`` is a non-numeric column, the new column\n        will be of list type. If ``column_names`` is a list of two columns, the new\n        column will be of dict type where the keys are taken from the first\n        column in the list.\n\n        Parameters\n        ----------\n        column_names : str | [str, str]\n            The column(s) that is(are) to be concatenated.\n            If str, then collapsed column type is either array or list.\n            If [str, str], then collapsed column type is dict\n\n        new_column_name : str, optional\n            New column name. If not given, a name is generated automatically.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame containing the grouped columns as well as the new\n            column.\n\n        See Also\n        --------\n        stack : The inverse of unstack.\n\n        groupby : ``unstack`` is a special version of ``groupby`` that uses the\n          :mod:`~turicreate.aggregate.CONCAT` aggregator\n\n        Notes\n        -----\n        - There is no guarantee the resulting SFrame maintains the same order as\n          the original SFrame.\n\n        - Missing values are maintained during unstack.\n\n        - When unstacking into a dictionary, if there is more than one instance\n          of a given key for a particular group, an arbitrary value is selected.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'count':[4, 2, 1, 1, 2, None],\n        ...                       'topic':['cat', 'cat', 'dog', 'elephant', 'elephant', 'fish'],\n        ...                       'word':['a', 'c', 'c', 'a', 'b', None]})\n        >>> sf.unstack(column_names=['word', 'count'], new_column_name='words')\n        +----------+------------------+\n        |  topic   |      words       |\n        +----------+------------------+\n        | elephant | {'a': 1, 'b': 2} |\n        |   dog    |     {'c': 1}     |\n        |   cat    | {'a': 4, 'c': 2} |\n        |   fish   |       None       |\n        +----------+------------------+\n        [4 rows x 2 columns]\n\n        >>> sf = turicreate.SFrame({'friend': [2, 3, 4, 5, 6, 4, 5, 2, 3],\n        ...                      'user': [1, 1, 1, 2, 2, 2, 3, 4, 4]})\n        >>> sf.unstack('friend', new_column_name='new name')\n        +------+-----------+\n        | user |  new name |\n        +------+-----------+\n        |  3   |    [5]    |\n        |  1   | [2, 3, 4] |\n        |  2   | [6, 4, 5] |\n        |  4   |   [2, 3]  |\n        +------+-----------+\n        [4 rows x 2 columns]\n        "
        if type(column_names) != str and len(column_names) != 2:
            raise TypeError("'column_names' parameter has to be either a string or a list of two strings.")
        with cython_context():
            if type(column_names) == str:
                key_columns = [i for i in self.column_names() if i != column_names]
                if new_column_name is not None:
                    return self.groupby(key_columns, {new_column_name: aggregate.CONCAT(column_names)})
                else:
                    return self.groupby(key_columns, aggregate.CONCAT(column_names))
            elif len(column_names) == 2:
                key_columns = [i for i in self.column_names() if i not in column_names]
                if new_column_name is not None:
                    return self.groupby(key_columns, {new_column_name: aggregate.CONCAT(column_names[0], column_names[1])})
                else:
                    return self.groupby(key_columns, aggregate.CONCAT(column_names[0], column_names[1]))

    def unique(self):
        if False:
            print('Hello World!')
        "\n        Remove duplicate rows of the SFrame. Will not necessarily preserve the\n        order of the given SFrame in the new SFrame.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that contains the unique rows of the current SFrame.\n\n        Raises\n        ------\n        TypeError\n          If any column in the SFrame is a dictionary type.\n\n        See Also\n        --------\n        SArray.unique\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id':[1,2,3,3,4], 'value':[1,2,3,3,4]})\n        >>> sf\n        +----+-------+\n        | id | value |\n        +----+-------+\n        | 1  |   1   |\n        | 2  |   2   |\n        | 3  |   3   |\n        | 3  |   3   |\n        | 4  |   4   |\n        +----+-------+\n        [5 rows x 2 columns]\n\n        >>> sf.unique()\n        +----+-------+\n        | id | value |\n        +----+-------+\n        | 2  |   2   |\n        | 4  |   4   |\n        | 3  |   3   |\n        | 1  |   1   |\n        +----+-------+\n        [4 rows x 2 columns]\n        "
        return self.groupby(self.column_names(), {})

    def sort(self, key_column_names, ascending=True):
        if False:
            while True:
                i = 10
        "\n        Sort current SFrame by the given columns, using the given sort order.\n        Only columns that are type of str, int and float can be sorted.\n\n        Parameters\n        ----------\n        key_column_names : str | list of str | list of (str, bool) pairs\n            Names of columns to be sorted.  The result will be sorted first by\n            first column, followed by second column, and so on. All columns will\n            be sorted in the same order as governed by the `ascending`\n            parameter. To control the sort ordering for each column\n            individually, `key_column_names` must be a list of (str, bool) pairs.\n            Given this case, the first value is the column name and the second\n            value is a boolean indicating whether the sort order is ascending.\n\n        ascending : bool, optional\n            Sort all columns in the given order.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame that is sorted according to given sort criteria\n\n        See Also\n        --------\n        topk\n\n        Examples\n        --------\n        Suppose 'sf' is an sframe that has three columns 'a', 'b', 'c'.\n        To sort by column 'a', ascending\n\n        >>> sf = turicreate.SFrame({'a':[1,3,2,1],\n        ...                       'b':['a','c','b','b'],\n        ...                       'c':['x','y','z','y']})\n        >>> sf\n        +---+---+---+\n        | a | b | c |\n        +---+---+---+\n        | 1 | a | x |\n        | 3 | c | y |\n        | 2 | b | z |\n        | 1 | b | y |\n        +---+---+---+\n        [4 rows x 3 columns]\n\n        >>> sf.sort('a')\n        +---+---+---+\n        | a | b | c |\n        +---+---+---+\n        | 1 | a | x |\n        | 1 | b | y |\n        | 2 | b | z |\n        | 3 | c | y |\n        +---+---+---+\n        [4 rows x 3 columns]\n\n        To sort by column 'a', descending\n\n        >>> sf.sort('a', ascending = False)\n        +---+---+---+\n        | a | b | c |\n        +---+---+---+\n        | 3 | c | y |\n        | 2 | b | z |\n        | 1 | a | x |\n        | 1 | b | y |\n        +---+---+---+\n        [4 rows x 3 columns]\n\n        To sort by column 'a' and 'b', all ascending\n\n        >>> sf.sort(['a', 'b'])\n        +---+---+---+\n        | a | b | c |\n        +---+---+---+\n        | 1 | a | x |\n        | 1 | b | y |\n        | 2 | b | z |\n        | 3 | c | y |\n        +---+---+---+\n        [4 rows x 3 columns]\n\n        To sort by column 'a' ascending, and then by column 'c' descending\n\n        >>> sf.sort([('a', True), ('c', False)])\n        +---+---+---+\n        | a | b | c |\n        +---+---+---+\n        | 1 | b | y |\n        | 1 | a | x |\n        | 2 | b | z |\n        | 3 | c | y |\n        +---+---+---+\n        [4 rows x 3 columns]\n        "
        sort_column_names = []
        sort_column_orders = []
        if type(key_column_names) == str:
            sort_column_names = [key_column_names]
        elif type(key_column_names) == list:
            if len(key_column_names) == 0:
                raise ValueError('Please provide at least one column to sort')
            first_param_types = set([type(i) for i in key_column_names])
            if len(first_param_types) != 1:
                raise ValueError('key_column_names element are not of the same type')
            first_param_type = first_param_types.pop()
            if first_param_type == tuple:
                sort_column_names = [i[0] for i in key_column_names]
                sort_column_orders = [i[1] for i in key_column_names]
            elif first_param_type == str:
                sort_column_names = key_column_names
            else:
                raise TypeError('key_column_names type is not supported')
        else:
            raise TypeError('key_column_names type is not correct. Supported types are str, list of str or list of (str,bool) pair.')
        if len(sort_column_orders) == 0:
            sort_column_orders = [ascending for i in sort_column_names]
        my_column_names = set(self.column_names())
        for column in sort_column_names:
            if type(column) != str:
                raise TypeError('Only string parameter can be passed in as column names')
            if column not in my_column_names:
                raise ValueError("SFrame has no column named: '" + str(column) + "'")
            if self[column].dtype not in (str, int, float, datetime.datetime):
                raise TypeError('Only columns of type (str, int, float) can be sorted')
        with cython_context():
            return SFrame(_proxy=self.__proxy__.sort(sort_column_names, sort_column_orders))

    def dropna(self, columns=None, how='any', recursive=False):
        if False:
            while True:
                i = 10
        '\n        Remove missing values from an SFrame. A missing value is either ``None``\n        or ``NaN``.  If ``how`` is \'any\', a row will be removed if any of the\n        columns in the ``columns`` parameter contains at least one missing\n        value.  If ``how`` is \'all\', a row will be removed if all of the columns\n        in the ``columns`` parameter are missing values.\n\n        If the ``columns`` parameter is not specified, the default is to\n        consider all columns when searching for missing values.\n\n        Parameters\n        ----------\n        columns : list or str, optional\n            The columns to use when looking for missing values. By default, all\n            columns are used.\n\n        how : {\'any\', \'all\'}, optional\n            Specifies whether a row should be dropped if at least one column\n            has missing values, or if all columns have missing values.  \'any\' is\n            default.\n\n        recursive: bool\n            By default is False. If this flag is set to True, then `nan` check will\n            be performed on each element of a sframe cell in a DFS manner if the cell\n            has a nested structure, such as dict, list.\n\n        Returns\n        -------\n        out : SFrame\n            SFrame with missing values removed (according to the given rules).\n\n        See Also\n        --------\n        dropna_split :  Drops missing rows from the SFrame and returns them.\n\n        Examples\n        --------\n        Drop all missing values.\n\n        >>> sf = turicreate.SFrame({\'a\': [1, None, None], \'b\': [\'a\', \'b\', None]})\n        >>> sf.dropna()\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | a |\n        +---+---+\n        [1 rows x 2 columns]\n\n        Drop rows where every value is missing.\n\n        >>> sf.dropna(any="all")\n        +------+---+\n        |  a   | b |\n        +------+---+\n        |  1   | a |\n        | None | b |\n        +------+---+\n        [2 rows x 2 columns]\n\n        Drop rows where column \'a\' has a missing value.\n\n        >>> sf.dropna(\'a\', any="all")\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | a |\n        +---+---+\n        [1 rows x 2 columns]\n        '
        if type(columns) is list and len(columns) == 0:
            return SFrame(_proxy=self.__proxy__)
        (columns, all_behavior) = self.__dropna_errchk(columns, how)
        with cython_context():
            return SFrame(_proxy=self.__proxy__.drop_missing_values(columns, all_behavior, False, recursive))

    def dropna_split(self, columns=None, how='any', recursive=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Split rows with missing values from this SFrame. This function has the\n        same functionality as :py:func:`~turicreate.SFrame.dropna`, but returns a\n        tuple of two SFrames.  The first item is the expected output from\n        :py:func:`~turicreate.SFrame.dropna`, and the second item contains all the\n        rows filtered out by the `dropna` algorithm.\n\n        Parameters\n        ----------\n        columns : list or str, optional\n            The columns to use when looking for missing values. By default, all\n            columns are used.\n\n        how : {'any', 'all'}, optional\n            Specifies whether a row should be dropped if at least one column\n            has missing values, or if all columns have missing values.  'any' is\n            default.\n\n        recursive: bool\n            By default is False. If this flag is set to True, then `nan` check will\n            be performed on each element of a sframe cell in a recursive manner if the cell\n            has a nested structure, such as dict, list.\n\n\n        Returns\n        -------\n        out : (SFrame, SFrame)\n            (SFrame with missing values removed,\n             SFrame with the removed missing values)\n\n        See Also\n        --------\n        dropna\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'a': [1, None, None], 'b': ['a', 'b', None]})\n        >>> good, bad = sf.dropna_split()\n        >>> good\n        +---+---+\n        | a | b |\n        +---+---+\n        | 1 | a |\n        +---+---+\n        [1 rows x 2 columns]\n\n        >>> bad\n        +------+------+\n        |  a   |  b   |\n        +------+------+\n        | None |  b   |\n        | None | None |\n        +------+------+\n        [2 rows x 2 columns]\n        "
        if type(columns) is list and len(columns) == 0:
            return (SFrame(_proxy=self.__proxy__), SFrame())
        (columns, all_behavior) = self.__dropna_errchk(columns, how)
        sframe_tuple = self.__proxy__.drop_missing_values(columns, all_behavior, True, recursive)
        if len(sframe_tuple) != 2:
            raise RuntimeError('Did not return two SFrames!')
        with cython_context():
            return (SFrame(_proxy=sframe_tuple[0]), SFrame(_proxy=sframe_tuple[1]))

    def __dropna_errchk(self, columns, how):
        if False:
            return 10
        if columns is None:
            columns = list()
        elif type(columns) is str:
            columns = [columns]
        elif type(columns) is not list:
            raise TypeError("Must give columns as a list, str, or 'None'")
        else:
            list_types = set([type(i) for i in columns])
            if str not in list_types or len(list_types) > 1:
                raise TypeError("All columns must be of 'str' type")
        if how not in ['any', 'all']:
            raise ValueError("Must specify 'any' or 'all'")
        if how == 'all':
            all_behavior = True
        else:
            all_behavior = False
        return (columns, all_behavior)

    def fillna(self, column_name, value):
        if False:
            i = 10
            return i + 15
        "\n        Fill all missing values with a given value in a given column. If the\n        ``value`` is not the same type as the values in ``column_name``, this method\n        attempts to convert the value to the original column's type. If this\n        fails, an error is raised.\n\n        Parameters\n        ----------\n        column_name : str\n            The name of the column to modify.\n\n        value : type convertible to SArray's type\n            The value used to replace all missing values.\n\n        Returns\n        -------\n        out : SFrame\n            A new SFrame with the specified value in place of missing values.\n\n        See Also\n        --------\n        dropna\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'a':[1, None, None],\n        ...                       'b':['13.1', '17.2', None]})\n        >>> sf = sf.fillna('a', 0)\n        >>> sf\n        +---+------+\n        | a |  b   |\n        +---+------+\n        | 1 | 13.1 |\n        | 0 | 17.2 |\n        | 0 | None |\n        +---+------+\n        [3 rows x 2 columns]\n        "
        if type(column_name) is not str:
            raise TypeError('column_name must be a str')
        ret = self[self.column_names()]
        ret[column_name] = ret[column_name].fillna(value)
        return ret

    def add_row_number(self, column_name='id', start=0, inplace=False):
        if False:
            print('Hello World!')
        "\n        Returns an SFrame with a new column that numbers each row\n        sequentially. By default the count starts at 0, but this can be changed\n        to a positive or negative number.  The new column will be named with\n        the given column name.  An error will be raised if the given column\n        name already exists in the SFrame.\n\n        If inplace == False (default) this operation does not modify the\n        current SFrame, returning a new SFrame.\n\n        If inplace == True, this operation modifies the current\n        SFrame, returning self.\n\n        Parameters\n        ----------\n        column_name : str, optional\n            The name of the new column that will hold the row numbers.\n\n        start : int, optional\n            The number used to start the row number count.\n\n\n        inplace : bool, optional. Defaults to False.\n            Whether the SFrame is modified in place.\n\n        Returns\n        -------\n        out : SFrame\n            The new SFrame with a column name\n\n        Notes\n        -----\n        The range of numbers is constrained by a signed 64-bit integer, so\n        beware of overflow if you think the results in the row number column\n        will be greater than 9 quintillion.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'a': [1, None, None], 'b': ['a', 'b', None]})\n        >>> sf.add_row_number()\n        +----+------+------+\n        | id |  a   |  b   |\n        +----+------+------+\n        | 0  |  1   |  a   |\n        | 1  | None |  b   |\n        | 2  | None | None |\n        +----+------+------+\n        [3 rows x 3 columns]\n        "
        if type(column_name) is not str:
            raise TypeError('Must give column_name as strs')
        if type(start) is not int:
            raise TypeError('Must give start as int')
        if column_name in self.column_names():
            raise RuntimeError("Column '" + column_name + "' already exists in the current SFrame")
        the_col = _create_sequential_sarray(self.num_rows(), start)
        new_sf = SFrame()
        new_sf.add_column(the_col, column_name, inplace=True)
        new_sf.add_columns(self, inplace=True)
        if inplace:
            self.__proxy__ = new_sf.__proxy__
            return self
        else:
            return new_sf

    @property
    def shape(self):
        if False:
            print('Hello World!')
        "\n        The shape of the SFrame, in a tuple. The first entry is the number of\n        rows, the second is the number of columns.\n\n        Examples\n        --------\n        >>> sf = turicreate.SFrame({'id':[1,2,3], 'val':['A','B','C']})\n        >>> sf.shape\n        (3, 2)\n        "
        return (self.num_rows(), self.num_columns())

    @property
    def __proxy__(self):
        if False:
            i = 10
            return i + 15
        return self._proxy

    @__proxy__.setter
    def __proxy__(self, value):
        if False:
            i = 10
            return i + 15
        assert type(value) is UnitySFrameProxy
        self._cache = None
        self._proxy = value
        self._cache = None