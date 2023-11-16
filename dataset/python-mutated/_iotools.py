"""A collection of functions designed to help I/O with ascii files.

"""
__docformat__ = 'restructuredtext en'
import numpy as np
import numpy._core.numeric as nx
from numpy._utils import asbytes, asunicode

def _decode_line(line, encoding=None):
    if False:
        print('Hello World!')
    "Decode bytes from binary input streams.\n\n    Defaults to decoding from 'latin1'. That differs from the behavior of\n    np.compat.asunicode that decodes from 'ascii'.\n\n    Parameters\n    ----------\n    line : str or bytes\n         Line to be decoded.\n    encoding : str\n         Encoding used to decode `line`.\n\n    Returns\n    -------\n    decoded_line : str\n\n    "
    if type(line) is bytes:
        if encoding is None:
            encoding = 'latin1'
        line = line.decode(encoding)
    return line

def _is_string_like(obj):
    if False:
        i = 10
        return i + 15
    '\n    Check whether obj behaves like a string.\n    '
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def _is_bytes_like(obj):
    if False:
        i = 10
        return i + 15
    '\n    Check whether obj behaves like a bytes object.\n    '
    try:
        obj + b''
    except (TypeError, ValueError):
        return False
    return True

def has_nested_fields(ndtype):
    if False:
        return 10
    "\n    Returns whether one or several fields of a dtype are nested.\n\n    Parameters\n    ----------\n    ndtype : dtype\n        Data-type of a structured array.\n\n    Raises\n    ------\n    AttributeError\n        If `ndtype` does not have a `names` attribute.\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float)])\n    >>> np.lib._iotools.has_nested_fields(dt)\n    False\n\n    "
    for name in ndtype.names or ():
        if ndtype[name].names is not None:
            return True
    return False

def flatten_dtype(ndtype, flatten_base=False):
    if False:
        print('Hello World!')
    "\n    Unpack a structured data-type by collapsing nested fields and/or fields\n    with a shape.\n\n    Note that the field names are lost.\n\n    Parameters\n    ----------\n    ndtype : dtype\n        The datatype to collapse\n    flatten_base : bool, optional\n       If True, transform a field with a shape into several fields. Default is\n       False.\n\n    Examples\n    --------\n    >>> dt = np.dtype([('name', 'S4'), ('x', float), ('y', float),\n    ...                ('block', int, (2, 3))])\n    >>> np.lib._iotools.flatten_dtype(dt)\n    [dtype('S4'), dtype('float64'), dtype('float64'), dtype('int64')]\n    >>> np.lib._iotools.flatten_dtype(dt, flatten_base=True)\n    [dtype('S4'),\n     dtype('float64'),\n     dtype('float64'),\n     dtype('int64'),\n     dtype('int64'),\n     dtype('int64'),\n     dtype('int64'),\n     dtype('int64'),\n     dtype('int64')]\n\n    "
    names = ndtype.names
    if names is None:
        if flatten_base:
            return [ndtype.base] * int(np.prod(ndtype.shape))
        return [ndtype.base]
    else:
        types = []
        for field in names:
            info = ndtype.fields[field]
            flat_dt = flatten_dtype(info[0], flatten_base)
            types.extend(flat_dt)
        return types

class LineSplitter:
    """
    Object to split a string at a given delimiter or at given places.

    Parameters
    ----------
    delimiter : str, int, or sequence of ints, optional
        If a string, character used to delimit consecutive fields.
        If an integer or a sequence of integers, width(s) of each field.
    comments : str, optional
        Character used to mark the beginning of a comment. Default is '#'.
    autostrip : bool, optional
        Whether to strip each individual field. Default is True.

    """

    def autostrip(self, method):
        if False:
            return 10
        '\n        Wrapper to strip each member of the output of `method`.\n\n        Parameters\n        ----------\n        method : function\n            Function that takes a single argument and returns a sequence of\n            strings.\n\n        Returns\n        -------\n        wrapped : function\n            The result of wrapping `method`. `wrapped` takes a single input\n            argument and returns a list of strings that are stripped of\n            white-space.\n\n        '
        return lambda input: [_.strip() for _ in method(input)]

    def __init__(self, delimiter=None, comments='#', autostrip=True, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        delimiter = _decode_line(delimiter)
        comments = _decode_line(comments)
        self.comments = comments
        if delimiter is None or isinstance(delimiter, str):
            delimiter = delimiter or None
            _handyman = self._delimited_splitter
        elif hasattr(delimiter, '__iter__'):
            _handyman = self._variablewidth_splitter
            idx = np.cumsum([0] + list(delimiter))
            delimiter = [slice(i, j) for (i, j) in zip(idx[:-1], idx[1:])]
        elif int(delimiter):
            (_handyman, delimiter) = (self._fixedwidth_splitter, int(delimiter))
        else:
            (_handyman, delimiter) = (self._delimited_splitter, None)
        self.delimiter = delimiter
        if autostrip:
            self._handyman = self.autostrip(_handyman)
        else:
            self._handyman = _handyman
        self.encoding = encoding

    def _delimited_splitter(self, line):
        if False:
            while True:
                i = 10
        'Chop off comments, strip, and split at delimiter. '
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip(' \r\n')
        if not line:
            return []
        return line.split(self.delimiter)

    def _fixedwidth_splitter(self, line):
        if False:
            return 10
        if self.comments is not None:
            line = line.split(self.comments)[0]
        line = line.strip('\r\n')
        if not line:
            return []
        fixed = self.delimiter
        slices = [slice(i, i + fixed) for i in range(0, len(line), fixed)]
        return [line[s] for s in slices]

    def _variablewidth_splitter(self, line):
        if False:
            i = 10
            return i + 15
        if self.comments is not None:
            line = line.split(self.comments)[0]
        if not line:
            return []
        slices = self.delimiter
        return [line[s] for s in slices]

    def __call__(self, line):
        if False:
            i = 10
            return i + 15
        return self._handyman(_decode_line(line, self.encoding))

class NameValidator:
    """
    Object to validate a list of strings to use as field names.

    The strings are stripped of any non alphanumeric character, and spaces
    are replaced by '_'. During instantiation, the user can define a list
    of names to exclude, as well as a list of invalid characters. Names in
    the exclusion list are appended a '_' character.

    Once an instance has been created, it can be called with a list of
    names, and a list of valid names will be created.  The `__call__`
    method accepts an optional keyword "default" that sets the default name
    in case of ambiguity. By default this is 'f', so that names will
    default to `f0`, `f1`, etc.

    Parameters
    ----------
    excludelist : sequence, optional
        A list of names to exclude. This list is appended to the default
        list ['return', 'file', 'print']. Excluded names are appended an
        underscore: for example, `file` becomes `file_` if supplied.
    deletechars : str, optional
        A string combining invalid characters that must be deleted from the
        names.
    case_sensitive : {True, False, 'upper', 'lower'}, optional
        * If True, field names are case-sensitive.
        * If False or 'upper', field names are converted to upper case.
        * If 'lower', field names are converted to lower case.

        The default value is True.
    replace_space : '_', optional
        Character(s) used in replacement of white spaces.

    Notes
    -----
    Calling an instance of `NameValidator` is the same as calling its
    method `validate`.

    Examples
    --------
    >>> validator = np.lib._iotools.NameValidator()
    >>> validator(['file', 'field2', 'with space', 'CaSe'])
    ('file_', 'field2', 'with_space', 'CaSe')

    >>> validator = np.lib._iotools.NameValidator(excludelist=['excl'],
    ...                                           deletechars='q',
    ...                                           case_sensitive=False)
    >>> validator(['excl', 'field2', 'no_q', 'with space', 'CaSe'])
    ('EXCL', 'FIELD2', 'NO_Q', 'WITH_SPACE', 'CASE')

    """
    defaultexcludelist = ['return', 'file', 'print']
    defaultdeletechars = set("~!@#$%^&*()-=+~\\|]}[{';: /?.>,<")

    def __init__(self, excludelist=None, deletechars=None, case_sensitive=None, replace_space='_'):
        if False:
            print('Hello World!')
        if excludelist is None:
            excludelist = []
        excludelist.extend(self.defaultexcludelist)
        self.excludelist = excludelist
        if deletechars is None:
            delete = self.defaultdeletechars
        else:
            delete = set(deletechars)
        delete.add('"')
        self.deletechars = delete
        if case_sensitive is None or case_sensitive is True:
            self.case_converter = lambda x: x
        elif case_sensitive is False or case_sensitive.startswith('u'):
            self.case_converter = lambda x: x.upper()
        elif case_sensitive.startswith('l'):
            self.case_converter = lambda x: x.lower()
        else:
            msg = 'unrecognized case_sensitive value %s.' % case_sensitive
            raise ValueError(msg)
        self.replace_space = replace_space

    def validate(self, names, defaultfmt='f%i', nbfields=None):
        if False:
            i = 10
            return i + 15
        '\n        Validate a list of strings as field names for a structured array.\n\n        Parameters\n        ----------\n        names : sequence of str\n            Strings to be validated.\n        defaultfmt : str, optional\n            Default format string, used if validating a given string\n            reduces its length to zero.\n        nbfields : integer, optional\n            Final number of validated names, used to expand or shrink the\n            initial list of names.\n\n        Returns\n        -------\n        validatednames : list of str\n            The list of validated field names.\n\n        Notes\n        -----\n        A `NameValidator` instance can be called directly, which is the\n        same as calling `validate`. For examples, see `NameValidator`.\n\n        '
        if names is None:
            if nbfields is None:
                return None
            names = []
        if isinstance(names, str):
            names = [names]
        if nbfields is not None:
            nbnames = len(names)
            if nbnames < nbfields:
                names = list(names) + [''] * (nbfields - nbnames)
            elif nbnames > nbfields:
                names = names[:nbfields]
        deletechars = self.deletechars
        excludelist = self.excludelist
        case_converter = self.case_converter
        replace_space = self.replace_space
        validatednames = []
        seen = dict()
        nbempty = 0
        for item in names:
            item = case_converter(item).strip()
            if replace_space:
                item = item.replace(' ', replace_space)
            item = ''.join([c for c in item if c not in deletechars])
            if item == '':
                item = defaultfmt % nbempty
                while item in names:
                    nbempty += 1
                    item = defaultfmt % nbempty
                nbempty += 1
            elif item in excludelist:
                item += '_'
            cnt = seen.get(item, 0)
            if cnt > 0:
                validatednames.append(item + '_%d' % cnt)
            else:
                validatednames.append(item)
            seen[item] = cnt + 1
        return tuple(validatednames)

    def __call__(self, names, defaultfmt='f%i', nbfields=None):
        if False:
            print('Hello World!')
        return self.validate(names, defaultfmt=defaultfmt, nbfields=nbfields)

def str2bool(value):
    if False:
        return 10
    "\n    Tries to transform a string supposed to represent a boolean to a boolean.\n\n    Parameters\n    ----------\n    value : str\n        The string that is transformed to a boolean.\n\n    Returns\n    -------\n    boolval : bool\n        The boolean representation of `value`.\n\n    Raises\n    ------\n    ValueError\n        If the string is not 'True' or 'False' (case independent)\n\n    Examples\n    --------\n    >>> np.lib._iotools.str2bool('TRUE')\n    True\n    >>> np.lib._iotools.str2bool('false')\n    False\n\n    "
    value = value.upper()
    if value == 'TRUE':
        return True
    elif value == 'FALSE':
        return False
    else:
        raise ValueError('Invalid boolean')

class ConverterError(Exception):
    """
    Exception raised when an error occurs in a converter for string values.

    """
    pass

class ConverterLockError(ConverterError):
    """
    Exception raised when an attempt is made to upgrade a locked converter.

    """
    pass

class ConversionWarning(UserWarning):
    """
    Warning issued when a string converter has a problem.

    Notes
    -----
    In `genfromtxt` a `ConversionWarning` is issued if raising exceptions
    is explicitly suppressed with the "invalid_raise" keyword.

    """
    pass

class StringConverter:
    """
    Factory class for function transforming a string into another object
    (int, float).

    After initialization, an instance can be called to transform a string
    into another object. If the string is recognized as representing a
    missing value, a default value is returned.

    Attributes
    ----------
    func : function
        Function used for the conversion.
    default : any
        Default value to return when the input corresponds to a missing
        value.
    type : type
        Type of the output.
    _status : int
        Integer representing the order of the conversion.
    _mapper : sequence of tuples
        Sequence of tuples (dtype, function, default value) to evaluate in
        order.
    _locked : bool
        Holds `locked` parameter.

    Parameters
    ----------
    dtype_or_func : {None, dtype, function}, optional
        If a `dtype`, specifies the input data type, used to define a basic
        function and a default value for missing data. For example, when
        `dtype` is float, the `func` attribute is set to `float` and the
        default value to `np.nan`.  If a function, this function is used to
        convert a string to another object. In this case, it is recommended
        to give an associated default value as input.
    default : any, optional
        Value to return by default, that is, when the string to be
        converted is flagged as missing. If not given, `StringConverter`
        tries to supply a reasonable default value.
    missing_values : {None, sequence of str}, optional
        ``None`` or sequence of strings indicating a missing value. If ``None``
        then missing values are indicated by empty entries. The default is
        ``None``.
    locked : bool, optional
        Whether the StringConverter should be locked to prevent automatic
        upgrade or not. Default is False.

    """
    _mapper = [(nx.bool_, str2bool, False), (nx.int_, int, -1)]
    if nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize:
        _mapper.append((nx.int64, int, -1))
    _mapper.extend([(nx.float64, float, nx.nan), (nx.complex128, complex, nx.nan + 0j), (nx.longdouble, nx.longdouble, nx.nan), (nx.integer, int, -1), (nx.floating, float, nx.nan), (nx.complexfloating, complex, nx.nan + 0j), (nx.str_, asunicode, '???'), (nx.bytes_, asbytes, '???')])

    @classmethod
    def _getdtype(cls, val):
        if False:
            while True:
                i = 10
        'Returns the dtype of the input variable.'
        return np.array(val).dtype

    @classmethod
    def _getsubdtype(cls, val):
        if False:
            i = 10
            return i + 15
        'Returns the type of the dtype of the input variable.'
        return np.array(val).dtype.type

    @classmethod
    def _dtypeortype(cls, dtype):
        if False:
            i = 10
            return i + 15
        'Returns dtype for datetime64 and type of dtype otherwise.'
        if dtype.type == np.datetime64:
            return dtype
        return dtype.type

    @classmethod
    def upgrade_mapper(cls, func, default=None):
        if False:
            print('Hello World!')
        '\n        Upgrade the mapper of a StringConverter by adding a new function and\n        its corresponding default.\n\n        The input function (or sequence of functions) and its associated\n        default value (if any) is inserted in penultimate position of the\n        mapper.  The corresponding type is estimated from the dtype of the\n        default value.\n\n        Parameters\n        ----------\n        func : var\n            Function, or sequence of functions\n\n        Examples\n        --------\n        >>> import dateutil.parser\n        >>> import datetime\n        >>> dateparser = dateutil.parser.parse\n        >>> defaultdate = datetime.date(2000, 1, 1)\n        >>> StringConverter.upgrade_mapper(dateparser, default=defaultdate)\n        '
        if hasattr(func, '__call__'):
            cls._mapper.insert(-1, (cls._getsubdtype(default), func, default))
            return
        elif hasattr(func, '__iter__'):
            if isinstance(func[0], (tuple, list)):
                for _ in func:
                    cls._mapper.insert(-1, _)
                return
            if default is None:
                default = [None] * len(func)
            else:
                default = list(default)
                default.append([None] * (len(func) - len(default)))
            for (fct, dft) in zip(func, default):
                cls._mapper.insert(-1, (cls._getsubdtype(dft), fct, dft))

    @classmethod
    def _find_map_entry(cls, dtype):
        if False:
            while True:
                i = 10
        for (i, (deftype, func, default_def)) in enumerate(cls._mapper):
            if dtype.type == deftype:
                return (i, (deftype, func, default_def))
        for (i, (deftype, func, default_def)) in enumerate(cls._mapper):
            if np.issubdtype(dtype.type, deftype):
                return (i, (deftype, func, default_def))
        raise LookupError

    def __init__(self, dtype_or_func=None, default=None, missing_values=None, locked=False):
        if False:
            i = 10
            return i + 15
        self._locked = bool(locked)
        if dtype_or_func is None:
            self.func = str2bool
            self._status = 0
            self.default = default or False
            dtype = np.dtype('bool')
        else:
            try:
                self.func = None
                dtype = np.dtype(dtype_or_func)
            except TypeError:
                if not hasattr(dtype_or_func, '__call__'):
                    errmsg = "The input argument `dtype` is neither a function nor a dtype (got '%s' instead)"
                    raise TypeError(errmsg % type(dtype_or_func))
                self.func = dtype_or_func
                if default is None:
                    try:
                        default = self.func('0')
                    except ValueError:
                        default = None
                dtype = self._getdtype(default)
            try:
                (self._status, (_, func, default_def)) = self._find_map_entry(dtype)
            except LookupError:
                self.default = default
                (_, func, _) = self._mapper[-1]
                self._status = 0
            else:
                if default is None:
                    self.default = default_def
                else:
                    self.default = default
            if self.func is None:
                self.func = func
            if self.func == self._mapper[1][1]:
                if issubclass(dtype.type, np.uint64):
                    self.func = np.uint64
                elif issubclass(dtype.type, np.int64):
                    self.func = np.int64
                else:
                    self.func = lambda x: int(float(x))
        if missing_values is None:
            self.missing_values = {''}
        else:
            if isinstance(missing_values, str):
                missing_values = missing_values.split(',')
            self.missing_values = set(list(missing_values) + [''])
        self._callingfunction = self._strict_call
        self.type = self._dtypeortype(dtype)
        self._checked = False
        self._initial_default = default

    def _loose_call(self, value):
        if False:
            i = 10
            return i + 15
        try:
            return self.func(value)
        except ValueError:
            return self.default

    def _strict_call(self, value):
        if False:
            while True:
                i = 10
        try:
            new_value = self.func(value)
            if self.func is int:
                try:
                    np.array(value, dtype=self.type)
                except OverflowError:
                    raise ValueError
            return new_value
        except ValueError:
            if value.strip() in self.missing_values:
                if not self._status:
                    self._checked = False
                return self.default
            raise ValueError("Cannot convert string '%s'" % value)

    def __call__(self, value):
        if False:
            i = 10
            return i + 15
        return self._callingfunction(value)

    def _do_upgrade(self):
        if False:
            return 10
        if self._locked:
            errmsg = 'Converter is locked and cannot be upgraded'
            raise ConverterLockError(errmsg)
        _statusmax = len(self._mapper)
        _status = self._status
        if _status == _statusmax:
            errmsg = 'Could not find a valid conversion function'
            raise ConverterError(errmsg)
        elif _status < _statusmax - 1:
            _status += 1
        (self.type, self.func, default) = self._mapper[_status]
        self._status = _status
        if self._initial_default is not None:
            self.default = self._initial_default
        else:
            self.default = default

    def upgrade(self, value):
        if False:
            while True:
                i = 10
        '\n        Find the best converter for a given string, and return the result.\n\n        The supplied string `value` is converted by testing different\n        converters in order. First the `func` method of the\n        `StringConverter` instance is tried, if this fails other available\n        converters are tried.  The order in which these other converters\n        are tried is determined by the `_status` attribute of the instance.\n\n        Parameters\n        ----------\n        value : str\n            The string to convert.\n\n        Returns\n        -------\n        out : any\n            The result of converting `value` with the appropriate converter.\n\n        '
        self._checked = True
        try:
            return self._strict_call(value)
        except ValueError:
            self._do_upgrade()
            return self.upgrade(value)

    def iterupgrade(self, value):
        if False:
            return 10
        self._checked = True
        if not hasattr(value, '__iter__'):
            value = (value,)
        _strict_call = self._strict_call
        try:
            for _m in value:
                _strict_call(_m)
        except ValueError:
            self._do_upgrade()
            self.iterupgrade(value)

    def update(self, func, default=None, testing_value=None, missing_values='', locked=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set StringConverter attributes directly.\n\n        Parameters\n        ----------\n        func : function\n            Conversion function.\n        default : any, optional\n            Value to return by default, that is, when the string to be\n            converted is flagged as missing. If not given,\n            `StringConverter` tries to supply a reasonable default value.\n        testing_value : str, optional\n            A string representing a standard input value of the converter.\n            This string is used to help defining a reasonable default\n            value.\n        missing_values : {sequence of str, None}, optional\n            Sequence of strings indicating a missing value. If ``None``, then\n            the existing `missing_values` are cleared. The default is `''`.\n        locked : bool, optional\n            Whether the StringConverter should be locked to prevent\n            automatic upgrade or not. Default is False.\n\n        Notes\n        -----\n        `update` takes the same parameters as the constructor of\n        `StringConverter`, except that `func` does not accept a `dtype`\n        whereas `dtype_or_func` in the constructor does.\n\n        "
        self.func = func
        self._locked = locked
        if default is not None:
            self.default = default
            self.type = self._dtypeortype(self._getdtype(default))
        else:
            try:
                tester = func(testing_value or '1')
            except (TypeError, ValueError):
                tester = None
            self.type = self._dtypeortype(self._getdtype(tester))
        if missing_values is None:
            self.missing_values = set()
        else:
            if not np.iterable(missing_values):
                missing_values = [missing_values]
            if not all((isinstance(v, str) for v in missing_values)):
                raise TypeError('missing_values must be strings or unicode')
            self.missing_values.update(missing_values)

def easy_dtype(ndtype, names=None, defaultfmt='f%i', **validationargs):
    if False:
        print('Hello World!')
    '\n    Convenience function to create a `np.dtype` object.\n\n    The function processes the input `dtype` and matches it with the given\n    names.\n\n    Parameters\n    ----------\n    ndtype : var\n        Definition of the dtype. Can be any string or dictionary recognized\n        by the `np.dtype` function, or a sequence of types.\n    names : str or sequence, optional\n        Sequence of strings to use as field names for a structured dtype.\n        For convenience, `names` can be a string of a comma-separated list\n        of names.\n    defaultfmt : str, optional\n        Format string used to define missing names, such as ``"f%i"``\n        (default) or ``"fields_%02i"``.\n    validationargs : optional\n        A series of optional arguments used to initialize a\n        `NameValidator`.\n\n    Examples\n    --------\n    >>> np.lib._iotools.easy_dtype(float)\n    dtype(\'float64\')\n    >>> np.lib._iotools.easy_dtype("i4, f8")\n    dtype([(\'f0\', \'<i4\'), (\'f1\', \'<f8\')])\n    >>> np.lib._iotools.easy_dtype("i4, f8", defaultfmt="field_%03i")\n    dtype([(\'field_000\', \'<i4\'), (\'field_001\', \'<f8\')])\n\n    >>> np.lib._iotools.easy_dtype((int, float, float), names="a,b,c")\n    dtype([(\'a\', \'<i8\'), (\'b\', \'<f8\'), (\'c\', \'<f8\')])\n    >>> np.lib._iotools.easy_dtype(float, names="a,b,c")\n    dtype([(\'a\', \'<f8\'), (\'b\', \'<f8\'), (\'c\', \'<f8\')])\n\n    '
    try:
        ndtype = np.dtype(ndtype)
    except TypeError:
        validate = NameValidator(**validationargs)
        nbfields = len(ndtype)
        if names is None:
            names = [''] * len(ndtype)
        elif isinstance(names, str):
            names = names.split(',')
        names = validate(names, nbfields=nbfields, defaultfmt=defaultfmt)
        ndtype = np.dtype(dict(formats=ndtype, names=names))
    else:
        if names is not None:
            validate = NameValidator(**validationargs)
            if isinstance(names, str):
                names = names.split(',')
            if ndtype.names is None:
                formats = tuple([ndtype.type] * len(names))
                names = validate(names, defaultfmt=defaultfmt)
                ndtype = np.dtype(list(zip(names, formats)))
            else:
                ndtype.names = validate(names, nbfields=len(ndtype.names), defaultfmt=defaultfmt)
        elif ndtype.names is not None:
            validate = NameValidator(**validationargs)
            numbered_names = tuple(('f%i' % i for i in range(len(ndtype.names))))
            if ndtype.names == numbered_names and defaultfmt != 'f%i':
                ndtype.names = validate([''] * len(ndtype.names), defaultfmt=defaultfmt)
            else:
                ndtype.names = validate(ndtype.names, defaultfmt=defaultfmt)
    return ndtype