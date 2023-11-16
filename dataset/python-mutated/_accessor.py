from distutils.util import strtobool
from math import isnan
import numpy
import pandas
from ..core.data import deconstruct_numpy, deconstruct_pandas, make_null_mask
from ..core.data.pd import _parse_datetime_index
from ..core.exception import PerspectiveError
from ._date_validator import _PerspectiveDateValidator
from .libpsppy import t_dtype

def _flatten_structure(array):
    if False:
        i = 10
        return i + 15
    'Flatten numpy.recarray or structured arrays into a dict.'
    columns = [numpy.copy(array[col]) for col in array.dtype.names]
    return dict(zip(array.dtype.names, columns))

def _type_to_format(data_or_schema):
    if False:
        for i in range(10):
            print('nop')
    'Deconstructs data passed in by the user into a standard format:\n\n    - A :obj:`list` of dicts, each of which represents a single row.\n    - A dict of :obj:`list`s, each of which represents a single column.\n\n    Schemas passed in by the user are preserved as-is.\n    :class:`pandas.DataFrame`s are flattened and returned as a columnar\n    dataset.  Finally, an integer is assigned to represent the type of the\n    dataset to the internal engine.\n\n    Returns:\n        :obj:`int`: type\n                - 0: records (:obj:`list` of :obj:`dict`)\n                - 1: columns (:obj:`dict` of :obj:`str` to :obj:`list`)\n                - 2: schema (dist[str]/dict[type])\n        :obj:`list`: column names\n        ():obj:`list`/:obj:`dict`): processed data\n    '
    if isinstance(data_or_schema, list):
        names = list(data_or_schema[0].keys()) if len(data_or_schema) > 0 else []
        return (False, 0, names, data_or_schema)
    elif isinstance(data_or_schema, dict):
        for v in data_or_schema.values():
            if isinstance(v, type) or isinstance(v, str):
                return (False, 2, list(data_or_schema.keys()), data_or_schema)
            elif isinstance(v, list):
                return (False, 1, list(data_or_schema.keys()), data_or_schema)
            else:
                try:
                    iter(v)
                except TypeError:
                    raise NotImplementedError('Cannot load dataset of non-iterable type: Data passed in through a dict must be of type `list` or `numpy.ndarray`.')
                else:
                    return (isinstance(v, numpy.ndarray), 1, list(data_or_schema.keys()), data_or_schema)
    elif isinstance(data_or_schema, numpy.ndarray):
        if not isinstance(data_or_schema.dtype.names, tuple):
            raise NotImplementedError('Data should be dict of numpy.ndarray or a structured array.')
        flattened = _flatten_structure(data_or_schema)
        return (True, 1, list(flattened.keys()), flattened)
    elif not (isinstance(data_or_schema, pandas.DataFrame) or isinstance(data_or_schema, pandas.Series)):
        raise NotImplementedError('Invalid data format `{}` - Data must be dataframe, dict, list, numpy.recarray, or a numpy structured array.'.format(type(data_or_schema)))
    else:
        (df, _) = deconstruct_pandas(data_or_schema)
        return (True, 1, df.columns.tolist(), {c: df[c].values for c in df.columns})

class _PerspectiveAccessor(object):
    """A uniform accessor that wraps data/schemas of varying formats with a
    common :func:`marshal` function.
    """
    INTEGER_TYPES = (int, numpy.integer)

    def __init__(self, data_or_schema):
        if False:
            for i in range(10):
                print('nop')
        (self._is_numpy, self._format, self._names, self._data_or_schema) = _type_to_format(data_or_schema)
        self._date_validator = _PerspectiveDateValidator()
        self._row_count = len(self._data_or_schema) if self._format == 0 else len(max(self._data_or_schema.values(), key=len)) if self._format == 1 else 0
        self._types = []
        for name in self._names:
            if not isinstance(name, str):
                raise PerspectiveError('Column names should be strings, not type `{0}`'.format(type(name).__name__))
            if self._is_numpy:
                array = self._data_or_schema[name]
                if not isinstance(array, numpy.ndarray):
                    raise PerspectiveError('Mixed datasets of numpy.ndarray and lists are not supported.')
                dtype = array.dtype
                if name == 'index' and hasattr(data_or_schema, 'index') and isinstance(data_or_schema.index, pandas.DatetimeIndex):
                    dtype = _parse_datetime_index(data_or_schema.index)
                self._types.append(str(dtype))
        self._numpy_column_masks = {}

    def data(self):
        if False:
            while True:
                i = 10
        return self._data_or_schema

    def format(self):
        if False:
            i = 10
            return i + 15
        return self._format

    def names(self):
        if False:
            for i in range(10):
                print('nop')
        return self._names

    def types(self):
        if False:
            i = 10
            return i + 15
        return self._types

    def date_validator(self):
        if False:
            for i in range(10):
                print('nop')
        return self._date_validator

    def row_count(self):
        if False:
            return 10
        return self._row_count

    def get(self, column_name, ridx):
        if False:
            i = 10
            return i + 15
        'Get the element at the specified column name and row index.\n\n        If the element does not exist, return None.\n\n        Args:\n            column_name (str)\n            ridx (int)\n\n        Returns:\n            object or None\n        '
        val = None
        try:
            if self._format == 0:
                return self._data_or_schema[ridx][column_name]
            elif self._format == 1:
                return self._data_or_schema[column_name][ridx]
            else:
                raise NotImplementedError()
            return val
        except (KeyError, IndexError):
            return None

    def marshal(self, cidx, ridx, dtype):
        if False:
            i = 10
            return i + 15
        "Returns the element at the specified column and row index, and\n        marshals it into an object compatible with the core engine's\n        :func:`fill` method.\n\n        If DTYPE_DATE or DTYPE_TIME is specified for a string value, attempt\n        to parse the string value or return :obj:`None`.\n\n        Args:\n            cidx (:obj:`int`)\n            ridx (:obj:`int`)\n            dtype (:obj:`.libpsppy.t_dtype`)\n\n        Returns:\n            object or None\n        "
        column_name = self._names[cidx]
        val = self.get(column_name, ridx)
        if val is None:
            return val
        if hasattr(val, '_psp_repr_'):
            val = val._psp_repr_()
        if isinstance(val, float) and isnan(val):
            return None
        elif isinstance(val, list) and len(val) == 1:
            val = val[0]
        elif dtype == t_dtype.DTYPE_STR:
            if isinstance(val, (bytes, bytearray)):
                return val.decode('utf-8')
            else:
                return str(val)
        elif dtype == t_dtype.DTYPE_DATE:
            if isinstance(val, str):
                parsed = self._date_validator.parse(val)
                return self._date_validator.to_date_components(parsed)
            else:
                return self._date_validator.to_date_components(val)
        elif dtype == t_dtype.DTYPE_TIME:
            if isinstance(val, str):
                parsed = self._date_validator.parse(val)
                return self._date_validator.to_timestamp(parsed)
            else:
                return self._date_validator.to_timestamp(val)
        elif dtype == t_dtype.DTYPE_BOOL:
            return bool(strtobool(str(val)))
        elif dtype == t_dtype.DTYPE_INT32 or dtype == t_dtype.DTYPE_INT64:
            if not isinstance(val, bool) and isinstance(val, (float, numpy.floating)):
                return int(val)
        elif dtype == t_dtype.DTYPE_FLOAT32 or dtype == t_dtype.DTYPE_FLOAT64:
            if not isinstance(val, bool) and isinstance(val, _PerspectiveAccessor.INTEGER_TYPES):
                return float(val)
        return val

    def try_cast_numpy_arrays(self):
        if False:
            i = 10
            return i + 15
        "When a numpy dataset is used to update, and when self._types\n        contains t_dtype objects from Perspective's already-initialized table,\n        use perspective dtypes and numpy dtypes to cast trivially comparable\n        dtypes to avoid iterative fills in C++.\n        "
        for i in range(len(self._names)):
            name = self._names[i]
            if name == '__INDEX__':
                continue
            array = self._data_or_schema.get(name, None)
            if array is None:
                continue
            type = self._types[i]
            if array.dtype == numpy.float64 and type == t_dtype.DTYPE_INT64:
                mask = make_null_mask(array)
                self._numpy_column_masks[name] = mask
                self._data_or_schema[name] = numpy.int64(array)
            elif array.dtype == numpy.int64 and type == t_dtype.DTYPE_FLOAT64:
                self._data_or_schema[name] = array.astype(numpy.float64)
            elif array.dtype == numpy.float32 and type == t_dtype.DTYPE_FLOAT64:
                mask = make_null_mask(array)
                self._numpy_column_masks[name] = mask
                self._data_or_schema[name] = numpy.float64(array)
            elif array.dtype == numpy.float32 and type == t_dtype.DTYPE_FLOAT32:
                mask = make_null_mask(array)
                self._numpy_column_masks[name] = mask
                self._data_or_schema[name] = numpy.float32(array)

    def _get_numpy_column(self, name):
        if False:
            i = 10
            return i + 15
        "For columnar datasets, return the :obj:`list`/Numpy array that\n        contains the data for a single column.\n\n        Args:\n            name (:obj:`str`): the column name to look up\n\n        Returns:\n            (:obj:`list`/numpy.array/None): returns the column's data, or None\n                if it cannot be found.\n        "
        data = self._data_or_schema.get(name, None)
        if data is None:
            raise PerspectiveError('Column `{0}` does not exist.'.format(name))
        mask = self._numpy_column_masks.get(name, None)
        return deconstruct_numpy(data, mask)

    def _has_column(self, ridx, name):
        if False:
            i = 10
            return i + 15
        'Given a row index and a column name, validate that the column exists\n        in the row.\n\n        This allows differentiation between value is None (unset) and value not\n        in row (no-op), which is important to prevent unintentional overwriting\n        of values during a partial update.\n\n        Args:\n            ridx (:obj:`int`)\n            name (:obj:`str`)\n\n        Returns:\n            bool: True if column is in row, or if column belongs to pkey/op\n                columns required by the engine. False otherwise.\n        '
        if self._format == 2 or name in ('psp_pkey', 'psp_okey', 'psp_op'):
            return True
        elif self._format == 1:
            return name in self._data_or_schema
        else:
            return name in self._data_or_schema[ridx]