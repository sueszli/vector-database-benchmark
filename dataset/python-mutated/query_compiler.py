"""
Module contains ``DFAlgQueryCompiler`` class.

``DFAlgQueryCompiler`` is used for lazy DataFrame Algebra based engine.
"""
from functools import wraps
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import is_bool_dtype, is_integer_dtype, is_list_like
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.core.storage_formats.base.query_compiler import _get_axis as default_axis_getter
from modin.core.storage_formats.base.query_compiler import _set_axis as default_axis_setter
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings

def is_inoperable(value):
    if False:
        return 10
    '\n    Check if value cannot be processed by HDK engine.\n\n    Parameters\n    ----------\n    value : any\n        A value to check.\n\n    Returns\n    -------\n    bool\n    '
    if isinstance(value, (tuple, list)):
        result = False
        for val in value:
            result = result or is_inoperable(val)
        return result
    elif isinstance(value, dict):
        return is_inoperable(list(value.values()))
    else:
        value = getattr(value, '_query_compiler', value)
        if hasattr(value, '_modin_frame'):
            return value._modin_frame._has_unsupported_data
    return False

def build_method_wrapper(name, method):
    if False:
        while True:
            i = 10
    "\n    Build method wrapper to handle inoperable data types.\n\n    Wrapper calls the original method if all its arguments can be processed\n    by HDK engine and fallback to parent's method otherwise.\n\n    Parameters\n    ----------\n    name : str\n        Parent's method name to fallback to.\n    method : callable\n        A method to wrap.\n\n    Returns\n    -------\n    callable\n    "

    @wraps(method)
    def method_wrapper(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        default_method = getattr(super(type(self), self), name, None)
        if is_inoperable([self, args, kwargs]):
            if default_method is None:
                raise NotImplementedError('Frame contains data of unsupported types.')
            return default_method(*args, **kwargs)
        try:
            return method(self, *args, **kwargs)
        except NotImplementedError as err:
            if default_method is None:
                raise err
            ErrorMessage.default_to_pandas(message=str(err))
            return default_method(*args, **kwargs)
    return method_wrapper

def bind_wrappers(cls):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrap class methods.\n\n    Decorator allows to fallback to the parent query compiler methods when unsupported\n    data types are used in a frame.\n\n    Returns\n    -------\n    class\n    '
    exclude = set(['__init__', 'to_pandas', 'from_pandas', 'from_arrow', 'default_to_pandas', '_get_index', '_set_index', '_get_columns', '_set_columns'])
    for (name, method) in cls.__dict__.items():
        if name in exclude:
            continue
        if callable(method):
            setattr(cls, name, build_method_wrapper(name, method))
    return cls

@bind_wrappers
@_inherit_docstrings(BaseQueryCompiler)
class DFAlgQueryCompiler(BaseQueryCompiler):
    """
    Query compiler for the HDK storage format.

    This class doesn't perform much processing and mostly forwards calls to
    :py:class:`~modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe.HdkOnNativeDataframe`
    for lazy execution trees build.

    Parameters
    ----------
    frame : HdkOnNativeDataframe
        Modin Frame to query with the compiled queries.
    shape_hint : {"row", "column", None}, default: None
        Shape hint for frames known to be a column or a row, otherwise None.

    Attributes
    ----------
    _modin_frame : HdkOnNativeDataframe
        Modin Frame to query with the compiled queries.
    _shape_hint : {"row", "column", None}
        Shape hint for frames known to be a column or a row, otherwise None.
    """
    lazy_execution = True

    def __init__(self, frame, shape_hint=None):
        if False:
            while True:
                i = 10
        assert frame is not None
        self._modin_frame = frame
        if shape_hint is None and len(self._modin_frame.columns) == 1:
            shape_hint = 'column'
        self._shape_hint = shape_hint

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def execute(self):
        if False:
            return 10
        self._modin_frame._execute()

    def force_import(self):
        if False:
            i = 10
            return i + 15
        'Force table import.'
        self._modin_frame.force_import()

    def support_materialization_in_worker_process(self) -> bool:
        if False:
            print('Hello World!')
        return True

    def to_pandas(self):
        if False:
            i = 10
            return i + 15
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        if False:
            return 10
        if len(df.columns) == 1:
            shape_hint = 'column'
        elif len(df) == 1:
            shape_hint = 'row'
        else:
            shape_hint = None
        return cls(data_cls.from_pandas(df), shape_hint=shape_hint)

    @classmethod
    def from_arrow(cls, at, data_cls):
        if False:
            print('Hello World!')
        if len(at.columns) == 1:
            shape_hint = 'column'
        elif len(at) == 1:
            shape_hint = 'row'
        else:
            shape_hint = None
        return cls(data_cls.from_arrow(at), shape_hint=shape_hint)

    def to_dataframe(self, nan_as_null: bool=False, allow_copy: bool=True):
        if False:
            while True:
                i = 10
        return self._modin_frame.__dataframe__(nan_as_null=nan_as_null, allow_copy=allow_copy)

    @classmethod
    def from_dataframe(cls, df, data_cls):
        if False:
            i = 10
            return i + 15
        return cls(data_cls.from_dataframe(df))
    default_to_pandas = PandasQueryCompiler.default_to_pandas

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__constructor__(self._modin_frame.copy(), self._shape_hint)

    def getitem_column_array(self, key, numeric=False, ignore_order=False):
        if False:
            print('Hello World!')
        shape_hint = 'column' if len(key) == 1 else None
        if numeric:
            new_modin_frame = self._modin_frame.take_2d_labels_or_positional(col_positions=key)
        else:
            new_modin_frame = self._modin_frame.take_2d_labels_or_positional(col_labels=key)
        return self.__constructor__(new_modin_frame, shape_hint)

    def getitem_array(self, key):
        if False:
            return 10
        if isinstance(key, type(self)):
            new_modin_frame = self._modin_frame.filter(key._modin_frame)
            return self.__constructor__(new_modin_frame, self._shape_hint)
        if is_bool_indexer(key):
            return self.default_to_pandas(lambda df: df[key])
        if any((k not in self.columns for k in key)):
            raise KeyError('{} not index'.format(str([k for k in key if k not in self.columns]).replace(',', '')))
        return self.getitem_column_array(key)

    def merge(self, right, **kwargs):
        if False:
            return 10
        on = kwargs.get('on', None)
        left_on = kwargs.get('left_on', None)
        right_on = kwargs.get('right_on', None)
        left_index = kwargs.get('left_index', False)
        right_index = kwargs.get('right_index', False)
        "Only non-index joins with explicit 'on' are supported"
        if left_index is False and right_index is False:
            if left_on is None and right_on is None:
                if on is None:
                    on = [c for c in self.columns if c in right.columns]
                left_on = on
                right_on = on
            if not isinstance(left_on, list):
                left_on = [left_on]
            if not isinstance(right_on, list):
                right_on = [right_on]
            how = kwargs.get('how', 'inner')
            sort = kwargs.get('sort', False)
            suffixes = kwargs.get('suffixes', None)
            return self.__constructor__(self._modin_frame.join(right._modin_frame, how=how, left_on=left_on, right_on=right_on, sort=sort, suffixes=suffixes))
        else:
            return self.default_to_pandas(pandas.DataFrame.merge, right, **kwargs)

    def take_2d_positional(self, index=None, columns=None):
        if False:
            return 10
        return self.__constructor__(self._modin_frame.take_2d_labels_or_positional(row_positions=index, col_positions=columns))

    def groupby_size(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
        if False:
            return 10
        if len(self.columns) == 0:
            raise NotImplementedError('Grouping on empty frame or on index level is not yet implemented.')
        groupby_kwargs = groupby_kwargs.copy()
        as_index = groupby_kwargs.get('as_index', True)
        groupby_kwargs['as_index'] = True
        new_frame = self._modin_frame.groupby_agg(by, axis, {self._modin_frame.columns[0]: 'size'}, groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)
        if as_index:
            shape_hint = 'column'
            new_frame = new_frame._set_columns([MODIN_UNNAMED_SERIES_LABEL])
        else:
            shape_hint = None
            new_frame = new_frame._set_columns(['size']).reset_index(drop=False)
        return self.__constructor__(new_frame, shape_hint=shape_hint)

    def groupby_sum(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
        if False:
            while True:
                i = 10
        new_frame = self._modin_frame.groupby_agg(by, axis, 'sum', groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)
        return self.__constructor__(new_frame)

    def groupby_count(self, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False):
        if False:
            i = 10
            return i + 15
        new_frame = self._modin_frame.groupby_agg(by, axis, 'count', groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)
        return self.__constructor__(new_frame)

    def groupby_agg(self, by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, how='axis_wise', drop=False, series_groupby=False):
        if False:
            return 10
        if callable(agg_func):
            raise NotImplementedError('Python callable is not a valid aggregation function for HDK storage format.')
        if how != 'axis_wise':
            raise NotImplementedError(f"'{how}' type of groupby-aggregation functions is not supported for HDK storage format.")
        new_frame = self._modin_frame.groupby_agg(by, axis, agg_func, groupby_kwargs, agg_args=agg_args, agg_kwargs=agg_kwargs, drop=drop)
        return self.__constructor__(new_frame)

    def count(self, **kwargs):
        if False:
            while True:
                i = 10
        return self._agg('count', **kwargs)

    def max(self, **kwargs):
        if False:
            return 10
        return self._agg('max', **kwargs)

    def min(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._agg('min', **kwargs)

    def sum(self, **kwargs):
        if False:
            print('Hello World!')
        min_count = kwargs.pop('min_count', 0)
        if min_count != 0:
            raise NotImplementedError(f"HDK's sum does not support such set of parameters: min_count={min_count}.")
        _check_int_or_float('sum', self.dtypes)
        return self._agg('sum', **kwargs)

    def mean(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        _check_int_or_float('mean', self.dtypes)
        return self._agg('mean', **kwargs)

    def nunique(self, axis=0, dropna=True):
        if False:
            return 10
        if axis != 0 or not dropna:
            raise NotImplementedError(f"HDK's nunique does not support such set of parameters: axis={axis}, dropna={dropna}.")
        return self._agg('nunique')

    def _agg(self, agg, axis=0, level=None, **kwargs):
        if False:
            return 10
        '\n        Perform specified aggregation along rows/columns.\n\n        Parameters\n        ----------\n        agg : str\n            Name of the aggregation function to perform.\n        axis : {0, 1}, default: 0\n            Axis to perform aggregation along. 0 is to apply function against each column,\n            all the columns will be reduced into a single scalar. 1 is to aggregate\n            across rows.\n            *Note:* HDK storage format supports aggregation for 0 axis only, aggregation\n            along rows will be defaulted to pandas.\n        level : None, default: None\n            Serves the compatibility purpose, always have to be None.\n        **kwargs : dict\n            Additional parameters to pass to the aggregation function.\n\n        Returns\n        -------\n        DFAlgQueryCompiler\n            New single-column (``axis=1``) or single-row (``axis=0``) query compiler containing\n            the result of aggregation.\n        '
        if level is not None or axis != 0:
            raise NotImplementedError("HDK's aggregation functions does not support 'level' and 'axis' parameters.")
        if not kwargs.get('skipna', True) or kwargs.get('numeric_only'):
            raise NotImplementedError("HDK's aggregation functions does not support 'skipna' and 'numeric_only' parameters.")
        kwargs.pop('skipna', None)
        kwargs.pop('numeric_only', None)
        new_frame = self._modin_frame.agg(agg)
        new_frame = new_frame._set_index(pandas.Index.__new__(pandas.Index, data=[MODIN_UNNAMED_SERIES_LABEL], dtype='O'))
        return self.__constructor__(new_frame, shape_hint='row')

    def _get_index(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return frame's index.\n\n        Returns\n        -------\n        pandas.Index\n        "
        if self._modin_frame._has_unsupported_data:
            return default_axis_getter(0)(self)
        return self._modin_frame.index

    def _set_index(self, index):
        if False:
            print('Hello World!')
        '\n        Set new index.\n\n        Parameters\n        ----------\n        index : pandas.Index\n            A new index.\n        '
        default_axis_setter(0)(self, index)

    def _get_columns(self):
        if False:
            while True:
                i = 10
        "\n        Return frame's columns.\n\n        Returns\n        -------\n        pandas.Index\n        "
        if self._modin_frame._has_unsupported_data:
            return default_axis_getter(1)(self)
        return self._modin_frame.columns

    def _set_columns(self, columns):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set new columns.\n\n        Parameters\n        ----------\n        columns : list-like\n            New columns.\n        '
        if self._modin_frame._has_unsupported_data:
            default_axis_setter(1)(self, columns)
        else:
            try:
                self._modin_frame = self._modin_frame._set_columns(columns)
            except NotImplementedError:
                default_axis_setter(1)(self, columns)
                self._modin_frame._has_unsupported_data = True

    def fillna(self, squeeze_self=False, squeeze_value=False, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
        if False:
            while True:
                i = 10
        assert not inplace, 'inplace=True should be handled on upper level'
        if isinstance(value, dict) and len(self._modin_frame.columns) == 1 and (self._modin_frame.columns[0] == MODIN_UNNAMED_SERIES_LABEL):
            raise NotImplementedError('Series fillna with dict value')
        new_frame = self._modin_frame.fillna(value=value, method=method, axis=axis, limit=limit, downcast=downcast)
        return self.__constructor__(new_frame, self._shape_hint)

    def concat(self, axis, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, list):
            other = [other]
        assert all((isinstance(o, type(self)) for o in other)), 'Different Manager objects are being used. This is not allowed'
        sort = kwargs.get('sort', False)
        if sort is None:
            raise ValueError("The 'sort' keyword only accepts boolean values; None was passed.")
        join = kwargs.get('join', 'outer')
        ignore_index = kwargs.get('ignore_index', False)
        other_modin_frames = [o._modin_frame for o in other]
        new_modin_frame = self._modin_frame.concat(axis, other_modin_frames, join=join, sort=sort, ignore_index=ignore_index)
        return self.__constructor__(new_modin_frame)

    def drop(self, index=None, columns=None, errors: str='raise'):
        if False:
            return 10
        if index is not None:
            raise NotImplementedError('Row drop')
        if errors != 'raise':
            raise NotImplementedError('This lazy query compiler will always ' + 'raise an error on invalid columns.')
        columns = self.columns.drop(columns)
        new_frame = self._modin_frame.take_2d_labels_or_positional(row_labels=index, col_labels=columns)
        if len(columns) == 0 and new_frame._index_cols is None:
            assert index is None, "Can't copy old indexes as there was a row drop"
            new_frame.set_index_cache(self._modin_frame.index.copy())
        return self.__constructor__(new_frame)

    def dropna(self, axis=0, how=no_default, thresh=no_default, subset=None):
        if False:
            return 10
        if thresh is not no_default or axis != 0:
            raise NotImplementedError("HDK's dropna does not support 'thresh' and 'axis' parameters.")
        if subset is None:
            subset = self.columns
        if how is no_default:
            how = 'any'
        return self.__constructor__(self._modin_frame.dropna(subset=subset, how=how), shape_hint=self._shape_hint)

    def isna(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__constructor__(self._modin_frame.isna(invert=False))

    def notna(self):
        if False:
            print('Hello World!')
        return self.__constructor__(self._modin_frame.isna(invert=True))

    def invert(self):
        if False:
            while True:
                i = 10
        return self.__constructor__(self._modin_frame.invert())

    def dt_year(self):
        if False:
            while True:
                i = 10
        return self.__constructor__(self._modin_frame.dt_extract('year'), self._shape_hint)

    def dt_month(self):
        if False:
            i = 10
            return i + 15
        return self.__constructor__(self._modin_frame.dt_extract('month'), self._shape_hint)

    def dt_day(self):
        if False:
            i = 10
            return i + 15
        return self.__constructor__(self._modin_frame.dt_extract('day'), self._shape_hint)

    def dt_hour(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__constructor__(self._modin_frame.dt_extract('hour'), self._shape_hint)

    def dt_minute(self):
        if False:
            return 10
        return self.__constructor__(self._modin_frame.dt_extract('minute'), self._shape_hint)

    def dt_second(self):
        if False:
            i = 10
            return i + 15
        return self.__constructor__(self._modin_frame.dt_extract('second'), self._shape_hint)

    def dt_microsecond(self):
        if False:
            return 10
        return self.__constructor__(self._modin_frame.dt_extract('microsecond'), self._shape_hint)

    def dt_nanosecond(self):
        if False:
            return 10
        return self.__constructor__(self._modin_frame.dt_extract('nanosecond'), self._shape_hint)

    def dt_quarter(self):
        if False:
            print('Hello World!')
        return self.__constructor__(self._modin_frame.dt_extract('quarter'), self._shape_hint)

    def dt_dayofweek(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__constructor__(self._modin_frame.dt_extract('isodow'), self._shape_hint)

    def dt_weekday(self):
        if False:
            i = 10
            return i + 15
        return self.__constructor__(self._modin_frame.dt_extract('isodow'), self._shape_hint)

    def dt_dayofyear(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__constructor__(self._modin_frame.dt_extract('doy'), self._shape_hint)

    def _bin_op(self, other, op_name, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform a binary operation on a frame.\n\n        Parameters\n        ----------\n        other : any\n            The second operand.\n        op_name : str\n            Operation name.\n        **kwargs : dict\n            Keyword args.\n\n        Returns\n        -------\n        DFAlgQueryCompiler\n            A new query compiler.\n        '
        level = kwargs.get('level', None)
        if level is not None:
            return getattr(super(), op_name)(other=other, op_name=op_name, **kwargs)
        if isinstance(other, DFAlgQueryCompiler):
            shape_hint = self._shape_hint if self._shape_hint == other._shape_hint else None
            other = other._modin_frame
        else:
            shape_hint = self._shape_hint
        new_modin_frame = self._modin_frame.bin_op(other, op_name, **kwargs)
        return self.__constructor__(new_modin_frame, shape_hint)

    def add(self, other, **kwargs):
        if False:
            while True:
                i = 10
        return self._bin_op(other, 'add', **kwargs)

    def sub(self, other, **kwargs):
        if False:
            return 10
        return self._bin_op(other, 'sub', **kwargs)

    def mul(self, other, **kwargs):
        if False:
            return 10
        return self._bin_op(other, 'mul', **kwargs)

    def pow(self, other, **kwargs):
        if False:
            return 10
        return self._bin_op(other, 'pow', **kwargs)

    def mod(self, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')

        def check_int(obj):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(obj, DFAlgQueryCompiler):
                cond = all((is_integer_dtype(t) for t in obj._modin_frame.dtypes))
            elif isinstance(obj, list):
                cond = all((isinstance(i, int) for i in obj))
            else:
                cond = isinstance(obj, int)
            if not cond:
                raise NotImplementedError('Non-integer operands in modulo operation')
        check_int(self)
        check_int(other)
        return self._bin_op(other, 'mod', **kwargs)

    def floordiv(self, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._bin_op(other, 'floordiv', **kwargs)

    def truediv(self, other, **kwargs):
        if False:
            return 10
        return self._bin_op(other, 'truediv', **kwargs)

    def eq(self, other, **kwargs):
        if False:
            while True:
                i = 10
        return self._bin_op(other, 'eq', **kwargs)

    def ge(self, other, **kwargs):
        if False:
            return 10
        return self._bin_op(other, 'ge', **kwargs)

    def gt(self, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._bin_op(other, 'gt', **kwargs)

    def le(self, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._bin_op(other, 'le', **kwargs)

    def lt(self, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._bin_op(other, 'lt', **kwargs)

    def ne(self, other, **kwargs):
        if False:
            while True:
                i = 10
        return self._bin_op(other, 'ne', **kwargs)

    def __and__(self, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._bool_op(other, 'and', **kwargs)

    def __or__(self, other, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._bool_op(other, 'or', **kwargs)

    def _bool_op(self, other, op, **kwargs):
        if False:
            return 10

        def check_bool(obj):
            if False:
                return 10
            if isinstance(obj, DFAlgQueryCompiler):
                cond = all((is_bool_dtype(t) for t in obj._modin_frame.dtypes))
            elif isinstance(obj, list):
                cond = all((isinstance(i, bool) for i in obj))
            else:
                cond = isinstance(obj, bool)
            if not cond:
                raise NotImplementedError('Non-boolean operands in logic operation')
        check_bool(self)
        check_bool(other)
        return self._bin_op(other, op, **kwargs)

    def reset_index(self, **kwargs):
        if False:
            print('Hello World!')
        level = kwargs.get('level', None)
        if level is not None:
            raise NotImplementedError("HDK's reset_index does not support 'level' parameter.")
        drop = kwargs.get('drop', False)
        shape_hint = self._shape_hint if drop else None
        return self.__constructor__(self._modin_frame.reset_index(drop), shape_hint=shape_hint)

    def astype(self, col_dtypes, errors: str='raise'):
        if False:
            i = 10
            return i + 15
        if errors != 'raise':
            raise NotImplementedError('This lazy query compiler will always ' + 'raise an error on invalid type keys.')
        return self.__constructor__(self._modin_frame.astype(col_dtypes), self._shape_hint)

    def setitem(self, axis, key, value):
        if False:
            i = 10
            return i + 15
        if axis == 1 or not isinstance(value, type(self)):
            raise NotImplementedError(f"HDK's setitem does not support such set of parameters: axis={axis}, value={value}.")
        return self._setitem(axis, key, value)
    _setitem = PandasQueryCompiler._setitem

    def insert(self, loc, column, value):
        if False:
            return 10
        if isinstance(value, type(self)):
            value.columns = [column]
            return self.insert_item(axis=1, loc=loc, value=value)
        if is_list_like(value):
            raise NotImplementedError("HDK's insert does not support list-like values.")
        return self.__constructor__(self._modin_frame.insert(loc, column, value))

    def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
        if False:
            return 10
        if kwargs.get('key', None) is not None:
            raise NotImplementedError('Sort with key function')
        ignore_index = kwargs.get('ignore_index', False)
        na_position = kwargs.get('na_position', 'last')
        return self.__constructor__(self._modin_frame.sort_rows(columns, ascending, ignore_index, na_position), self._shape_hint)

    def columnarize(self):
        if False:
            while True:
                i = 10
        if self._shape_hint == 'column':
            assert len(self.columns) == 1, 'wrong shape hint'
            return self
        if self._shape_hint == 'row':
            assert len(self.index) == 1, 'wrong shape hint'
            return self.transpose()
        if len(self.columns) != 1 or (len(self.index) == 1 and self.index[0] == MODIN_UNNAMED_SERIES_LABEL):
            res = self.transpose()
            res._shape_hint = 'column'
            return res
        self._shape_hint = 'column'
        return self

    def is_series_like(self):
        if False:
            i = 10
            return i + 15
        if self._shape_hint is not None:
            return True
        return len(self.columns) == 1 or len(self.index) == 1

    def cat_codes(self):
        if False:
            i = 10
            return i + 15
        return self.__constructor__(self._modin_frame.cat_codes(), self._shape_hint)

    def has_multiindex(self, axis=0):
        if False:
            for i in range(10):
                print('nop')
        if axis == 0:
            return self._modin_frame.has_multiindex()
        assert axis == 1
        return isinstance(self.columns, pandas.MultiIndex)

    def get_index_name(self, axis=0):
        if False:
            for i in range(10):
                print('nop')
        return self.columns.name if axis else self._modin_frame.get_index_name()

    def set_index_name(self, name, axis=0):
        if False:
            return 10
        if axis == 0:
            self._modin_frame = self._modin_frame.set_index_name(name)
        else:
            self.columns.name = name

    def get_index_names(self, axis=0):
        if False:
            print('Hello World!')
        return self.columns.names if axis else self._modin_frame.get_index_names()

    def set_index_names(self, names=None, axis=0):
        if False:
            print('Hello World!')
        if axis == 0:
            self._modin_frame = self._modin_frame.set_index_names(names)
        else:
            self.columns.names = names

    def free(self):
        if False:
            while True:
                i = 10
        return
    index = property(_get_index, _set_index)
    columns = property(_get_columns, _set_columns)

    @property
    def dtypes(self):
        if False:
            print('Hello World!')
        return self._modin_frame.dtypes
_SUPPORTED_NUM_TYPE_CODES = set(np.typecodes['AllInteger'] + np.typecodes['Float'] + '?') - {np.dtype(np.float16).char}

def _check_int_or_float(op, dtypes):
    if False:
        print('Hello World!')
    for t in dtypes:
        if t.char not in _SUPPORTED_NUM_TYPE_CODES:
            raise NotImplementedError(f"Operation '{op}' on type '{t.name}'")