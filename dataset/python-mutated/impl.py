from collections import OrderedDict, abc
from copy import copy
from datetime import datetime, timedelta
from typing import Any, List, Optional, Sequence, Set, Union
import attr
import numpy as np
import pandas
from hypothesis import strategies as st
from hypothesis._settings import note_deprecation
from hypothesis.control import reject
from hypothesis.errors import InvalidArgument
from hypothesis.extra import numpy as npst
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check, check_function
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type, check_valid_interval, check_valid_size, try_convert
from hypothesis.strategies._internal.strategies import Ex, check_strategy
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
try:
    from pandas.core.arrays.integer import IntegerDtype
except ImportError:
    IntegerDtype = ()

def dtype_for_elements_strategy(s):
    if False:
        for i in range(10):
            print('nop')
    return st.shared(s.map(lambda x: pandas.Series([x]).dtype), key=('hypothesis.extra.pandas.dtype_for_elements_strategy', s))

def infer_dtype_if_necessary(dtype, values, elements, draw):
    if False:
        i = 10
        return i + 15
    if dtype is None and (not values):
        return draw(dtype_for_elements_strategy(elements))
    return dtype

@check_function
def elements_and_dtype(elements, dtype, source=None):
    if False:
        print('Hello World!')
    if source is None:
        prefix = ''
    else:
        prefix = f'{source}.'
    if elements is not None:
        check_strategy(elements, f'{prefix}elements')
    else:
        with check('dtype is not None'):
            if dtype is None:
                raise InvalidArgument(f'At least one of {prefix}elements or {prefix}dtype must be provided.')
    with check('isinstance(dtype, CategoricalDtype)'):
        if pandas.api.types.CategoricalDtype.is_dtype(dtype):
            raise InvalidArgument(f'{prefix}dtype is categorical, which is currently unsupported')
    if isinstance(dtype, type) and issubclass(dtype, IntegerDtype):
        raise InvalidArgument(f'Passed dtype={dtype!r} is a dtype class, please pass in an instance of this class.Otherwise it would be treated as dtype=object')
    if isinstance(dtype, type) and np.dtype(dtype).kind == 'O' and (dtype is not object):
        err_msg = f'Passed dtype={dtype!r} is not a valid Pandas dtype.'
        if issubclass(dtype, datetime):
            err_msg += ' To generate valid datetimes, pass `dtype="datetime64[ns]"`'
            raise InvalidArgument(err_msg)
        elif issubclass(dtype, timedelta):
            err_msg += ' To generate valid timedeltas, pass `dtype="timedelta64[ns]"`'
            raise InvalidArgument(err_msg)
        note_deprecation(f"{err_msg}  We'll treat it as dtype=object for now, but this will be an error in a future version.", since='2021-12-31', has_codemod=False, stacklevel=1)
    if isinstance(dtype, st.SearchStrategy):
        raise InvalidArgument(f'Passed dtype={dtype!r} is a strategy, but we require a concrete dtype here.  See https://stackoverflow.com/q/74355937 for workaround patterns.')
    _get_subclasses = getattr(IntegerDtype, '__subclasses__', list)
    dtype = {t.name: t() for t in _get_subclasses()}.get(dtype, dtype)
    if isinstance(dtype, IntegerDtype):
        is_na_dtype = True
        dtype = np.dtype(dtype.name.lower())
    elif dtype is not None:
        is_na_dtype = False
        dtype = try_convert(np.dtype, dtype, 'dtype')
    else:
        is_na_dtype = False
    if elements is None:
        elements = npst.from_dtype(dtype)
        if is_na_dtype:
            elements = st.none() | elements
    elif dtype is not None:

        def convert_element(value):
            if False:
                print('Hello World!')
            if is_na_dtype and value is None:
                return None
            name = f'draw({prefix}elements)'
            try:
                return np.array([value], dtype=dtype)[0]
            except (TypeError, ValueError):
                raise InvalidArgument('Cannot convert %s=%r of type %s to dtype %s' % (name, value, type(value).__name__, dtype.str)) from None
        elements = elements.map(convert_element)
    assert elements is not None
    return (elements, dtype)

class ValueIndexStrategy(st.SearchStrategy):

    def __init__(self, elements, dtype, min_size, max_size, unique, name):
        if False:
            print('Hello World!')
        super().__init__()
        self.elements = elements
        self.dtype = dtype
        self.min_size = min_size
        self.max_size = max_size
        self.unique = unique
        self.name = name

    def do_draw(self, data):
        if False:
            print('Hello World!')
        result = []
        seen = set()
        iterator = cu.many(data, min_size=self.min_size, max_size=self.max_size, average_size=(self.min_size + self.max_size) / 2)
        while iterator.more():
            elt = data.draw(self.elements)
            if self.unique:
                if elt in seen:
                    iterator.reject()
                    continue
                seen.add(elt)
            result.append(elt)
        dtype = infer_dtype_if_necessary(dtype=self.dtype, values=result, elements=self.elements, draw=data.draw)
        return pandas.Index(result, dtype=dtype, tupleize_cols=False, name=data.draw(self.name))
DEFAULT_MAX_SIZE = 10

@cacheable
@defines_strategy()
def range_indexes(min_size: int=0, max_size: Optional[int]=None, name: st.SearchStrategy[Optional[str]]=st.none()) -> st.SearchStrategy[pandas.RangeIndex]:
    if False:
        while True:
            i = 10
    'Provides a strategy which generates an :class:`~pandas.Index` whose\n    values are 0, 1, ..., n for some n.\n\n    Arguments:\n\n    * min_size is the smallest number of elements the index can have.\n    * max_size is the largest number of elements the index can have. If None\n      it will default to some suitable value based on min_size.\n    * name is the name of the index. If st.none(), the index will have no name.\n    '
    check_valid_size(min_size, 'min_size')
    check_valid_size(max_size, 'max_size')
    if max_size is None:
        max_size = min([min_size + DEFAULT_MAX_SIZE, 2 ** 63 - 1])
    check_valid_interval(min_size, max_size, 'min_size', 'max_size')
    check_strategy(name)
    return st.builds(pandas.RangeIndex, st.integers(min_size, max_size), name=name)

@cacheable
@defines_strategy()
def indexes(*, elements: Optional[st.SearchStrategy[Ex]]=None, dtype: Any=None, min_size: int=0, max_size: Optional[int]=None, unique: bool=True, name: st.SearchStrategy[Optional[str]]=st.none()) -> st.SearchStrategy[pandas.Index]:
    if False:
        print('Hello World!')
    'Provides a strategy for producing a :class:`pandas.Index`.\n\n    Arguments:\n\n    * elements is a strategy which will be used to generate the individual\n      values of the index. If None, it will be inferred from the dtype. Note:\n      even if the elements strategy produces tuples, the generated value\n      will not be a MultiIndex, but instead be a normal index whose elements\n      are tuples.\n    * dtype is the dtype of the resulting index. If None, it will be inferred\n      from the elements strategy. At least one of dtype or elements must be\n      provided.\n    * min_size is the minimum number of elements in the index.\n    * max_size is the maximum number of elements in the index. If None then it\n      will default to a suitable small size. If you want larger indexes you\n      should pass a max_size explicitly.\n    * unique specifies whether all of the elements in the resulting index\n      should be distinct.\n    * name is a strategy for strings or ``None``, which will be passed to\n      the :class:`pandas.Index` constructor.\n    '
    check_valid_size(min_size, 'min_size')
    check_valid_size(max_size, 'max_size')
    check_valid_interval(min_size, max_size, 'min_size', 'max_size')
    check_type(bool, unique, 'unique')
    (elements, dtype) = elements_and_dtype(elements, dtype)
    if max_size is None:
        max_size = min_size + DEFAULT_MAX_SIZE
    return ValueIndexStrategy(elements, dtype, min_size, max_size, unique, name)

@defines_strategy()
def series(*, elements: Optional[st.SearchStrategy[Ex]]=None, dtype: Any=None, index: Optional[st.SearchStrategy[Union[Sequence, pandas.Index]]]=None, fill: Optional[st.SearchStrategy[Ex]]=None, unique: bool=False, name: st.SearchStrategy[Optional[str]]=st.none()) -> st.SearchStrategy[pandas.Series]:
    if False:
        i = 10
        return i + 15
    "Provides a strategy for producing a :class:`pandas.Series`.\n\n    Arguments:\n\n    * elements: a strategy that will be used to generate the individual\n      values in the series. If None, we will attempt to infer a suitable\n      default from the dtype.\n\n    * dtype: the dtype of the resulting series and may be any value\n      that can be passed to :class:`numpy.dtype`. If None, will use\n      pandas's standard behaviour to infer it from the type of the elements\n      values. Note that if the type of values that comes out of your\n      elements strategy varies, then so will the resulting dtype of the\n      series.\n\n    * index: If not None, a strategy for generating indexes for the\n      resulting Series. This can generate either :class:`pandas.Index`\n      objects or any sequence of values (which will be passed to the\n      Index constructor).\n\n      You will probably find it most convenient to use the\n      :func:`~hypothesis.extra.pandas.indexes` or\n      :func:`~hypothesis.extra.pandas.range_indexes` function to produce\n      values for this argument.\n\n    * name: is a strategy for strings or ``None``, which will be passed to\n      the :class:`pandas.Series` constructor.\n\n    Usage:\n\n    .. code-block:: pycon\n\n        >>> series(dtype=int).example()\n        0   -2001747478\n        1    1153062837\n    "
    if index is None:
        index = range_indexes()
    else:
        check_strategy(index, 'index')
    (elements, np_dtype) = elements_and_dtype(elements, dtype)
    index_strategy = index
    if np_dtype is not None and np_dtype.kind == 'O' and (not isinstance(dtype, IntegerDtype)):
        dtype = np_dtype

    @st.composite
    def result(draw):
        if False:
            for i in range(10):
                print('nop')
        index = draw(index_strategy)
        if len(index) > 0:
            if dtype is not None:
                result_data = draw(npst.arrays(dtype=object, elements=elements, shape=len(index), fill=fill, unique=unique)).tolist()
            else:
                result_data = list(draw(npst.arrays(dtype=object, elements=elements, shape=len(index), fill=fill, unique=unique)).tolist())
            return pandas.Series(result_data, index=index, dtype=dtype, name=draw(name))
        else:
            return pandas.Series((), index=index, dtype=dtype if dtype is not None else draw(dtype_for_elements_strategy(elements)), name=draw(name))
    return result()

@attr.s(slots=True)
class column:
    """Data object for describing a column in a DataFrame.

    Arguments:

    * name: the column name, or None to default to the column position. Must
      be hashable, but can otherwise be any value supported as a pandas column
      name.
    * elements: the strategy for generating values in this column, or None
      to infer it from the dtype.
    * dtype: the dtype of the column, or None to infer it from the element
      strategy. At least one of dtype or elements must be provided.
    * fill: A default value for elements of the column. See
      :func:`~hypothesis.extra.numpy.arrays` for a full explanation.
    * unique: If all values in this column should be distinct.
    """
    name = attr.ib(default=None)
    elements = attr.ib(default=None)
    dtype = attr.ib(default=None, repr=get_pretty_function_description)
    fill = attr.ib(default=None)
    unique = attr.ib(default=False)

def columns(names_or_number: Union[int, Sequence[str]], *, dtype: Any=None, elements: Optional[st.SearchStrategy[Ex]]=None, fill: Optional[st.SearchStrategy[Ex]]=None, unique: bool=False) -> List[column]:
    if False:
        while True:
            i = 10
    'A convenience function for producing a list of :class:`column` objects\n    of the same general shape.\n\n    The names_or_number argument is either a sequence of values, the\n    elements of which will be used as the name for individual column\n    objects, or a number, in which case that many unnamed columns will\n    be created. All other arguments are passed through verbatim to\n    create the columns.\n    '
    if isinstance(names_or_number, (int, float)):
        names: List[Union[int, str, None]] = [None] * names_or_number
    else:
        names = list(names_or_number)
    return [column(name=n, dtype=dtype, elements=elements, fill=fill, unique=unique) for n in names]

@defines_strategy()
def data_frames(columns: Optional[Sequence[column]]=None, *, rows: Optional[st.SearchStrategy[Union[dict, Sequence[Any]]]]=None, index: Optional[st.SearchStrategy[Ex]]=None) -> st.SearchStrategy[pandas.DataFrame]:
    if False:
        for i in range(10):
            print('nop')
    'Provides a strategy for producing a :class:`pandas.DataFrame`.\n\n    Arguments:\n\n    * columns: An iterable of :class:`column` objects describing the shape\n      of the generated DataFrame.\n\n    * rows: A strategy for generating a row object. Should generate\n      either dicts mapping column names to values or a sequence mapping\n      column position to the value in that position (note that unlike the\n      :class:`pandas.DataFrame` constructor, single values are not allowed\n      here. Passing e.g. an integer is an error, even if there is only one\n      column).\n\n      At least one of rows and columns must be provided. If both are\n      provided then the generated rows will be validated against the\n      columns and an error will be raised if they don\'t match.\n\n      Caveats on using rows:\n\n      * In general you should prefer using columns to rows, and only use\n        rows if the columns interface is insufficiently flexible to\n        describe what you need - you will get better performance and\n        example quality that way.\n      * If you provide rows and not columns, then the shape and dtype of\n        the resulting DataFrame may vary. e.g. if you have a mix of int\n        and float in the values for one column in your row entries, the\n        column will sometimes have an integral dtype and sometimes a float.\n\n    * index: If not None, a strategy for generating indexes for the\n      resulting DataFrame. This can generate either :class:`pandas.Index`\n      objects or any sequence of values (which will be passed to the\n      Index constructor).\n\n      You will probably find it most convenient to use the\n      :func:`~hypothesis.extra.pandas.indexes` or\n      :func:`~hypothesis.extra.pandas.range_indexes` function to produce\n      values for this argument.\n\n    Usage:\n\n    The expected usage pattern is that you use :class:`column` and\n    :func:`columns` to specify a fixed shape of the DataFrame you want as\n    follows. For example the following gives a two column data frame:\n\n    .. code-block:: pycon\n\n        >>> from hypothesis.extra.pandas import column, data_frames\n        >>> data_frames([\n        ... column(\'A\', dtype=int), column(\'B\', dtype=float)]).example()\n                    A              B\n        0  2021915903  1.793898e+232\n        1  1146643993            inf\n        2 -2096165693   1.000000e+07\n\n    If you want the values in different columns to interact in some way you\n    can use the rows argument. For example the following gives a two column\n    DataFrame where the value in the first column is always at most the value\n    in the second:\n\n    .. code-block:: pycon\n\n        >>> from hypothesis.extra.pandas import column, data_frames\n        >>> import hypothesis.strategies as st\n        >>> data_frames(\n        ...     rows=st.tuples(st.floats(allow_nan=False),\n        ...                    st.floats(allow_nan=False)).map(sorted)\n        ... ).example()\n                       0             1\n        0  -3.402823e+38  9.007199e+15\n        1 -1.562796e-298  5.000000e-01\n\n    You can also combine the two:\n\n    .. code-block:: pycon\n\n        >>> from hypothesis.extra.pandas import columns, data_frames\n        >>> import hypothesis.strategies as st\n        >>> data_frames(\n        ...     columns=columns(["lo", "hi"], dtype=float),\n        ...     rows=st.tuples(st.floats(allow_nan=False),\n        ...                    st.floats(allow_nan=False)).map(sorted)\n        ... ).example()\n                 lo            hi\n        0   9.314723e-49  4.353037e+45\n        1  -9.999900e-01  1.000000e+07\n        2 -2.152861e+134 -1.069317e-73\n\n    (Note that the column dtype must still be specified and will not be\n    inferred from the rows. This restriction may be lifted in future).\n\n    Combining rows and columns has the following behaviour:\n\n    * The column names and dtypes will be used.\n    * If the column is required to be unique, this will be enforced.\n    * Any values missing from the generated rows will be provided using the\n      column\'s fill.\n    * Any values in the row not present in the column specification (if\n      dicts are passed, if there are keys with no corresponding column name,\n      if sequences are passed if there are too many items) will result in\n      InvalidArgument being raised.\n    '
    if index is None:
        index = range_indexes()
    else:
        check_strategy(index, 'index')
    index_strategy = index
    if columns is None:
        if rows is None:
            raise InvalidArgument('At least one of rows and columns must be provided')
        else:

            @st.composite
            def rows_only(draw):
                if False:
                    for i in range(10):
                        print('nop')
                index = draw(index_strategy)

                @check_function
                def row():
                    if False:
                        i = 10
                        return i + 15
                    result = draw(rows)
                    check_type(abc.Iterable, result, 'draw(row)')
                    return result
                if len(index) > 0:
                    return pandas.DataFrame([row() for _ in index], index=index)
                else:
                    base = pandas.DataFrame([row()])
                    return base.drop(0)
            return rows_only()
    assert columns is not None
    cols = try_convert(tuple, columns, 'columns')
    rewritten_columns = []
    column_names: Set[str] = set()
    for (i, c) in enumerate(cols):
        check_type(column, c, f'columns[{i}]')
        c = copy(c)
        if c.name is None:
            label = f'columns[{i}]'
            c.name = i
        else:
            label = c.name
            try:
                hash(c.name)
            except TypeError:
                raise InvalidArgument(f'Column names must be hashable, but columns[{i}].name was {c.name!r} of type {type(c.name).__name__}, which cannot be hashed.') from None
        if c.name in column_names:
            raise InvalidArgument(f'duplicate definition of column name {c.name!r}')
        column_names.add(c.name)
        (c.elements, _) = elements_and_dtype(c.elements, c.dtype, label)
        if c.dtype is None and rows is not None:
            raise InvalidArgument('Must specify a dtype for all columns when combining rows with columns.')
        c.fill = npst.fill_for(fill=c.fill, elements=c.elements, unique=c.unique, name=label)
        rewritten_columns.append(c)
    if rows is None:

        @st.composite
        def just_draw_columns(draw):
            if False:
                for i in range(10):
                    print('nop')
            index = draw(index_strategy)
            local_index_strategy = st.just(index)
            data = OrderedDict(((c.name, None) for c in rewritten_columns))
            columns_without_fill = [c for c in rewritten_columns if c.fill.is_empty]
            if columns_without_fill:
                for c in columns_without_fill:
                    data[c.name] = pandas.Series(np.zeros(shape=len(index), dtype=object), index=index, dtype=c.dtype)
                seen = {c.name: set() for c in columns_without_fill if c.unique}
                for i in range(len(index)):
                    for c in columns_without_fill:
                        if c.unique:
                            for _ in range(5):
                                value = draw(c.elements)
                                if value not in seen[c.name]:
                                    seen[c.name].add(value)
                                    break
                            else:
                                reject()
                        else:
                            value = draw(c.elements)
                        try:
                            data[c.name][i] = value
                        except ValueError as err:
                            if c.dtype is None and (not isinstance(value, (float, int, str, bool, datetime, timedelta))):
                                raise ValueError(f'Failed to add value={value!r} to column {c.name} with dtype=None.  Maybe passing dtype=object would help?') from err
                            raise
            for c in rewritten_columns:
                if not c.fill.is_empty:
                    data[c.name] = draw(series(index=local_index_strategy, dtype=c.dtype, elements=c.elements, fill=c.fill, unique=c.unique))
            return pandas.DataFrame(data, index=index)
        return just_draw_columns()
    else:

        @st.composite
        def assign_rows(draw):
            if False:
                print('Hello World!')
            index = draw(index_strategy)
            result = pandas.DataFrame(OrderedDict(((c.name, pandas.Series(np.zeros(dtype=c.dtype, shape=len(index)), dtype=c.dtype)) for c in rewritten_columns)), index=index)
            fills = {}
            any_unique = any((c.unique for c in rewritten_columns))
            if any_unique:
                all_seen = [set() if c.unique else None for c in rewritten_columns]
                while all_seen[-1] is None:
                    all_seen.pop()
            for row_index in range(len(index)):
                for _ in range(5):
                    original_row = draw(rows)
                    row = original_row
                    if isinstance(row, dict):
                        as_list = [None] * len(rewritten_columns)
                        for (i, c) in enumerate(rewritten_columns):
                            try:
                                as_list[i] = row[c.name]
                            except KeyError:
                                try:
                                    as_list[i] = fills[i]
                                except KeyError:
                                    if c.fill.is_empty:
                                        raise InvalidArgument(f'Empty fill strategy in {c!r} cannot complete row {original_row!r}') from None
                                    fills[i] = draw(c.fill)
                                    as_list[i] = fills[i]
                        for k in row:
                            if k not in column_names:
                                raise InvalidArgument('Row %r contains column %r not in columns %r)' % (row, k, [c.name for c in rewritten_columns]))
                        row = as_list
                    if any_unique:
                        has_duplicate = False
                        for (seen, value) in zip(all_seen, row):
                            if seen is None:
                                continue
                            if value in seen:
                                has_duplicate = True
                                break
                            seen.add(value)
                        if has_duplicate:
                            continue
                    row = list(try_convert(tuple, row, 'draw(rows)'))
                    if len(row) > len(rewritten_columns):
                        raise InvalidArgument(f'Row {original_row!r} contains too many entries. Has {len(row)} but expected at most {len(rewritten_columns)}')
                    while len(row) < len(rewritten_columns):
                        c = rewritten_columns[len(row)]
                        if c.fill.is_empty:
                            raise InvalidArgument(f'Empty fill strategy in {c!r} cannot complete row {original_row!r}')
                        row.append(draw(c.fill))
                    result.iloc[row_index] = row
                    break
                else:
                    reject()
            return result
        return assign_rows()