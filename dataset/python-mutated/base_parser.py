from __future__ import annotations
from collections import defaultdict
from copy import copy
import csv
import datetime
from enum import Enum
import itertools
from typing import TYPE_CHECKING, Any, Callable, cast, final, overload
import warnings
import numpy as np
from pandas._libs import lib, parsers
import pandas._libs.ops as libops
from pandas._libs.parsers import STR_NA_VALUES
from pandas._libs.tslibs import parsing
from pandas.compat._optional import import_optional_dependency
from pandas.errors import ParserError, ParserWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.common import ensure_object, is_bool_dtype, is_dict_like, is_extension_array_dtype, is_float_dtype, is_integer, is_integer_dtype, is_list_like, is_object_dtype, is_scalar, is_string_dtype, pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, ExtensionDtype
from pandas.core.dtypes.missing import isna
from pandas import ArrowDtype, DataFrame, DatetimeIndex, StringDtype, concat
from pandas.core import algorithms
from pandas.core.arrays import ArrowExtensionArray, BooleanArray, Categorical, ExtensionArray, FloatingArray, IntegerArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.indexes.api import Index, MultiIndex, default_index, ensure_index_from_sequences
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools
from pandas.io.common import is_potential_multi_index
if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence
    from pandas._typing import ArrayLike, DtypeArg, DtypeObj, Scalar

class ParserBase:

    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2
    _implicit_index: bool
    _first_chunk: bool
    keep_default_na: bool
    dayfirst: bool
    cache_dates: bool
    keep_date_col: bool
    usecols_dtype: str | None

    def __init__(self, kwds) -> None:
        if False:
            print('Hello World!')
        self._implicit_index = False
        self.names = kwds.get('names')
        self.orig_names: Sequence[Hashable] | None = None
        self.index_col = kwds.get('index_col', None)
        self.unnamed_cols: set = set()
        self.index_names: Sequence[Hashable] | None = None
        self.col_names: Sequence[Hashable] | None = None
        self.parse_dates = _validate_parse_dates_arg(kwds.pop('parse_dates', False))
        self._parse_date_cols: Iterable = []
        self.date_parser = kwds.pop('date_parser', lib.no_default)
        self.date_format = kwds.pop('date_format', None)
        self.dayfirst = kwds.pop('dayfirst', False)
        self.keep_date_col = kwds.pop('keep_date_col', False)
        self.na_values = kwds.get('na_values')
        self.na_fvalues = kwds.get('na_fvalues')
        self.na_filter = kwds.get('na_filter', False)
        self.keep_default_na = kwds.get('keep_default_na', True)
        self.dtype = copy(kwds.get('dtype', None))
        self.converters = kwds.get('converters')
        self.dtype_backend = kwds.get('dtype_backend')
        self.true_values = kwds.get('true_values')
        self.false_values = kwds.get('false_values')
        self.cache_dates = kwds.pop('cache_dates', True)
        self._date_conv = _make_date_converter(date_parser=self.date_parser, date_format=self.date_format, dayfirst=self.dayfirst, cache_dates=self.cache_dates)
        self.header = kwds.get('header')
        if is_list_like(self.header, allow_sets=False):
            if kwds.get('usecols'):
                raise ValueError('cannot specify usecols when specifying a multi-index header')
            if kwds.get('names'):
                raise ValueError('cannot specify names when specifying a multi-index header')
            if self.index_col is not None:
                if is_integer(self.index_col):
                    self.index_col = [self.index_col]
                elif not (is_list_like(self.index_col, allow_sets=False) and all(map(is_integer, self.index_col))):
                    raise ValueError('index_col must only contain row numbers when specifying a multi-index header')
                else:
                    self.index_col = list(self.index_col)
        self._name_processed = False
        self._first_chunk = True
        (self.usecols, self.usecols_dtype) = self._validate_usecols_arg(kwds['usecols'])
        self.on_bad_lines = kwds.get('on_bad_lines', self.BadLineHandleMethod.ERROR)

    def _validate_parse_dates_presence(self, columns: Sequence[Hashable]) -> Iterable:
        if False:
            i = 10
            return i + 15
        '\n        Check if parse_dates are in columns.\n\n        If user has provided names for parse_dates, check if those columns\n        are available.\n\n        Parameters\n        ----------\n        columns : list\n            List of names of the dataframe.\n\n        Returns\n        -------\n        The names of the columns which will get parsed later if a dict or list\n        is given as specification.\n\n        Raises\n        ------\n        ValueError\n            If column to parse_date is not in dataframe.\n\n        '
        cols_needed: Iterable
        if is_dict_like(self.parse_dates):
            cols_needed = itertools.chain(*self.parse_dates.values())
        elif is_list_like(self.parse_dates):
            cols_needed = itertools.chain.from_iterable((col if is_list_like(col) and (not isinstance(col, tuple)) else [col] for col in self.parse_dates))
        else:
            cols_needed = []
        cols_needed = list(cols_needed)
        missing_cols = ', '.join(sorted({col for col in cols_needed if isinstance(col, str) and col not in columns}))
        if missing_cols:
            raise ValueError(f"Missing column provided to 'parse_dates': '{missing_cols}'")
        return [col if isinstance(col, str) or col in columns else columns[col] for col in cols_needed]

    def close(self) -> None:
        if False:
            while True:
                i = 10
        pass

    @final
    @property
    def _has_complex_date_col(self) -> bool:
        if False:
            i = 10
            return i + 15
        return isinstance(self.parse_dates, dict) or (isinstance(self.parse_dates, list) and len(self.parse_dates) > 0 and isinstance(self.parse_dates[0], list))

    @final
    def _should_parse_dates(self, i: int) -> bool:
        if False:
            while True:
                i = 10
        if lib.is_bool(self.parse_dates):
            return bool(self.parse_dates)
        else:
            if self.index_names is not None:
                name = self.index_names[i]
            else:
                name = None
            j = i if self.index_col is None else self.index_col[i]
            return j in self.parse_dates or (name is not None and name in self.parse_dates)

    @final
    def _extract_multi_indexer_columns(self, header, index_names: Sequence[Hashable] | None, passed_names: bool=False) -> tuple[Sequence[Hashable], Sequence[Hashable] | None, Sequence[Hashable] | None, bool]:
        if False:
            print('Hello World!')
        '\n        Extract and return the names, index_names, col_names if the column\n        names are a MultiIndex.\n\n        Parameters\n        ----------\n        header: list of lists\n            The header rows\n        index_names: list, optional\n            The names of the future index\n        passed_names: bool, default False\n            A flag specifying if names where passed\n\n        '
        if len(header) < 2:
            return (header[0], index_names, None, passed_names)
        ic = self.index_col
        if ic is None:
            ic = []
        if not isinstance(ic, (list, tuple, np.ndarray)):
            ic = [ic]
        sic = set(ic)
        index_names = header.pop(-1)
        (index_names, _, _) = self._clean_index_names(index_names, self.index_col)
        field_count = len(header[0])
        if not all((len(header_iter) == field_count for header_iter in header[1:])):
            raise ParserError('Header rows must have an equal number of columns.')

        def extract(r):
            if False:
                for i in range(10):
                    print('nop')
            return tuple((r[i] for i in range(field_count) if i not in sic))
        columns = list(zip(*(extract(r) for r in header)))
        names = columns.copy()
        for single_ic in sorted(ic):
            names.insert(single_ic, single_ic)
        if len(ic):
            col_names = [r[ic[0]] if r[ic[0]] is not None and r[ic[0]] not in self.unnamed_cols else None for r in header]
        else:
            col_names = [None] * len(header)
        passed_names = True
        return (names, index_names, col_names, passed_names)

    @final
    def _maybe_make_multi_index_columns(self, columns: Sequence[Hashable], col_names: Sequence[Hashable] | None=None) -> Sequence[Hashable] | MultiIndex:
        if False:
            print('Hello World!')
        if is_potential_multi_index(columns):
            list_columns = cast(list[tuple], columns)
            return MultiIndex.from_tuples(list_columns, names=col_names)
        return columns

    @final
    def _make_index(self, data, alldata, columns, indexnamerow: list[Scalar] | None=None) -> tuple[Index | None, Sequence[Hashable] | MultiIndex]:
        if False:
            print('Hello World!')
        index: Index | None
        if not is_index_col(self.index_col) or not self.index_col:
            index = None
        elif not self._has_complex_date_col:
            simple_index = self._get_simple_index(alldata, columns)
            index = self._agg_index(simple_index)
        elif self._has_complex_date_col:
            if not self._name_processed:
                (self.index_names, _, self.index_col) = self._clean_index_names(list(columns), self.index_col)
                self._name_processed = True
            date_index = self._get_complex_date_index(data, columns)
            index = self._agg_index(date_index, try_parse_dates=False)
        if indexnamerow:
            coffset = len(indexnamerow) - len(columns)
            assert index is not None
            index = index.set_names(indexnamerow[:coffset])
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)
        return (index, columns)

    @final
    def _get_simple_index(self, data, columns):
        if False:
            while True:
                i = 10

        def ix(col):
            if False:
                print('Hello World!')
            if not isinstance(col, str):
                return col
            raise ValueError(f'Index {col} invalid')
        to_remove = []
        index = []
        for idx in self.index_col:
            i = ix(idx)
            to_remove.append(i)
            index.append(data[i])
        for i in sorted(to_remove, reverse=True):
            data.pop(i)
            if not self._implicit_index:
                columns.pop(i)
        return index

    @final
    def _get_complex_date_index(self, data, col_names):
        if False:
            while True:
                i = 10

        def _get_name(icol):
            if False:
                while True:
                    i = 10
            if isinstance(icol, str):
                return icol
            if col_names is None:
                raise ValueError(f'Must supply column order to use {icol!s} as index')
            for (i, c) in enumerate(col_names):
                if i == icol:
                    return c
        to_remove = []
        index = []
        for idx in self.index_col:
            name = _get_name(idx)
            to_remove.append(name)
            index.append(data[name])
        for c in sorted(to_remove, reverse=True):
            data.pop(c)
            col_names.remove(c)
        return index

    @final
    def _clean_mapping(self, mapping):
        if False:
            i = 10
            return i + 15
        'converts col numbers to names'
        if not isinstance(mapping, dict):
            return mapping
        clean = {}
        assert self.orig_names is not None
        for (col, v) in mapping.items():
            if isinstance(col, int) and col not in self.orig_names:
                col = self.orig_names[col]
            clean[col] = v
        if isinstance(mapping, defaultdict):
            remaining_cols = set(self.orig_names) - set(clean.keys())
            clean.update({col: mapping[col] for col in remaining_cols})
        return clean

    @final
    def _agg_index(self, index, try_parse_dates: bool=True) -> Index:
        if False:
            for i in range(10):
                print('nop')
        arrays = []
        converters = self._clean_mapping(self.converters)
        for (i, arr) in enumerate(index):
            if try_parse_dates and self._should_parse_dates(i):
                arr = self._date_conv(arr, col=self.index_names[i] if self.index_names is not None else None)
            if self.na_filter:
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                col_na_values = set()
                col_na_fvalues = set()
            if isinstance(self.na_values, dict):
                assert self.index_names is not None
                col_name = self.index_names[i]
                if col_name is not None:
                    (col_na_values, col_na_fvalues) = _get_na_values(col_name, self.na_values, self.na_fvalues, self.keep_default_na)
            clean_dtypes = self._clean_mapping(self.dtype)
            cast_type = None
            index_converter = False
            if self.index_names is not None:
                if isinstance(clean_dtypes, dict):
                    cast_type = clean_dtypes.get(self.index_names[i], None)
                if isinstance(converters, dict):
                    index_converter = converters.get(self.index_names[i]) is not None
            try_num_bool = not (cast_type and is_string_dtype(cast_type) or index_converter)
            (arr, _) = self._infer_types(arr, col_na_values | col_na_fvalues, cast_type is None, try_num_bool)
            arrays.append(arr)
        names = self.index_names
        index = ensure_index_from_sequences(arrays, names)
        return index

    @final
    def _convert_to_ndarrays(self, dct: Mapping, na_values, na_fvalues, verbose: bool=False, converters=None, dtypes=None):
        if False:
            print('Hello World!')
        result = {}
        for (c, values) in dct.items():
            conv_f = None if converters is None else converters.get(c, None)
            if isinstance(dtypes, dict):
                cast_type = dtypes.get(c, None)
            else:
                cast_type = dtypes
            if self.na_filter:
                (col_na_values, col_na_fvalues) = _get_na_values(c, na_values, na_fvalues, self.keep_default_na)
            else:
                (col_na_values, col_na_fvalues) = (set(), set())
            if c in self._parse_date_cols:
                mask = algorithms.isin(values, set(col_na_values) | col_na_fvalues)
                np.putmask(values, mask, np.nan)
                result[c] = values
                continue
            if conv_f is not None:
                if cast_type is not None:
                    warnings.warn(f'Both a converter and dtype were specified for column {c} - only the converter will be used.', ParserWarning, stacklevel=find_stack_level())
                try:
                    values = lib.map_infer(values, conv_f)
                except ValueError:
                    mask = algorithms.isin(values, list(na_values)).view(np.uint8)
                    values = lib.map_infer_mask(values, conv_f, mask)
                (cvals, na_count) = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool=False)
            else:
                is_ea = is_extension_array_dtype(cast_type)
                is_str_or_ea_dtype = is_ea or is_string_dtype(cast_type)
                try_num_bool = not (cast_type and is_str_or_ea_dtype)
                (cvals, na_count) = self._infer_types(values, set(col_na_values) | col_na_fvalues, cast_type is None, try_num_bool)
                if cast_type is not None:
                    cast_type = pandas_dtype(cast_type)
                if cast_type and (cvals.dtype != cast_type or is_ea):
                    if not is_ea and na_count > 0:
                        if is_bool_dtype(cast_type):
                            raise ValueError(f'Bool column has NA values in column {c}')
                    cvals = self._cast_types(cvals, cast_type, c)
            result[c] = cvals
            if verbose and na_count:
                print(f'Filled {na_count} NA values in column {c!s}')
        return result

    @final
    def _set_noconvert_dtype_columns(self, col_indices: list[int], names: Sequence[Hashable]) -> set[int]:
        if False:
            return 10
        '\n        Set the columns that should not undergo dtype conversions.\n\n        Currently, any column that is involved with date parsing will not\n        undergo such conversions. If usecols is specified, the positions of the columns\n        not to cast is relative to the usecols not to all columns.\n\n        Parameters\n        ----------\n        col_indices: The indices specifying order and positions of the columns\n        names: The column names which order is corresponding with the order\n               of col_indices\n\n        Returns\n        -------\n        A set of integers containing the positions of the columns not to convert.\n        '
        usecols: list[int] | list[str] | None
        noconvert_columns = set()
        if self.usecols_dtype == 'integer':
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ('empty', None):
            usecols = col_indices
        else:
            usecols = None

        def _set(x) -> int:
            if False:
                return 10
            if usecols is not None and is_integer(x):
                x = usecols[x]
            if not is_integer(x):
                x = col_indices[names.index(x)]
            return x
        if isinstance(self.parse_dates, list):
            for val in self.parse_dates:
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))
        elif isinstance(self.parse_dates, dict):
            for val in self.parse_dates.values():
                if isinstance(val, list):
                    for k in val:
                        noconvert_columns.add(_set(k))
                else:
                    noconvert_columns.add(_set(val))
        elif self.parse_dates:
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    noconvert_columns.add(_set(k))
            elif self.index_col is not None:
                noconvert_columns.add(_set(self.index_col))
        return noconvert_columns

    @final
    def _infer_types(self, values, na_values, no_dtype_specified, try_num_bool: bool=True) -> tuple[ArrayLike, int]:
        if False:
            i = 10
            return i + 15
        '\n        Infer types of values, possibly casting\n\n        Parameters\n        ----------\n        values : ndarray\n        na_values : set\n        no_dtype_specified: Specifies if we want to cast explicitly\n        try_num_bool : bool, default try\n           try to cast values to numeric (first preference) or boolean\n\n        Returns\n        -------\n        converted : ndarray or ExtensionArray\n        na_count : int\n        '
        na_count = 0
        if issubclass(values.dtype.type, (np.number, np.bool_)):
            na_values = np.array([val for val in na_values if not isinstance(val, str)])
            mask = algorithms.isin(values, na_values)
            na_count = mask.astype('uint8', copy=False).sum()
            if na_count > 0:
                if is_integer_dtype(values):
                    values = values.astype(np.float64)
                np.putmask(values, mask, np.nan)
            return (values, na_count)
        dtype_backend = self.dtype_backend
        non_default_dtype_backend = no_dtype_specified and dtype_backend is not lib.no_default
        result: ArrayLike
        if try_num_bool and is_object_dtype(values.dtype):
            try:
                (result, result_mask) = lib.maybe_convert_numeric(values, na_values, False, convert_to_masked_nullable=non_default_dtype_backend)
            except (ValueError, TypeError):
                na_count = parsers.sanitize_objects(values, na_values)
                result = values
            else:
                if non_default_dtype_backend:
                    if result_mask is None:
                        result_mask = np.zeros(result.shape, dtype=np.bool_)
                    if result_mask.all():
                        result = IntegerArray(np.ones(result_mask.shape, dtype=np.int64), result_mask)
                    elif is_integer_dtype(result):
                        result = IntegerArray(result, result_mask)
                    elif is_bool_dtype(result):
                        result = BooleanArray(result, result_mask)
                    elif is_float_dtype(result):
                        result = FloatingArray(result, result_mask)
                    na_count = result_mask.sum()
                else:
                    na_count = isna(result).sum()
        else:
            result = values
            if values.dtype == np.object_:
                na_count = parsers.sanitize_objects(values, na_values)
        if result.dtype == np.object_ and try_num_bool:
            (result, bool_mask) = libops.maybe_convert_bool(np.asarray(values), true_values=self.true_values, false_values=self.false_values, convert_to_masked_nullable=non_default_dtype_backend)
            if result.dtype == np.bool_ and non_default_dtype_backend:
                if bool_mask is None:
                    bool_mask = np.zeros(result.shape, dtype=np.bool_)
                result = BooleanArray(result, bool_mask)
            elif result.dtype == np.object_ and non_default_dtype_backend:
                if not lib.is_datetime_array(result, skipna=True):
                    result = StringDtype().construct_array_type()._from_sequence(values)
        if dtype_backend == 'pyarrow':
            pa = import_optional_dependency('pyarrow')
            if isinstance(result, np.ndarray):
                result = ArrowExtensionArray(pa.array(result, from_pandas=True))
            else:
                result = ArrowExtensionArray(pa.array(result.to_numpy(), from_pandas=True))
        return (result, na_count)

    @final
    def _cast_types(self, values: ArrayLike, cast_type: DtypeObj, column) -> ArrayLike:
        if False:
            i = 10
            return i + 15
        '\n        Cast values to specified type\n\n        Parameters\n        ----------\n        values : ndarray or ExtensionArray\n        cast_type : np.dtype or ExtensionDtype\n           dtype to cast values to\n        column : string\n            column name - used only for error reporting\n\n        Returns\n        -------\n        converted : ndarray or ExtensionArray\n        '
        if isinstance(cast_type, CategoricalDtype):
            known_cats = cast_type.categories is not None
            if not is_object_dtype(values.dtype) and (not known_cats):
                values = lib.ensure_string_array(values, skipna=False, convert_na_value=False)
            cats = Index(values).unique().dropna()
            values = Categorical._from_inferred_categories(cats, cats.get_indexer(values), cast_type, true_values=self.true_values)
        elif isinstance(cast_type, ExtensionDtype):
            array_type = cast_type.construct_array_type()
            try:
                if isinstance(cast_type, BooleanDtype):
                    return array_type._from_sequence_of_strings(values, dtype=cast_type, true_values=self.true_values, false_values=self.false_values)
                else:
                    return array_type._from_sequence_of_strings(values, dtype=cast_type)
            except NotImplementedError as err:
                raise NotImplementedError(f'Extension Array: {array_type} must implement _from_sequence_of_strings in order to be used in parser methods') from err
        elif isinstance(values, ExtensionArray):
            values = values.astype(cast_type, copy=False)
        elif issubclass(cast_type.type, str):
            values = lib.ensure_string_array(values, skipna=True, convert_na_value=False)
        else:
            try:
                values = astype_array(values, cast_type, copy=True)
            except ValueError as err:
                raise ValueError(f'Unable to convert column {column} to type {cast_type}') from err
        return values

    @overload
    def _do_date_conversions(self, names: Index, data: DataFrame) -> tuple[Sequence[Hashable] | Index, DataFrame]:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def _do_date_conversions(self, names: Sequence[Hashable], data: Mapping[Hashable, ArrayLike]) -> tuple[Sequence[Hashable], Mapping[Hashable, ArrayLike]]:
        if False:
            while True:
                i = 10
        ...

    @final
    def _do_date_conversions(self, names: Sequence[Hashable] | Index, data: Mapping[Hashable, ArrayLike] | DataFrame) -> tuple[Sequence[Hashable] | Index, Mapping[Hashable, ArrayLike] | DataFrame]:
        if False:
            for i in range(10):
                print('nop')
        if self.parse_dates is not None:
            (data, names) = _process_date_conversion(data, self._date_conv, self.parse_dates, self.index_col, self.index_names, names, keep_date_col=self.keep_date_col, dtype_backend=self.dtype_backend)
        return (names, data)

    @final
    def _check_data_length(self, columns: Sequence[Hashable], data: Sequence[ArrayLike]) -> None:
        if False:
            i = 10
            return i + 15
        'Checks if length of data is equal to length of column names.\n\n        One set of trailing commas is allowed. self.index_col not False\n        results in a ParserError previously when lengths do not match.\n\n        Parameters\n        ----------\n        columns: list of column names\n        data: list of array-likes containing the data column-wise.\n        '
        if not self.index_col and len(columns) != len(data) and columns:
            empty_str = is_object_dtype(data[-1]) and data[-1] == ''
            empty_str_or_na = empty_str | isna(data[-1])
            if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
                return
            warnings.warn('Length of header or names does not match length of data. This leads to a loss of data with index_col=False.', ParserWarning, stacklevel=find_stack_level())

    @overload
    def _evaluate_usecols(self, usecols: set[int] | Callable[[Hashable], object], names: Sequence[Hashable]) -> set[int]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def _evaluate_usecols(self, usecols: set[str], names: Sequence[Hashable]) -> set[str]:
        if False:
            print('Hello World!')
        ...

    @final
    def _evaluate_usecols(self, usecols: Callable[[Hashable], object] | set[str] | set[int], names: Sequence[Hashable]) -> set[str] | set[int]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Check whether or not the 'usecols' parameter\n        is a callable.  If so, enumerates the 'names'\n        parameter and returns a set of indices for\n        each entry in 'names' that evaluates to True.\n        If not a callable, returns 'usecols'.\n        "
        if callable(usecols):
            return {i for (i, name) in enumerate(names) if usecols(name)}
        return usecols

    @final
    def _validate_usecols_names(self, usecols, names: Sequence):
        if False:
            print('Hello World!')
        '\n        Validates that all usecols are present in a given\n        list of names. If not, raise a ValueError that\n        shows what usecols are missing.\n\n        Parameters\n        ----------\n        usecols : iterable of usecols\n            The columns to validate are present in names.\n        names : iterable of names\n            The column names to check against.\n\n        Returns\n        -------\n        usecols : iterable of usecols\n            The `usecols` parameter if the validation succeeds.\n\n        Raises\n        ------\n        ValueError : Columns were missing. Error message will list them.\n        '
        missing = [c for c in usecols if c not in names]
        if len(missing) > 0:
            raise ValueError(f'Usecols do not match columns, columns expected but not found: {missing}')
        return usecols

    @final
    def _validate_usecols_arg(self, usecols):
        if False:
            print('Hello World!')
        "\n        Validate the 'usecols' parameter.\n\n        Checks whether or not the 'usecols' parameter contains all integers\n        (column selection by index), strings (column by name) or is a callable.\n        Raises a ValueError if that is not the case.\n\n        Parameters\n        ----------\n        usecols : list-like, callable, or None\n            List of columns to use when parsing or a callable that can be used\n            to filter a list of table columns.\n\n        Returns\n        -------\n        usecols_tuple : tuple\n            A tuple of (verified_usecols, usecols_dtype).\n\n            'verified_usecols' is either a set if an array-like is passed in or\n            'usecols' if a callable or None is passed in.\n\n            'usecols_dtype` is the inferred dtype of 'usecols' if an array-like\n            is passed in or None if a callable or None is passed in.\n        "
        msg = "'usecols' must either be list-like of all strings, all unicode, all integers or a callable."
        if usecols is not None:
            if callable(usecols):
                return (usecols, None)
            if not is_list_like(usecols):
                raise ValueError(msg)
            usecols_dtype = lib.infer_dtype(usecols, skipna=False)
            if usecols_dtype not in ('empty', 'integer', 'string'):
                raise ValueError(msg)
            usecols = set(usecols)
            return (usecols, usecols_dtype)
        return (usecols, None)

    @final
    def _clean_index_names(self, columns, index_col) -> tuple[list | None, list, list]:
        if False:
            i = 10
            return i + 15
        if not is_index_col(index_col):
            return (None, columns, index_col)
        columns = list(columns)
        if not columns:
            return ([None] * len(index_col), columns, index_col)
        cp_cols = list(columns)
        index_names: list[str | int | None] = []
        index_col = list(index_col)
        for (i, c) in enumerate(index_col):
            if isinstance(c, str):
                index_names.append(c)
                for (j, name) in enumerate(cp_cols):
                    if name == c:
                        index_col[i] = j
                        columns.remove(name)
                        break
            else:
                name = cp_cols[c]
                columns.remove(name)
                index_names.append(name)
        for (i, name) in enumerate(index_names):
            if isinstance(name, str) and name in self.unnamed_cols:
                index_names[i] = None
        return (index_names, columns, index_col)

    @final
    def _get_empty_meta(self, columns, dtype: DtypeArg | None=None):
        if False:
            return 10
        columns = list(columns)
        index_col = self.index_col
        index_names = self.index_names
        dtype_dict: defaultdict[Hashable, Any]
        if not is_dict_like(dtype):
            default_dtype = dtype or object
            dtype_dict = defaultdict(lambda : default_dtype)
        else:
            dtype = cast(dict, dtype)
            dtype_dict = defaultdict(lambda : object, {columns[k] if is_integer(k) else k: v for (k, v) in dtype.items()})
        index: Index
        if (index_col is None or index_col is False) or index_names is None:
            index = default_index(0)
        else:
            data = [Series([], dtype=dtype_dict[name]) for name in index_names]
            index = ensure_index_from_sequences(data, names=index_names)
            index_col.sort()
            for (i, n) in enumerate(index_col):
                columns.pop(n - i)
        col_dict = {col_name: Series([], dtype=dtype_dict[col_name]) for col_name in columns}
        return (index, columns, col_dict)

def _make_date_converter(date_parser=lib.no_default, dayfirst: bool=False, cache_dates: bool=True, date_format: dict[Hashable, str] | str | None=None):
    if False:
        print('Hello World!')
    if date_parser is not lib.no_default:
        warnings.warn("The argument 'date_parser' is deprecated and will be removed in a future version. Please use 'date_format' instead, or read your data in as 'object' dtype and then call 'to_datetime'.", FutureWarning, stacklevel=find_stack_level())
    if date_parser is not lib.no_default and date_format is not None:
        raise TypeError("Cannot use both 'date_parser' and 'date_format'")

    def unpack_if_single_element(arg):
        if False:
            i = 10
            return i + 15
        if isinstance(arg, np.ndarray) and arg.ndim == 1 and (len(arg) == 1):
            return arg[0]
        return arg

    def converter(*date_cols, col: Hashable):
        if False:
            for i in range(10):
                print('nop')
        if len(date_cols) == 1 and date_cols[0].dtype.kind in 'Mm':
            return date_cols[0]
        if date_parser is lib.no_default:
            strs = parsing.concat_date_cols(date_cols)
            date_fmt = date_format.get(col) if isinstance(date_format, dict) else date_format
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*parsing datetimes with mixed time zones will raise an error', category=FutureWarning)
                str_objs = ensure_object(strs)
                try:
                    result = tools.to_datetime(str_objs, format=date_fmt, utc=False, dayfirst=dayfirst, cache=cache_dates)
                except (ValueError, TypeError):
                    return str_objs
            if isinstance(result, DatetimeIndex):
                arr = result.to_numpy()
                arr.flags.writeable = True
                return arr
            return result._values
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*parsing datetimes with mixed time zones will raise an error', category=FutureWarning)
                    pre_parsed = date_parser(*(unpack_if_single_element(arg) for arg in date_cols))
                    try:
                        result = tools.to_datetime(pre_parsed, cache=cache_dates)
                    except (ValueError, TypeError):
                        result = pre_parsed
                if isinstance(result, datetime.datetime):
                    raise Exception('scalar parser')
                return result
            except Exception:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', '.*parsing datetimes with mixed time zones will raise an error', category=FutureWarning)
                    pre_parsed = parsing.try_parse_dates(parsing.concat_date_cols(date_cols), parser=date_parser)
                    try:
                        return tools.to_datetime(pre_parsed)
                    except (ValueError, TypeError):
                        return pre_parsed
    return converter
parser_defaults = {'delimiter': None, 'escapechar': None, 'quotechar': '"', 'quoting': csv.QUOTE_MINIMAL, 'doublequote': True, 'skipinitialspace': False, 'lineterminator': None, 'header': 'infer', 'index_col': None, 'names': None, 'skiprows': None, 'skipfooter': 0, 'nrows': None, 'na_values': None, 'keep_default_na': True, 'true_values': None, 'false_values': None, 'converters': None, 'dtype': None, 'cache_dates': True, 'thousands': None, 'comment': None, 'decimal': '.', 'parse_dates': False, 'keep_date_col': False, 'dayfirst': False, 'date_parser': lib.no_default, 'date_format': None, 'usecols': None, 'chunksize': None, 'verbose': False, 'encoding': None, 'compression': None, 'skip_blank_lines': True, 'encoding_errors': 'strict', 'on_bad_lines': ParserBase.BadLineHandleMethod.ERROR, 'dtype_backend': lib.no_default}

def _process_date_conversion(data_dict, converter: Callable, parse_spec, index_col, index_names, columns, keep_date_col: bool=False, dtype_backend=lib.no_default):
    if False:
        for i in range(10):
            print('nop')

    def _isindex(colspec):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(index_col, list) and colspec in index_col or (isinstance(index_names, list) and colspec in index_names)
    new_cols = []
    new_data = {}
    orig_names = columns
    columns = list(columns)
    date_cols = set()
    if parse_spec is None or isinstance(parse_spec, bool):
        return (data_dict, columns)
    if isinstance(parse_spec, list):
        for colspec in parse_spec:
            if is_scalar(colspec) or isinstance(colspec, tuple):
                if isinstance(colspec, int) and colspec not in data_dict:
                    colspec = orig_names[colspec]
                if _isindex(colspec):
                    continue
                elif dtype_backend == 'pyarrow':
                    import pyarrow as pa
                    dtype = data_dict[colspec].dtype
                    if isinstance(dtype, ArrowDtype) and (pa.types.is_timestamp(dtype.pyarrow_dtype) or pa.types.is_date(dtype.pyarrow_dtype)):
                        continue
                data_dict[colspec] = converter(np.asarray(data_dict[colspec]), col=colspec)
            else:
                (new_name, col, old_names) = _try_convert_dates(converter, colspec, data_dict, orig_names)
                if new_name in data_dict:
                    raise ValueError(f'New date column already in dict {new_name}')
                new_data[new_name] = col
                new_cols.append(new_name)
                date_cols.update(old_names)
    elif isinstance(parse_spec, dict):
        for (new_name, colspec) in parse_spec.items():
            if new_name in data_dict:
                raise ValueError(f'Date column {new_name} already in dict')
            (_, col, old_names) = _try_convert_dates(converter, colspec, data_dict, orig_names, target_name=new_name)
            new_data[new_name] = col
            if len(colspec) == 1:
                new_data[colspec[0]] = col
            new_cols.append(new_name)
            date_cols.update(old_names)
    if isinstance(data_dict, DataFrame):
        data_dict = concat([DataFrame(new_data), data_dict], axis=1, copy=False)
    else:
        data_dict.update(new_data)
    new_cols.extend(columns)
    if not keep_date_col:
        for c in list(date_cols):
            data_dict.pop(c)
            new_cols.remove(c)
    return (data_dict, new_cols)

def _try_convert_dates(parser: Callable, colspec, data_dict, columns, target_name: str | None=None):
    if False:
        for i in range(10):
            print('nop')
    colset = set(columns)
    colnames = []
    for c in colspec:
        if c in colset:
            colnames.append(c)
        elif isinstance(c, int) and c not in columns:
            colnames.append(columns[c])
        else:
            colnames.append(c)
    new_name: tuple | str
    if all((isinstance(x, tuple) for x in colnames)):
        new_name = tuple(map('_'.join, zip(*colnames)))
    else:
        new_name = '_'.join([str(x) for x in colnames])
    to_parse = [np.asarray(data_dict[c]) for c in colnames if c in data_dict]
    new_col = parser(*to_parse, col=new_name if target_name is None else target_name)
    return (new_name, new_col, colnames)

def _get_na_values(col, na_values, na_fvalues, keep_default_na: bool):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the NaN values for a given column.\n\n    Parameters\n    ----------\n    col : str\n        The name of the column.\n    na_values : array-like, dict\n        The object listing the NaN values as strings.\n    na_fvalues : array-like, dict\n        The object listing the NaN values as floats.\n    keep_default_na : bool\n        If `na_values` is a dict, and the column is not mapped in the\n        dictionary, whether to return the default NaN values or the empty set.\n\n    Returns\n    -------\n    nan_tuple : A length-two tuple composed of\n\n        1) na_values : the string NaN values for that column.\n        2) na_fvalues : the float NaN values for that column.\n    '
    if isinstance(na_values, dict):
        if col in na_values:
            return (na_values[col], na_fvalues[col])
        else:
            if keep_default_na:
                return (STR_NA_VALUES, set())
            return (set(), set())
    else:
        return (na_values, na_fvalues)

def _validate_parse_dates_arg(parse_dates):
    if False:
        return 10
    "\n    Check whether or not the 'parse_dates' parameter\n    is a non-boolean scalar. Raises a ValueError if\n    that is the case.\n    "
    msg = "Only booleans, lists, and dictionaries are accepted for the 'parse_dates' parameter"
    if not (parse_dates is None or lib.is_bool(parse_dates) or isinstance(parse_dates, (list, dict))):
        raise TypeError(msg)
    return parse_dates

def is_index_col(col) -> bool:
    if False:
        i = 10
        return i + 15
    return col is not None and col is not False