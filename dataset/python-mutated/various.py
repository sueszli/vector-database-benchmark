from __future__ import annotations
import inspect
import os
import re
import sys
import warnings
from collections.abc import MappingView, Sized
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable, Literal, Sequence, TypeVar
import polars as pl
from polars import functions as F
from polars.datatypes import FLOAT_DTYPES, INTEGER_DTYPES, Boolean, Date, Datetime, Decimal, Duration, Int64, Time, Utf8, unpack_dtypes
from polars.dependencies import _PYARROW_AVAILABLE, _check_for_numpy
from polars.dependencies import numpy as np
if TYPE_CHECKING:
    from collections.abc import Reversible
    from polars import DataFrame
    from polars.type_aliases import PolarsDataType, PolarsIntegerType, SizeUnit
    if sys.version_info >= (3, 10):
        from typing import ParamSpec, TypeGuard
    else:
        from typing_extensions import ParamSpec, TypeGuard
    P = ParamSpec('P')
    T = TypeVar('T')
if sys.version_info >= (3, 11):
    _views: list[Reversible[Any]] = [{}.keys(), {}.values(), {}.items()]
    _reverse_mapping_views = tuple((type(reversed(view)) for view in _views))

def _process_null_values(null_values: None | str | Sequence[str] | dict[str, str]=None) -> None | str | Sequence[str] | list[tuple[str, str]]:
    if False:
        print('Hello World!')
    if isinstance(null_values, dict):
        return list(null_values.items())
    else:
        return null_values

def _is_generator(val: object) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return isinstance(val, (Generator, Iterable)) and (not isinstance(val, Sized)) or isinstance(val, MappingView) or (sys.version_info >= (3, 11) and isinstance(val, _reverse_mapping_views))

def _is_iterable_of(val: Iterable[object], eltype: type | tuple[type, ...]) -> bool:
    if False:
        return 10
    'Check whether the given iterable is of the given type(s).'
    return all((isinstance(x, eltype) for x in val))

def is_bool_sequence(val: object, *, include_series: bool=False) -> TypeGuard[Sequence[bool]]:
    if False:
        for i in range(10):
            print('nop')
    'Check whether the given sequence is a sequence of booleans.'
    if _check_for_numpy(val) and isinstance(val, np.ndarray):
        return val.dtype == np.bool_
    elif include_series and isinstance(val, pl.Series):
        return val.dtype == pl.Boolean
    return isinstance(val, Sequence) and _is_iterable_of(val, bool)

def is_int_sequence(val: object, *, include_series: bool=False) -> TypeGuard[Sequence[int]]:
    if False:
        return 10
    'Check whether the given sequence is a sequence of integers.'
    if _check_for_numpy(val) and isinstance(val, np.ndarray):
        return np.issubdtype(val.dtype, np.integer)
    elif include_series and isinstance(val, pl.Series):
        return val.dtype.is_integer()
    return isinstance(val, Sequence) and _is_iterable_of(val, int)

def is_sequence(val: object, *, include_series: bool=False) -> TypeGuard[Sequence[Any]]:
    if False:
        while True:
            i = 10
    'Check whether the given input is a numpy array or python sequence.'
    return _check_for_numpy(val) and isinstance(val, np.ndarray) or (isinstance(val, (pl.Series, Sequence) if include_series else Sequence) and (not isinstance(val, str)))

def is_str_sequence(val: object, *, allow_str: bool=False, include_series: bool=False) -> TypeGuard[Sequence[str]]:
    if False:
        i = 10
        return i + 15
    '\n    Check that `val` is a sequence of strings.\n\n    Note that a single string is a sequence of strings by definition, use\n    `allow_str=False` to return False on a single string.\n    '
    if allow_str is False and isinstance(val, str):
        return False
    elif _check_for_numpy(val) and isinstance(val, np.ndarray):
        return np.issubdtype(val.dtype, np.str_)
    elif include_series and isinstance(val, pl.Series):
        return val.dtype == pl.Utf8
    return isinstance(val, Sequence) and _is_iterable_of(val, str)

def range_to_series(name: str, rng: range, dtype: PolarsIntegerType | None=None) -> pl.Series:
    if False:
        for i in range(10):
            print('nop')
    'Fast conversion of the given range to a Series.'
    dtype = dtype or Int64
    return F.int_range(start=rng.start, end=rng.stop, step=rng.step, dtype=dtype, eager=True).alias(name)

def range_to_slice(rng: range) -> slice:
    if False:
        while True:
            i = 10
    'Return the given range as an equivalent slice.'
    return slice(rng.start, rng.stop, rng.step)

def handle_projection_columns(columns: Sequence[str] | Sequence[int] | str | None) -> tuple[list[int] | None, Sequence[str] | None]:
    if False:
        for i in range(10):
            print('nop')
    'Disambiguates between columns specified as integers vs. strings.'
    projection: list[int] | None = None
    new_columns: Sequence[str] | None = None
    if columns is not None:
        if isinstance(columns, str):
            new_columns = [columns]
        elif is_int_sequence(columns):
            projection = list(columns)
        elif not is_str_sequence(columns):
            raise TypeError('`columns` arg should contain a list of all integers or all strings values')
        else:
            new_columns = columns
        if columns and len(set(columns)) != len(columns):
            raise ValueError(f'`columns` arg should only have unique values, got {columns!r}')
        if projection and len(set(projection)) != len(projection):
            raise ValueError(f'`columns` arg should only have unique values, got {projection!r}')
    return (projection, new_columns)

def _prepare_row_count_args(row_count_name: str | None=None, row_count_offset: int=0) -> tuple[str, int] | None:
    if False:
        while True:
            i = 10
    if row_count_name is not None:
        return (row_count_name, row_count_offset)
    else:
        return None

def _in_notebook() -> bool:
    if False:
        while True:
            i = 10
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def arrlen(obj: Any) -> int | None:
    if False:
        return 10
    'Return length of (non-string) sequence object; returns None for non-sequences.'
    try:
        return None if isinstance(obj, str) else len(obj)
    except TypeError:
        return None

def can_create_dicts_with_pyarrow(dtypes: Sequence[PolarsDataType]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Check if the given dtypes can be used to create dicts with pyarrow fast path.'
    return _PYARROW_AVAILABLE and (not any((getattr(tp, 'time_unit', None) == 'ns' for tp in unpack_dtypes(*dtypes))))

def normalize_filepath(path: str | Path, *, check_not_directory: bool=True) -> str:
    if False:
        print('Hello World!')
    'Create a string path, expanding the home directory if present.'
    path = os.path.expanduser(path)
    if check_not_directory and os.path.exists(path) and os.path.isdir(path):
        raise IsADirectoryError(f'expected a file path; {path!r} is a directory')
    return path

def parse_version(version: Sequence[str | int]) -> tuple[int, ...]:
    if False:
        while True:
            i = 10
    'Simple version parser; split into a tuple of ints for comparison.'
    if isinstance(version, str):
        version = version.split('.')
    return tuple((int(re.sub('\\D', '', str(v))) for v in version))

def ordered_unique(values: Sequence[Any]) -> list[Any]:
    if False:
        while True:
            i = 10
    'Return unique list of sequence values, maintaining their order of appearance.'
    seen: set[Any] = set()
    add_ = seen.add
    return [v for v in values if not (v in seen or add_(v))]

def scale_bytes(sz: int, unit: SizeUnit) -> int | float:
    if False:
        while True:
            i = 10
    'Scale size in bytes to other size units (eg: "kb", "mb", "gb", "tb").'
    if unit in {'b', 'bytes'}:
        return sz
    elif unit in {'kb', 'kilobytes'}:
        return sz / 1024
    elif unit in {'mb', 'megabytes'}:
        return sz / 1024 ** 2
    elif unit in {'gb', 'gigabytes'}:
        return sz / 1024 ** 3
    elif unit in {'tb', 'terabytes'}:
        return sz / 1024 ** 4
    else:
        raise ValueError(f"`unit` must be one of {{'b', 'kb', 'mb', 'gb', 'tb'}}, got {unit!r}")

def _cast_repr_strings_with_schema(df: DataFrame, schema: dict[str, PolarsDataType | None]) -> DataFrame:
    if False:
        print('Hello World!')
    '\n    Utility function to cast table repr/string values into frame-native types.\n\n    Parameters\n    ----------\n    df\n        Dataframe containing string-repr column data.\n    schema\n        DataFrame schema containing the desired end-state types.\n\n    Notes\n    -----\n    Table repr strings are less strict (or different) than equivalent CSV data, so need\n    special handling; as this function is only used for reprs, parsing is flexible.\n\n    '
    tp: PolarsDataType | None
    if not df.is_empty():
        for tp in df.schema.values():
            if tp != Utf8:
                raise TypeError(f'DataFrame should contain only Utf8 string repr data; found {tp!r}')
    ns_sec = 1000000000
    duration_scaling = {'ns': 1, 'us': 1000, 'µs': 1000, 'ms': 1000000, 's': ns_sec, 'm': ns_sec * 60, 'h': ns_sec * 60 * 60, 'd': ns_sec * 3600 * 24, 'w': ns_sec * 3600 * 24 * 7}

    def str_duration_(td: str | None) -> int | None:
        if False:
            while True:
                i = 10
        return None if td is None else sum((int(value) * duration_scaling[unit.strip()] for (value, unit) in re.findall('(\\d+)(\\D+)', td)))
    cast_cols = {}
    for (c, tp) in schema.items():
        if tp is not None:
            if tp.base_type() == Datetime:
                tp_base = Datetime(tp.time_unit)
                d = F.col(c).str.replace('[A-Z ]+$', '')
                cast_cols[c] = F.when(d.str.len_bytes() == 19).then(d + '.000000000').otherwise(d + '000000000').str.slice(0, 29).str.strptime(tp_base, '%Y-%m-%d %H:%M:%S.%9f')
                if getattr(tp, 'time_zone', None) is not None:
                    cast_cols[c] = cast_cols[c].dt.replace_time_zone(tp.time_zone)
            elif tp == Date:
                cast_cols[c] = F.col(c).str.strptime(tp, '%Y-%m-%d')
            elif tp == Time:
                cast_cols[c] = F.when(F.col(c).str.len_bytes() == 8).then(F.col(c) + '.000000000').otherwise(F.col(c) + '000000000').str.slice(0, 18).str.strptime(tp, '%H:%M:%S.%9f')
            elif tp == Duration:
                cast_cols[c] = F.col(c).apply(str_duration_, return_dtype=Int64).cast(Duration('ns')).cast(tp)
            elif tp == Boolean:
                cast_cols[c] = F.col(c).map_dict(remapping={'true': True, 'false': False}, return_dtype=Boolean)
            elif tp in INTEGER_DTYPES:
                int_string = F.col(c).str.replace_all('[^\\d+-]', '')
                cast_cols[c] = pl.when(int_string.str.len_bytes() > 0).then(int_string).cast(tp)
            elif tp in FLOAT_DTYPES or tp.base_type() == Decimal:
                integer_part = F.col(c).str.replace('^(.*)\\D(\\d*)$', '$1')
                fractional_part = F.col(c).str.replace('^(.*)\\D(\\d*)$', '$2')
                cast_cols[c] = pl.when(F.col(c).str.contains('^[+-]?\\d*$')).then(pl.when(F.col(c).str.len_bytes() > 0).then(F.col(c))).when(F.col(c).str.contains('[eE]')).then(F.col(c).str.replace('[^eE\\d]', '.')).otherwise(pl.concat_str(integer_part.str.replace_all('[^\\d+-]', ''), fractional_part, separator='.')).cast(Utf8).cast(tp)
            elif tp != df.schema[c]:
                cast_cols[c] = F.col(c).cast(tp)
    return df.with_columns(**cast_cols) if cast_cols else df
NS = TypeVar('NS')

class sphinx_accessor(property):

    def __get__(self, instance: Any, cls: type[NS]) -> NS:
        if False:
            return 10
        try:
            return self.fget(instance if isinstance(instance, cls) else cls)
        except AttributeError:
            return None

class _NoDefault(Enum):
    no_default = 'NO_DEFAULT'

    def __repr__(self) -> str:
        if False:
            return 10
        return '<no_default>'
no_default = _NoDefault.no_default
NoDefault = Literal[_NoDefault.no_default]

def find_stacklevel() -> int:
    if False:
        i = 10
        return i + 15
    '\n    Find the first place in the stack that is not inside polars.\n\n    Taken from:\n    https://github.com/pandas-dev/pandas/blob/ab89c53f48df67709a533b6a95ce3d911871a0a8/pandas/util/_exceptions.py#L30-L51\n    '
    pkg_dir = str(Path(pl.__file__).parent)
    frame = inspect.currentframe()
    n = 0
    try:
        while frame:
            fname = inspect.getfile(frame)
            if fname.startswith(pkg_dir) or ((qualname := getattr(frame.f_code, 'co_qualname', None)) and qualname.startswith('singledispatch.')):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        del frame
    return n

def _get_stack_locals(of_type: type | tuple[type, ...] | None=None, n_objects: int | None=None, n_frames: int | None=None, named: str | tuple[str, ...] | None=None) -> dict[str, Any]:
    if False:
        return 10
    "\n    Retrieve f_locals from all (or the last 'n') stack frames from the calling location.\n\n    Parameters\n    ----------\n    of_type\n        Only return objects of this type.\n    n_objects\n        If specified, return only the most recent `n` matching objects.\n    n_frames\n        If specified, look at objects in the last `n` stack frames only.\n    named\n        If specified, only return objects matching the given name(s).\n\n    "
    if isinstance(named, str):
        named = (named,)
    objects = {}
    examined_frames = 0
    if n_frames is None:
        n_frames = sys.maxsize
    stack_frame = inspect.currentframe()
    stack_frame = getattr(stack_frame, 'f_back', None)
    try:
        while stack_frame and examined_frames < n_frames:
            local_items = list(stack_frame.f_locals.items())
            for (nm, obj) in reversed(local_items):
                if nm not in objects and (named is None or nm in named) and (of_type is None or isinstance(obj, of_type)):
                    objects[nm] = obj
                    if n_objects is not None and len(objects) >= n_objects:
                        return objects
            stack_frame = stack_frame.f_back
            examined_frames += 1
    finally:
        del stack_frame
    return objects

def _polars_warn(msg: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    warnings.warn(msg, stacklevel=find_stacklevel())

def in_terminal_that_supports_colour() -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine (within reason) if we are in an interactive terminal that supports color.\n\n    Note: this is not exhaustive, but it covers a lot (most?) of the common cases.\n    '
    if hasattr(sys.stdout, 'isatty'):
        return sys.stdout.isatty() and (sys.platform != 'win32' or 'ANSICON' in os.environ or 'WT_SESSION' in os.environ or (os.environ.get('TERM_PROGRAM') == 'vscode') or (os.environ.get('TERM') == 'xterm-256color')) or os.environ.get('PYCHARM_HOSTED') == '1'
    return False

def parse_percentiles(percentiles: Sequence[float] | float | None, *, inject_median: bool=False) -> Sequence[float]:
    if False:
        return 10
    '\n    Transforms raw percentiles into our preferred format, adding the 50th percentile.\n\n    Raises a ValueError if the percentile sequence is invalid\n    (e.g. outside the range [0, 1])\n    '
    if isinstance(percentiles, float):
        percentiles = [percentiles]
    elif percentiles is None:
        percentiles = []
    if not all((0 <= p <= 1 for p in percentiles)):
        raise ValueError('`percentiles` must all be in the range [0, 1]')
    sub_50_percentiles = sorted((p for p in percentiles if p < 0.5))
    at_or_above_50_percentiles = sorted((p for p in percentiles if p >= 0.5))
    if inject_median and (not at_or_above_50_percentiles or at_or_above_50_percentiles[0] != 0.5):
        at_or_above_50_percentiles = [0.5, *at_or_above_50_percentiles]
    return [*sub_50_percentiles, *at_or_above_50_percentiles]