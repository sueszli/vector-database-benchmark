"""
Helper functions to generate range-like data for DatetimeArray
(and possibly TimedeltaArray/PeriodArray)
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.lib import i8max
from pandas._libs.tslibs import BaseOffset, OutOfBoundsDatetime, Timedelta, Timestamp, iNaT
if TYPE_CHECKING:
    from pandas._typing import npt

def generate_regular_range(start: Timestamp | Timedelta | None, end: Timestamp | Timedelta | None, periods: int | None, freq: BaseOffset, unit: str='ns') -> npt.NDArray[np.intp]:
    if False:
        i = 10
        return i + 15
    '\n    Generate a range of dates or timestamps with the spans between dates\n    described by the given `freq` DateOffset.\n\n    Parameters\n    ----------\n    start : Timedelta, Timestamp or None\n        First point of produced date range.\n    end : Timedelta, Timestamp or None\n        Last point of produced date range.\n    periods : int or None\n        Number of periods in produced date range.\n    freq : Tick\n        Describes space between dates in produced date range.\n    unit : str, default "ns"\n        The resolution the output is meant to represent.\n\n    Returns\n    -------\n    ndarray[np.int64]\n        Representing the given resolution.\n    '
    istart = start._value if start is not None else None
    iend = end._value if end is not None else None
    freq.nanos
    td = Timedelta(freq)
    b: int | np.int64 | np.uint64
    e: int | np.int64 | np.uint64
    try:
        td = td.as_unit(unit, round_ok=False)
    except ValueError as err:
        raise ValueError(f'freq={freq} is incompatible with unit={unit}. Use a lower freq or a higher unit instead.') from err
    stride = int(td._value)
    if periods is None and istart is not None and (iend is not None):
        b = istart
        e = b + (iend - b) // stride * stride + stride // 2 + 1
    elif istart is not None and periods is not None:
        b = istart
        e = _generate_range_overflow_safe(b, periods, stride, side='start')
    elif iend is not None and periods is not None:
        e = iend + stride
        b = _generate_range_overflow_safe(e, periods, stride, side='end')
    else:
        raise ValueError("at least 'start' or 'end' should be specified if a 'period' is given.")
    with np.errstate(over='raise'):
        try:
            values = np.arange(b, e, stride, dtype=np.int64)
        except FloatingPointError:
            xdr = [b]
            while xdr[-1] != e:
                xdr.append(xdr[-1] + stride)
            values = np.array(xdr[:-1], dtype=np.int64)
    return values

def _generate_range_overflow_safe(endpoint: int, periods: int, stride: int, side: str='start') -> np.int64 | np.uint64:
    if False:
        return 10
    "\n    Calculate the second endpoint for passing to np.arange, checking\n    to avoid an integer overflow.  Catch OverflowError and re-raise\n    as OutOfBoundsDatetime.\n\n    Parameters\n    ----------\n    endpoint : int\n        nanosecond timestamp of the known endpoint of the desired range\n    periods : int\n        number of periods in the desired range\n    stride : int\n        nanoseconds between periods in the desired range\n    side : {'start', 'end'}\n        which end of the range `endpoint` refers to\n\n    Returns\n    -------\n    other_end : np.int64 | np.uint64\n\n    Raises\n    ------\n    OutOfBoundsDatetime\n    "
    assert side in ['start', 'end']
    i64max = np.uint64(i8max)
    msg = f'Cannot generate range with {side}={endpoint} and periods={periods}'
    with np.errstate(over='raise'):
        try:
            addend = np.uint64(periods) * np.uint64(np.abs(stride))
        except FloatingPointError as err:
            raise OutOfBoundsDatetime(msg) from err
    if np.abs(addend) <= i64max:
        return _generate_range_overflow_safe_signed(endpoint, periods, stride, side)
    elif endpoint > 0 and side == 'start' and (stride > 0) or (endpoint < 0 < stride and side == 'end'):
        raise OutOfBoundsDatetime(msg)
    elif side == 'end' and endpoint - stride <= i64max < endpoint:
        return _generate_range_overflow_safe(endpoint - stride, periods - 1, stride, side)
    mid_periods = periods // 2
    remaining = periods - mid_periods
    assert 0 < remaining < periods, (remaining, periods, endpoint, stride)
    midpoint = int(_generate_range_overflow_safe(endpoint, mid_periods, stride, side))
    return _generate_range_overflow_safe(midpoint, remaining, stride, side)

def _generate_range_overflow_safe_signed(endpoint: int, periods: int, stride: int, side: str) -> np.int64 | np.uint64:
    if False:
        return 10
    '\n    A special case for _generate_range_overflow_safe where `periods * stride`\n    can be calculated without overflowing int64 bounds.\n    '
    assert side in ['start', 'end']
    if side == 'end':
        stride *= -1
    with np.errstate(over='raise'):
        addend = np.int64(periods) * np.int64(stride)
        try:
            result = np.int64(endpoint) + addend
            if result == iNaT:
                raise OverflowError
            return result
        except (FloatingPointError, OverflowError):
            pass
        assert stride > 0 and endpoint >= 0 or (stride < 0 and endpoint <= 0)
        if stride > 0:
            uresult = np.uint64(endpoint) + np.uint64(addend)
            i64max = np.uint64(i8max)
            assert uresult > i64max
            if uresult <= i64max + np.uint64(stride):
                return uresult
    raise OutOfBoundsDatetime(f'Cannot generate range with {side}={endpoint} and periods={periods}')