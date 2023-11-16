from __future__ import annotations
from enum import Enum
from typing import Literal
import pandas as pd
from packaging.version import Version
from xarray.coding import cftime_offsets

def count_not_none(*args) -> int:
    if False:
        while True:
            i = 10
    'Compute the number of non-None arguments.\n\n    Copied from pandas.core.common.count_not_none (not part of the public API)\n    '
    return sum((arg is not None for arg in args))

class _NoDefault(Enum):
    """Used by pandas to specify a default value for a deprecated argument.
    Copied from pandas._libs.lib._NoDefault.

    See also:
    - pandas-dev/pandas#30788
    - pandas-dev/pandas#40684
    - pandas-dev/pandas#40715
    - pandas-dev/pandas#47045
    """
    no_default = 'NO_DEFAULT'

    def __repr__(self) -> str:
        if False:
            return 10
        return '<no_default>'
no_default = _NoDefault.no_default
NoDefault = Literal[_NoDefault.no_default]

def _convert_base_to_offset(base, freq, index):
    if False:
        return 10
    'Required until we officially deprecate the base argument to resample.  This\n    translates a provided `base` argument to an `offset` argument, following logic\n    from pandas.\n    '
    from xarray.coding.cftimeindex import CFTimeIndex
    if isinstance(index, pd.DatetimeIndex):
        freq = pd.tseries.frequencies.to_offset(freq)
        if isinstance(freq, pd.offsets.Tick):
            return pd.Timedelta(base * freq.nanos // freq.n)
    elif isinstance(index, CFTimeIndex):
        freq = cftime_offsets.to_offset(freq)
        if isinstance(freq, cftime_offsets.Tick):
            return base * freq.as_timedelta() // freq.n
    else:
        raise ValueError('Can only resample using a DatetimeIndex or CFTimeIndex.')

def nanosecond_precision_timestamp(*args, **kwargs) -> pd.Timestamp:
    if False:
        i = 10
        return i + 15
    'Return a nanosecond-precision Timestamp object.\n\n    Note this function should no longer be needed after addressing GitHub issue\n    #7493.\n    '
    if Version(pd.__version__) >= Version('2.0.0'):
        return pd.Timestamp(*args, **kwargs).as_unit('ns')
    else:
        return pd.Timestamp(*args, **kwargs)