from __future__ import annotations
import typing
from itertools import chain
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError
from ..iapi import range_view
from ..utils import alias, array_kind, match
from ._expand import expand_range
from .range import RangeContinuous
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
if typing.TYPE_CHECKING:
    from typing import Sequence
    from plotnine.typing import Trans, TupleFloat2, TupleFloat4

@document
class scale_position_discrete(scale_discrete):
    """
    Base class for discrete position scales

    Parameters
    ----------
    {superclass_parameters}
    limits : array_like, optional
        Limits of the scale. For discrete scale, these are
        the categories (unique values) of the variable.
        For scales that deal with categoricals, these may
        be a subset or superset of the categories.
    """
    guide = None
    range_c: RangeContinuous

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.range_c = RangeContinuous()
        scale_discrete.__init__(self, *args, **kwargs)

    def reset(self):
        if False:
            while True:
                i = 10
        self.range_c.reset()

    def is_empty(self) -> bool:
        if False:
            while True:
                i = 10
        return super().is_empty() and self.range_c.is_empty()

    def train(self, series):
        if False:
            i = 10
            return i + 15
        if array_kind.continuous(series):
            self.range_c.train(series)
        else:
            self.range.train(series, drop=self.drop)

    def map(self, series, limits=None):
        if False:
            for i in range(10):
                print('nop')
        if limits is None:
            limits = self.limits
        if array_kind.discrete(series):
            seq = np.arange(1, len(limits) + 1)
            idx = np.asarray(match(series, limits, nomatch=len(series)))
            if not len(idx):
                return np.array([])
            try:
                seq = seq[idx]
            except IndexError:
                seq = np.hstack((seq.astype(float), np.nan))
                idx = np.clip(idx, 0, len(seq) - 1)
                seq = seq[idx]
            return seq
        return series

    @property
    def limits(self):
        if False:
            return 10
        if self.is_empty():
            return (0, 1)
        elif self._limits is not None and (not callable(self._limits)):
            return self._limits
        elif self._limits is None:
            return self.range.range
        elif callable(self._limits):
            limits = self._limits(self.range.range)
            if iter(limits) is limits:
                limits = list(limits)
            return limits
        else:
            raise PlotnineError('Lost, do not know what the limits are.')

    @limits.setter
    def limits(self, value):
        if False:
            while True:
                i = 10
        if isinstance(value, tuple):
            value = list(value)
        self._limits = value

    def dimension(self, expand=(0, 0, 0, 0), limits=None):
        if False:
            return 10
        '\n        Get the phyical size of the scale\n\n        Unlike limits, this always returns a numeric vector of length 2\n        '
        from mizani.bounds import expand_range_distinct
        if limits is None:
            limits = self.limits
        if self.is_empty():
            return (0, 1)
        if self.range.is_empty():
            return expand_range_distinct(self.range_c.range, expand)
        elif self.range_c.is_empty():
            return expand_range_distinct((1, len(self.limits)), expand)
        else:
            a = np.hstack([self.range_c.range, expand_range_distinct((1, len(self.range.range)), expand)])
            return (a.min(), a.max())

    def expand_limits(self, limits: Sequence[str], expand: TupleFloat2 | TupleFloat4, coord_limits: TupleFloat2, trans: Trans) -> range_view:
        if False:
            for i in range(10):
                print('nop')
        if self.is_empty():
            climits = (0, 1)
        else:
            climits = (1, len(limits))
            self.range_c.range
        if coord_limits is not None:
            (c0, c1) = coord_limits
            climits = (climits[0] if c0 is None else c0, climits[1] if c1 is None else c1)
        rv_d = expand_range(climits, expand, trans)
        if self.range_c.is_empty():
            return rv_d
        no_expand = self.default_expansion(0, 0)
        rv_c = expand_range(self.range_c.range, no_expand, trans)
        rv = range_view(range=(min(chain(rv_d.range, rv_c.range)), max(chain(rv_d.range, rv_c.range))), range_coord=(min(chain(rv_d.range_coord, rv_c.range_coord)), max(chain(rv_d.range_coord, rv_c.range_coord))))
        rv.range = (min(rv.range), max(rv.range))
        rv.range_coord = (min(rv.range_coord), max(rv.range_coord))
        return rv

@document
class scale_position_continuous(scale_continuous):
    """
    Base class for continuous position scales

    Parameters
    ----------
    {superclass_parameters}
    """
    guide = None

    def map(self, series, limits=None):
        if False:
            while True:
                i = 10
        if not len(series):
            return series
        if limits is None:
            limits = self.limits
        scaled = self.oob(series, limits)
        scaled[pd.isna(scaled)] = self.na_value
        return scaled

@document
class scale_x_discrete(scale_position_discrete):
    """
    Discrete x position

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['x', 'xmin', 'xmax', 'xend']

@document
class scale_y_discrete(scale_position_discrete):
    """
    Discrete y position

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['y', 'ymin', 'ymax', 'yend']
alias('scale_x_ordinal', scale_x_discrete)
alias('scale_y_ordinal', scale_y_discrete)

@document
class scale_x_continuous(scale_position_continuous):
    """
    Continuous x position

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['x', 'xmin', 'xmax', 'xend', 'xintercept']

@document
class scale_y_continuous(scale_position_continuous):
    """
    Continuous y position

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['y', 'ymin', 'ymax', 'yend', 'yintercept', 'ymin_final', 'ymax_final', 'lower', 'middle', 'upper']

@document
class scale_x_datetime(scale_datetime, scale_x_continuous):
    """
    Continuous x position for datetime data points

    Parameters
    ----------
    {superclass_parameters}
    """

@document
class scale_y_datetime(scale_datetime, scale_y_continuous):
    """
    Continuous y position for datetime data points

    Parameters
    ----------
    {superclass_parameters}
    """
alias('scale_x_date', scale_x_datetime)
alias('scale_y_date', scale_y_datetime)

@document
class scale_x_timedelta(scale_x_continuous):
    """
    Continuous x position for timedelta data points

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'pd_timedelta'

@document
class scale_y_timedelta(scale_y_continuous):
    """
    Continuous y position for timedelta data points

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'pd_timedelta'

@document
class scale_x_sqrt(scale_x_continuous):
    """
    Continuous x position sqrt transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'sqrt'

@document
class scale_y_sqrt(scale_y_continuous):
    """
    Continuous y position sqrt transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'sqrt'

@document
class scale_x_log10(scale_x_continuous):
    """
    Continuous x position log10 transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'log10'

@document
class scale_y_log10(scale_y_continuous):
    """
    Continuous y position log10 transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'log10'

@document
class scale_x_reverse(scale_x_continuous):
    """
    Continuous x position reverse transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'reverse'

@document
class scale_y_reverse(scale_y_continuous):
    """
    Continuous y position reverse transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'reverse'

@document
class scale_x_symlog(scale_x_continuous):
    """
    Continuous x position symmetric logarithm transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'symlog'

@document
class scale_y_symlog(scale_y_continuous):
    """
    Continuous y position symmetric logarithm transformed scale

    Parameters
    ----------
    {superclass_parameters}
    """
    _trans = 'symlog'