from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from mizani.bounds import expand_range_distinct
from ..doctools import document
from ..iapi import range_view, scale_view
from ..utils import match
from ._expand import expand_range
from .range import RangeDiscrete
from .scale import scale
if typing.TYPE_CHECKING:
    from typing import Any, Optional, Sequence
    from plotnine.typing import CoordRange, ScaleDiscreteBreaks, ScaleDiscreteBreaksRaw, ScaleDiscreteLimits, ScaleDiscreteLimitsRaw, ScaleLabels, Trans, TupleFloat2, TupleFloat4

@document
class scale_discrete(scale):
    """
    Base class for all discrete scales

    Parameters
    ----------
    {superclass_parameters}
    limits : array_like, optional
        Limits of the scale. For scales that deal with
        categoricals, these may be a subset or superset of
        the categories. Data values that are not in the limits
        will be treated as missing data and represented with
        the ``na_value``.
    drop : bool
        Whether to drop unused categories from
        the scale
    na_translate : bool
        If ``True`` translate missing values and show them.
        If ``False`` remove missing values. Default value is
        ``True``
    na_value : object
        If ``na_translate=True``, what aesthetic value should be
        assigned to the missing values. This parameter does not
        apply to position scales where ``nan`` is always placed
        on the right.
    """
    _range_class = RangeDiscrete
    _limits: Optional[ScaleDiscreteLimitsRaw]
    range: RangeDiscrete
    breaks: ScaleDiscreteBreaksRaw
    drop: bool = True
    na_translate: bool = True

    @property
    def limits(self) -> ScaleDiscreteLimits:
        if False:
            while True:
                i = 10
        if self.is_empty():
            return ('0', '1')
        if self._limits is None:
            return tuple(self.range.range)
        elif callable(self._limits):
            return tuple(self._limits(self.range.range))
        else:
            return tuple(self._limits)

    @limits.setter
    def limits(self, value: ScaleDiscreteLimitsRaw):
        if False:
            return 10
        self._limits = value

    @staticmethod
    def palette(n: int) -> Sequence[Any]:
        if False:
            return 10
        '\n        Aesthetic mapping function\n        '
        raise NotImplementedError('Not Implemented')

    def train(self, x, drop=False):
        if False:
            return 10
        '\n        Train scale\n\n        Parameters\n        ----------\n        x: pd.Series | np.array\n            A column of data to train over\n        drop : bool\n            Whether to drop(not include) unused categories\n\n        A discrete range is stored in a list\n        '
        if not len(x):
            return
        na_rm = not self.na_translate
        self.range.train(x, drop, na_rm=na_rm)

    def dimension(self, expand=(0, 0, 0, 0), limits=None):
        if False:
            while True:
                i = 10
        '\n        Get the phyical size of the scale\n\n        Unlike limits, this always returns a numeric vector of length 2\n        '
        if limits is None:
            limits = self.limits
        return expand_range_distinct((0, len(limits)), expand)

    def expand_limits(self, limits: ScaleDiscreteLimits, expand: TupleFloat2 | TupleFloat4, coord_limits: TupleFloat2, trans: Trans) -> range_view:
        if False:
            i = 10
            return i + 15
        '\n        Calculate the final range in coordinate space\n        '
        is_empty = self.is_empty() or len(limits) == 0
        climits = (0, 1) if is_empty else (1, len(limits))
        if coord_limits is not None:
            (c0, c1) = coord_limits
            climits = (climits[0] if c0 is None else c0, climits[1] if c1 is None else c1)
        return expand_range(climits, expand, trans)

    def view(self, limits: Optional[ScaleDiscreteLimits]=None, range: Optional[CoordRange]=None) -> scale_view:
        if False:
            for i in range(10):
                print('nop')
        '\n        Information about the trained scale\n        '
        if limits is None:
            limits = self.limits
        if range is None:
            range = self.dimension(limits=limits)
        breaks_d = self.get_breaks(limits)
        breaks = self.map(pd.Categorical(breaks_d))
        minor_breaks = []
        labels = self.get_labels(breaks_d)
        sv = scale_view(scale=self, aesthetics=self.aesthetics, name=self.name, limits=limits, range=range, breaks=breaks, labels=labels, minor_breaks=minor_breaks)
        return sv

    def default_expansion(self, mult=0, add=0.6, expand=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the default expansion for a discrete scale\n        '
        return super().default_expansion(mult, add, expand)

    def map(self, x, limits: Optional[ScaleDiscreteLimits]=None) -> Sequence[Any]:
        if False:
            return 10
        '\n        Map values in x to a palette\n        '
        if limits is None:
            limits = self.limits
        n = sum(~pd.isna(list(limits)))
        pal = self.palette(n)
        if isinstance(pal, dict):
            pal_match = []
            for val in x:
                try:
                    pal_match.append(pal[val])
                except KeyError:
                    pal_match.append(self.na_value)
        else:
            if not isinstance(pal, np.ndarray):
                pal = np.asarray(pal, dtype=object)
            idx = np.asarray(match(x, limits))
            try:
                pal_match = [pal[i] if i >= 0 else None for i in idx]
            except IndexError:
                pal = np.hstack((pal.astype(object), np.nan))
                idx = np.clip(idx, 0, len(pal) - 1)
                pal_match = list(pal[idx])
        if self.na_translate:
            bool_pal_match = pd.isna(pal_match)
            if len(bool_pal_match.shape) > 1:
                bool_pal_match = bool_pal_match.any(axis=1)
            bool_idx = pd.isna(x) | bool_pal_match
            if bool_idx.any():
                pal_match = [x if i else self.na_value for (x, i) in zip(pal_match, ~bool_idx)]
        return pal_match

    def get_breaks(self, limits: Optional[ScaleDiscreteLimits]=None) -> ScaleDiscreteBreaks:
        if False:
            print('Hello World!')
        "\n        Return an ordered list of breaks\n\n        The form is suitable for use by the guides e.g.\n            ['fair', 'good', 'very good', 'premium', 'ideal']\n        "
        if limits is None:
            limits = self.limits
        if self.is_empty():
            return []
        if self.breaks is True:
            breaks = list(limits)
        elif self.breaks in (False, None):
            breaks = []
        elif callable(self.breaks):
            breaks = self.breaks(limits)
        else:
            _wanted_breaks = set(self.breaks)
            breaks = [l for l in limits if l in _wanted_breaks]
        return breaks

    def get_bounded_breaks(self, limits: Optional[ScaleDiscreteLimits]=None) -> ScaleDiscreteBreaks:
        if False:
            return 10
        '\n        Return Breaks that are within limits\n        '
        if limits is None:
            limits = self.limits
        lookup = set(limits)
        breaks = self.get_breaks()
        strict_breaks = [b for b in breaks if b in lookup]
        return strict_breaks

    def get_labels(self, breaks: Optional[ScaleDiscreteBreaks]=None) -> ScaleLabels:
        if False:
            while True:
                i = 10
        '\n        Generate labels for the legend/guide breaks\n        '
        if self.is_empty():
            return []
        if breaks is None:
            breaks = self.get_breaks()
        if breaks in (None, False) or self.labels in (None, False):
            return []
        elif self.labels is True:
            return [str(b) for b in breaks]
        elif callable(self.labels):
            return self.labels(breaks)
        elif isinstance(self.labels, dict):
            labels = [str(self.labels[b]) if b in self.labels else str(b) for b in breaks]
            return labels
        else:
            return self.labels

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform dataframe\n        '
        return df

    def transform(self, x):
        if False:
            while True:
                i = 10
        '\n        Transform array|series x\n        '
        return x

    def inverse_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        '\n        Inverse Transform dataframe\n        '
        return df