from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from mizani.bounds import censor, expand_range_distinct, rescale, zero_range
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..iapi import range_view, scale_view
from ..utils import match
from ._expand import expand_range
from .range import RangeContinuous
from .scale import scale
if typing.TYPE_CHECKING:
    from typing import Any, Optional, Sequence, Type
    from plotnine.typing import AnyArrayLike, CoordRange, FloatArrayLike, ScaleContinuousBreaks, ScaleContinuousBreaksRaw, ScaleContinuousLimits, ScaleContinuousLimitsRaw, ScaleLabels, ScaleMinorBreaksRaw, TFloatArrayLike, Trans, TupleFloat2, TupleFloat4

@document
class scale_continuous(scale):
    """
    Base class for all continuous scales

    Parameters
    ----------
    {superclass_parameters}
    trans : str | function
        Name of a trans function or a trans function.
        See :mod:`mizani.transforms` for possible options.
    oob : function
        Function to deal with out of bounds (limits)
        data points. Default is to turn them into
        ``np.nan``, which then get dropped.
    minor_breaks : list-like or int or callable or None
        If a list-like, it is the minor breaks points.
        If an integer, it is the number of minor breaks between
        any set of major breaks.
        If a function, it should have the signature
        ``func(limits)`` and return a list-like of consisting
        of the minor break points.
        If ``None``, no minor breaks are calculated.
        The default is to automatically calculate them.
    rescaler : function, optional
        Function to rescale data points so that they can
        be handled by the palette. Default is to rescale
        them onto the [0, 1] range. Scales that inherit
        from this class may have another default.

    Notes
    -----
    If using the class directly all arguments must be
    keyword arguments.
    """
    _range_class = RangeContinuous
    _limits: Optional[ScaleContinuousLimitsRaw]
    rescaler = staticmethod(rescale)
    oob = staticmethod(censor)
    breaks: ScaleContinuousBreaksRaw
    minor_breaks: ScaleMinorBreaksRaw = True
    _trans: Trans | str = 'identity'

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.trans = kwargs.pop('trans', self._trans)
        with suppress(KeyError):
            self.limits = kwargs.pop('limits')
        scale.__init__(self, **kwargs)

    def _check_trans(self, t):
        if False:
            return 10
        '\n        Check if transform t is valid\n\n        When scales specialise on a specific transform (other than\n        the identity transform), the user should know when they\n        try to change the transform.\n\n        Parameters\n        ----------\n        t : mizani.transforms.trans\n            Transform object\n        '
        orig_trans_name = self.__class__._trans
        new_trans_name = t.__class__.__name__
        if new_trans_name.endswith('_trans'):
            new_trans_name = new_trans_name[:-6]
        if orig_trans_name != 'identity':
            if new_trans_name != orig_trans_name:
                warn('You have changed the transform of a specialised scale. The result may not be what you expect.\nOriginal transform: {}\nNew transform: {}'.format(orig_trans_name, new_trans_name), PlotnineWarning)

    @property
    def trans(self) -> Trans:
        if False:
            for i in range(10):
                print('nop')
        return self._trans

    @trans.setter
    def trans(self, value: Trans | str | Type[Trans]):
        if False:
            for i in range(10):
                print('nop')
        from mizani.transforms import gettrans
        t: Trans = gettrans(value)
        self._check_trans(t)
        self._trans = t

    @property
    def limits(self):
        if False:
            print('Hello World!')
        if self.is_empty():
            return (0, 1)
        if self._limits is None:
            return self.range.range
        elif callable(self._limits):
            _range = self.inverse(self.range.range)
            return tuple(self.trans.transform(self._limits(_range)))
        elif self._limits is not None and (not self.range.is_empty()):
            if len(self._limits) == len(self.range.range):
                (l1, l2) = self._limits
                (r1, r2) = self.range.range
                if l1 is None:
                    l1 = self.trans.transform([r1])[0]
                if l2 is None:
                    l2 = self.trans.transform([r2])[0]
                return (l1, l2)
        return tuple(self._limits)

    @limits.setter
    def limits(self, value: ScaleContinuousLimitsRaw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Limits for the continuous scale\n\n        Parameters\n        ----------\n        value : array-like | callable\n            Limits in the dataspace.\n        '
        if isinstance(value, bool) or value is None or callable(value):
            self._limits = value
            return
        (a, b) = value
        a = self.transform([a])[0] if a is not None else a
        b = self.transform([b])[0] if b is not None else b
        if a is not None and b is not None and (a > b):
            (a, b) = (b, a)
        self._limits = (a, b)

    def train(self, x):
        if False:
            print('Hello World!')
        '\n        Train scale\n\n        Parameters\n        ----------\n        x: pd.Series | np.array\n            A column of data to train over\n        '
        if not len(x):
            return
        self.range.train(x)

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        '\n        Transform dataframe\n        '
        if len(df) == 0:
            return df
        aesthetics = set(self.aesthetics) & set(df.columns)
        for ae in aesthetics:
            with suppress(TypeError):
                df[ae] = self.transform(df[ae])
        return df

    def transform(self, x: TFloatArrayLike) -> TFloatArrayLike:
        if False:
            while True:
                i = 10
        '\n        Transform array|series x\n        '
        try:
            return self.trans.transform(x)
        except TypeError:
            return [self.trans.transform(val) for val in x]

    def inverse_df(self, df):
        if False:
            while True:
                i = 10
        '\n        Inverse Transform dataframe\n        '
        if len(df) == 0:
            return
        aesthetics = set(self.aesthetics) & set(df.columns)
        for ae in aesthetics:
            with suppress(TypeError):
                df[ae] = self.inverse(df[ae])
        return df

    def inverse(self, x: TFloatArrayLike) -> TFloatArrayLike:
        if False:
            return 10
        '\n        Inverse transform array|series x\n        '
        try:
            return self.trans.inverse(x)
        except TypeError:
            return [self.trans.inverse(val) for val in x]

    @property
    def is_linear(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Return True if the scale is linear\n\n        Depends on the transformation.\n        '
        return self.trans.transform_is_linear

    def dimension(self, expand=(0, 0, 0, 0), limits=None):
        if False:
            return 10
        '\n        Get the phyical size of the scale\n\n        Unlike limits, this always returns a numeric vector of length 2\n        '
        if limits is None:
            limits = self.limits
        return expand_range_distinct(limits, expand)

    def expand_limits(self, limits: ScaleContinuousLimits, expand: TupleFloat2 | TupleFloat4, coord_limits: CoordRange | None, trans: Trans) -> range_view:
        if False:
            return 10
        '\n        Calculate the final range in coordinate space\n        '
        if coord_limits is not None:
            (c0, c1) = coord_limits
            limits = (limits[0] if c0 is None else c0, limits[1] if c1 is None else c1)
        return expand_range(limits, expand, trans)

    def view(self, limits: Optional[CoordRange]=None, range: Optional[CoordRange]=None) -> scale_view:
        if False:
            for i in range(10):
                print('nop')
        '\n        Information about the trained scale\n        '
        if limits is None:
            limits = self.limits
        if range is None:
            range = self.dimension(limits=limits)
        breaks = self.get_bounded_breaks(range)
        labels = self.get_labels(breaks)
        ubreaks = self.get_breaks(range)
        minor_breaks = self.get_minor_breaks(ubreaks, range)
        sv = scale_view(scale=self, aesthetics=self.aesthetics, name=self.name, limits=limits, range=range, breaks=breaks, labels=labels, minor_breaks=minor_breaks)
        return sv

    def default_expansion(self, mult=0.05, add=0, expand=True):
        if False:
            print('Hello World!')
        '\n        Get the default expansion for continuous scale\n        '
        return super().default_expansion(mult, add, expand)

    @staticmethod
    def palette(arr: FloatArrayLike) -> Sequence[Any]:
        if False:
            return 10
        '\n        Aesthetic mapping function\n        '
        raise NotImplementedError('Not Implemented')

    def map(self, x: AnyArrayLike, limits: Optional[ScaleContinuousLimits]=None) -> AnyArrayLike:
        if False:
            while True:
                i = 10
        if limits is None:
            limits = self.limits
        x = self.oob(self.rescaler(x, _from=limits))
        uniq = np.unique(x)
        pal = np.asarray(self.palette(uniq))
        scaled = pal[match(x, uniq)]
        if scaled.dtype.kind == 'U':
            scaled = [self.na_value if x == 'nan' else x for x in scaled]
        else:
            scaled[pd.isna(scaled)] = self.na_value
        return scaled

    def get_breaks(self, limits: Optional[ScaleContinuousLimits]=None) -> ScaleContinuousBreaks:
        if False:
            return 10
        '\n        Generate breaks for the axis or legend\n\n        Parameters\n        ----------\n        limits : list-like | None\n            If None the self.limits are used\n            They are expected to be in transformed\n            space.\n\n        Returns\n        -------\n        out : array-like\n\n        Notes\n        -----\n        Breaks are calculated in data space and\n        returned in transformed space since all\n        data is plotted in transformed space.\n        '
        if limits is None:
            limits = self.limits
        _limits = self.inverse(limits)
        if self.is_empty() or self.breaks is False or self.breaks is None:
            breaks = []
        elif self.breaks is True:
            _tlimits = self.trans.breaks(_limits)
            breaks: ScaleContinuousBreaks = _tlimits
        elif zero_range(_limits):
            breaks = [_limits[0]]
        elif callable(self.breaks):
            breaks = self.breaks(_limits)
        else:
            breaks = self.breaks
        breaks = self.transform(breaks)
        return breaks

    def get_bounded_breaks(self, limits: Optional[ScaleContinuousLimits]=None) -> ScaleContinuousBreaks:
        if False:
            print('Hello World!')
        '\n        Return Breaks that are within limits\n        '
        if limits is None:
            limits = self.limits
        breaks = self.get_breaks(limits)
        strict_breaks = [b for b in breaks if limits[0] <= b <= limits[1]]
        return strict_breaks

    def get_minor_breaks(self, major: ScaleContinuousBreaks, limits: Optional[ScaleContinuousLimits]=None) -> ScaleContinuousBreaks:
        if False:
            print('Hello World!')
        '\n        Return minor breaks\n        '
        if limits is None:
            limits = self.limits
        if self.minor_breaks is False or self.minor_breaks is None:
            minor_breaks = []
        elif self.minor_breaks is True:
            minor_breaks: ScaleContinuousBreaks = self.trans.minor_breaks(major, limits)
        elif isinstance(self.minor_breaks, int):
            minor_breaks: ScaleContinuousBreaks = self.trans.minor_breaks(major, limits, self.minor_breaks)
        elif callable(self.minor_breaks):
            breaks = self.minor_breaks(self.inverse(limits))
            _major = set(major)
            minor = self.transform(breaks)
            minor_breaks = [x for x in minor if x not in _major]
        else:
            minor_breaks = self.transform(self.minor_breaks)
        return minor_breaks

    def get_labels(self, breaks: Optional[ScaleContinuousBreaks]=None) -> ScaleLabels:
        if False:
            print('Hello World!')
        '\n        Generate labels for the axis or legend\n\n        Parameters\n        ----------\n        breaks: None or array-like\n            If None, use self.breaks.\n        '
        if breaks is None:
            breaks = self.get_breaks()
        breaks = self.inverse(breaks)
        labels: Sequence[str]
        if self.labels is False or self.labels is None:
            labels = []
        elif self.labels is True:
            labels = self.trans.format(breaks)
        elif callable(self.labels):
            labels = self.labels(breaks)
        elif isinstance(self.labels, dict):
            labels = [str(self.labels[b]) if b in self.labels else str(b) for b in breaks]
        else:
            from collections.abc import Sized
            labels = self.labels
            if len(labels) != len(breaks) and isinstance(self.breaks, Sized) and (len(labels) == len(self.breaks)):
                _wanted_breaks = set(breaks)
                labels = [l for (l, b) in zip(labels, self.breaks) if b in _wanted_breaks]
        if len(labels) != len(breaks):
            raise PlotnineError('Breaks and labels are different lengths')
        return labels