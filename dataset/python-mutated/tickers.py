""" Models for computing good tick locations on different kinds
of plots.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import Auto, Either, Enum, Float, Instance, Int, Nullable, Override, Required, Seq
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
__all__ = ('Ticker', 'BinnedTicker', 'ContinuousTicker', 'FixedTicker', 'AdaptiveTicker', 'CompositeTicker', 'SingleIntervalTicker', 'DaysTicker', 'MonthsTicker', 'YearsTicker', 'BasicTicker', 'LogTicker', 'MercatorTicker', 'CategoricalTicker', 'DatetimeTicker')

@abstract
class Ticker(Model):
    """ A base class for all ticker types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

@abstract
class ContinuousTicker(Ticker):
    """ A base class for non-categorical ticker types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    num_minor_ticks = Int(5, help='\n    The number of minor tick positions to generate between\n    adjacent major tick values.\n    ')
    desired_num_ticks = Int(6, help='\n    A desired target number of major tick positions to generate across\n    the plot range.\n\n    .. note:\n        This value is a suggestion, and ticker subclasses may ignore\n        it entirely, or use it only as an ideal goal to approach as well\n        as can be, in the context of a specific ticking strategy.\n    ')

class FixedTicker(ContinuousTicker):
    """ Generate ticks at fixed, explicitly supplied locations.

    .. note::
        The ``desired_num_ticks`` property is ignored by this Ticker.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    ticks = Seq(Float, default=[], help='\n    List of major tick locations.\n    ')
    minor_ticks = Seq(Float, default=[], help='\n    List of minor tick locations.\n    ')

class AdaptiveTicker(ContinuousTicker):
    """ Generate "nice" round ticks at any magnitude.

    Creates ticks that are "base" multiples of a set of given
    mantissas. For example, with ``base=10`` and
    ``mantissas=[1, 2, 5]``, the ticker will generate the sequence::

        ..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, ...

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    base = Float(10.0, help='\n    The multiplier to use for scaling mantissas.\n    ')
    mantissas = Seq(Float, default=[1, 2, 5], help='\n    The acceptable list numbers to generate multiples of.\n    ')
    min_interval = Float(0.0, help='\n    The smallest allowable interval between two adjacent ticks.\n    ')
    max_interval = Nullable(Float, help='\n    The largest allowable interval between two adjacent ticks.\n\n    .. note::\n        To specify an unbounded interval, set to ``None``.\n    ')

class CompositeTicker(ContinuousTicker):
    """ Combine different tickers at different scales.

    Uses the ``min_interval`` and ``max_interval`` interval attributes
    of the tickers to select the appropriate ticker at different
    scales.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    tickers = Seq(Instance(Ticker), default=[], help='\n    A list of Ticker objects to combine at different scales in order\n    to generate tick values. The supplied tickers should be in order.\n    Specifically, if S comes before T, then it should be the case that::\n\n        S.get_max_interval() < T.get_min_interval()\n\n    ')

class BaseSingleIntervalTicker(ContinuousTicker):
    """ Base class for single interval tickers. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class SingleIntervalTicker(BaseSingleIntervalTicker):
    """ Generate evenly spaced ticks at a fixed interval regardless of
    scale.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    interval = Required(Float, help='\n    The interval between adjacent ticks.\n    ')

class DaysTicker(BaseSingleIntervalTicker):
    """ Generate ticks spaced apart by specific, even multiples of days.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    days = Seq(Int, default=[], help='\n    The intervals of days to use.\n    ')
    num_minor_ticks = Override(default=0)

class MonthsTicker(BaseSingleIntervalTicker):
    """ Generate ticks spaced apart by specific, even multiples of months.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    months = Seq(Int, default=[], help='\n    The intervals of months to use.\n    ')

class YearsTicker(BaseSingleIntervalTicker):
    """ Generate ticks spaced apart even numbers of years.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class BasicTicker(AdaptiveTicker):
    """ Generate ticks on a linear scale.

    .. note::
        This class may be renamed to ``LinearTicker`` in the future.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

class LogTicker(AdaptiveTicker):
    """ Generate ticks on a log scale.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    mantissas = Override(default=[1, 5])

class MercatorTicker(BasicTicker):
    """ Generate nice lat/lon ticks form underlying WebMercator coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    dimension = Nullable(Enum(LatLon), help='\n    Specify whether to generate ticks for Latitude or Longitude.\n\n    Projected coordinates are not separable, computing Latitude and Longitude\n    tick locations from Web Mercator requires considering coordinates from\n    both dimensions together. Use this property to specify which result should\n    be returned.\n\n    Typically, if the ticker is for an x-axis, then dimension should be\n    ``"lon"`` and if the ticker is for a y-axis, then the dimension\n    should be `"lat"``.\n\n    In order to prevent hard to debug errors, there is no default value for\n    dimension. Using an un-configured ``MercatorTicker`` will result in a\n    validation error and a JavaScript console error.\n    ')

    @error(MISSING_MERCATOR_DIMENSION)
    def _check_missing_dimension(self):
        if False:
            while True:
                i = 10
        if self.dimension is None:
            return str(self)

class CategoricalTicker(Ticker):
    """ Generate ticks for categorical ranges.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
ONE_MILLI = 1.0
ONE_SECOND = 1000.0
ONE_MINUTE = 60.0 * ONE_SECOND
ONE_HOUR = 60 * ONE_MINUTE
ONE_DAY = 24 * ONE_HOUR
ONE_MONTH = 30 * ONE_DAY
ONE_YEAR = 365 * ONE_DAY

class DatetimeTicker(CompositeTicker):
    """ Generate nice ticks across different date and time scales.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    num_minor_ticks = Override(default=0)
    tickers = Override(default=lambda : [AdaptiveTicker(mantissas=[1, 2, 5], base=10, min_interval=0, max_interval=500 * ONE_MILLI, num_minor_ticks=0), AdaptiveTicker(mantissas=[1, 2, 5, 10, 15, 20, 30], base=60, min_interval=ONE_SECOND, max_interval=30 * ONE_MINUTE, num_minor_ticks=0), AdaptiveTicker(mantissas=[1, 2, 4, 6, 8, 12], base=24, min_interval=ONE_HOUR, max_interval=12 * ONE_HOUR, num_minor_ticks=0), DaysTicker(days=list(range(1, 32))), DaysTicker(days=list(range(1, 31, 3))), DaysTicker(days=[1, 8, 15, 22]), DaysTicker(days=[1, 15]), MonthsTicker(months=list(range(0, 12, 1))), MonthsTicker(months=list(range(0, 12, 2))), MonthsTicker(months=list(range(0, 12, 4))), MonthsTicker(months=list(range(0, 12, 6))), YearsTicker()])

class BinnedTicker(Ticker):
    """ Ticker that aligns ticks exactly at bin boundaries of a scanning color mapper.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    mapper = Instance(ScanningColorMapper, help='\n    A scanning color mapper (e.g. ``EqHistColorMapper``) to use.\n    ')
    num_major_ticks = Either(Int, Auto, default=8, help='\n    The number of major tick positions to show or "auto" to use the\n    number of bins provided by the mapper.\n    ')