""" Models for describing different kinds of ranges of values
in different kinds of spaces (e.g., continuous or categorical)
and with options for "auto sizing".

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from collections import Counter
from math import nan
from ..core.enums import PaddingUnits, StartEnd
from ..core.has_props import abstract
from ..core.properties import Auto, Bool, Datetime, Either, Enum, FactorSeq, Float, Instance, List, MinMaxBounds, Null, Nullable, Override, Readonly, Required, TimeDelta
from ..core.validation import error
from ..core.validation.errors import DUPLICATE_FACTORS
from ..model import Model
__all__ = ('DataRange', 'DataRange1d', 'FactorRange', 'Range', 'Range1d')

@abstract
class Range(Model):
    """ A base class for all range types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

@abstract
class NumericalRange(Range):
    """ A base class for numerical ranges.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    start = Required(Either(Float, Datetime, TimeDelta), help='\n    The start of the range.\n    ')
    end = Required(Either(Float, Datetime, TimeDelta), help='\n    The end of the range.\n    ')

class Range1d(NumericalRange):
    """ A fixed, closed range [start, end] in a continuous scalar
    dimension.

    In addition to supplying ``start`` and ``end`` keyword arguments
    to the ``Range1d`` initializer, you can also instantiate with
    the convenience syntax::

        Range(0, 10) # equivalent to Range(start=0, end=10)

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        if args and ('start' in kwargs or 'end' in kwargs):
            raise ValueError("'start' and 'end' keywords cannot be used with positional arguments")
        if args and len(args) != 2:
            raise ValueError('Only Range1d(start, end) acceptable when using positional arguments')
        if args:
            kwargs['start'] = args[0]
            kwargs['end'] = args[1]
        super().__init__(**kwargs)
    reset_start = Either(Null, Float, Datetime, TimeDelta, help='\n    The start of the range to apply after reset. If set to ``None`` defaults\n    to the ``start`` value during initialization.\n    ')
    reset_end = Either(Null, Float, Datetime, TimeDelta, help='\n    The end of the range to apply when resetting. If set to ``None`` defaults\n    to the ``end`` value during initialization.\n    ')
    bounds = Nullable(MinMaxBounds(accept_datetime=True), help="\n    The bounds that the range is allowed to go to. Typically used to prevent\n    the user from panning/zooming/etc away from the data.\n\n    If set to ``'auto'``, the bounds will be computed to the start and end of the Range.\n\n    Bounds are provided as a tuple of ``(min, max)`` so regardless of whether your range is\n    increasing or decreasing, the first item should be the minimum value of the range and the\n    second item should be the maximum. Setting min > max will result in a ``ValueError``.\n\n    By default, bounds are ``None`` and your plot to pan/zoom as far as you want. If you only\n    want to constrain one end of the plot, you can set min or max to None.\n\n    Examples:\n\n    .. code-block:: python\n\n        Range1d(0, 1, bounds='auto')  # Auto-bounded to 0 and 1 (Default behavior)\n        Range1d(start=0, end=1, bounds=(0, None))  # Maximum is unbounded, minimum bounded to 0\n\n    ")
    min_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom in, expressed as the\n    minimum visible interval. If set to ``None`` (default), the minimum\n    interval is not bound. Can be a ``TimeDelta``. ')
    max_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom out, expressed as the\n    maximum visible interval. Can be a ``TimeDelta``. Note that ``bounds`` can\n    impose an implicit constraint on the maximum interval as well. ')
    start = Override(default=0)
    end = Override(default=1)

@abstract
class DataRange(NumericalRange):
    """ A base class for all data range types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    renderers = Either(List(Instance(Model)), Auto, help='\n    An explicit list of renderers to autorange against. If unset,\n    defaults to all renderers on a plot.\n    ')
    start = Override(default=nan)
    end = Override(default=nan)

class DataRange1d(DataRange):
    """ An auto-fitting range in a continuous scalar dimension.

    By default the ``start`` and ``end`` of the range automatically
    assume min and max values of the data for associated renderers.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        if kwargs.get('follow') is not None:
            kwargs['bounds'] = None
        super().__init__(*args, **kwargs)
    range_padding = Either(Float, TimeDelta, default=0.1, help='\n    How much padding to add around the computed data bounds.\n\n    When ``range_padding_units`` is set to ``"percent"``, the span of the\n    range span is expanded to make the range ``range_padding`` percent larger.\n\n    When ``range_padding_units`` is set to ``"absolute"``, the start and end\n    of the range span are extended by the amount ``range_padding``.\n    ')
    range_padding_units = Enum(PaddingUnits, default='percent', help='\n    Whether the ``range_padding`` should be interpreted as a percentage, or\n    as an absolute quantity. (default: ``"percent"``)\n    ')
    bounds = Nullable(MinMaxBounds(accept_datetime=True), help="\n    The bounds that the range is allowed to go to. Typically used to prevent\n    the user from panning/zooming/etc away from the data.\n\n    By default, the bounds will be None, allowing your plot to pan/zoom as far\n    as you want. If bounds are 'auto' they will be computed to be the same as\n    the start and end of the ``DataRange1d``.\n\n    Bounds are provided as a tuple of ``(min, max)`` so regardless of whether\n    your range is increasing or decreasing, the first item should be the\n    minimum value of the range and the second item should be the maximum.\n    Setting ``min > max`` will result in a ``ValueError``.\n\n    If you only want to constrain one end of the plot, you can set ``min`` or\n    ``max`` to ``None`` e.g. ``DataRange1d(bounds=(None, 12))``\n    ")
    min_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom in, expressed as the\n    minimum visible interval. If set to ``None`` (default), the minimum\n    interval is not bound.')
    max_interval = Either(Null, Float, TimeDelta, help='\n    The level that the range is allowed to zoom out, expressed as the\n    maximum visible interval. Note that ``bounds`` can impose an\n    implicit constraint on the maximum interval as well.')
    flipped = Bool(default=False, help='\n    Whether the range should be "flipped" from its normal direction when\n    auto-ranging.\n    ')
    follow = Nullable(Enum(StartEnd), help='\n    Configure the data to follow one or the other data extreme, with a\n    maximum range size of ``follow_interval``.\n\n    If set to ``"start"`` then the range will adjust so that ``start`` always\n    corresponds to the minimum data value (or maximum, if ``flipped`` is\n    ``True``).\n\n    If set to ``"end"`` then the range will adjust so that ``end`` always\n    corresponds to the maximum data value (or minimum, if ``flipped`` is\n    ``True``).\n\n    If set to ``None`` (default), then auto-ranging does not follow, and\n    the range will encompass both the minimum and maximum data values.\n\n    ``follow`` cannot be used with bounds, and if set, bounds will be set to\n    ``None``.\n    ')
    follow_interval = Nullable(Either(Float, TimeDelta), help='\n    If ``follow`` is set to ``"start"`` or ``"end"`` then the range will\n    always be constrained to that::\n\n         abs(r.start - r.end) <= follow_interval\n\n    is maintained.\n\n    ')
    default_span = Either(Float, TimeDelta, default=2.0, help='\n    A default width for the interval, in case ``start`` is equal to ``end``\n    (if used with a log axis, default_span is in powers of 10).\n    ')
    only_visible = Bool(default=False, help='\n    If True, renderers that that are not visible will be excluded from automatic\n    bounds computations.\n    ')

class FactorRange(Range):
    """ A Range of values for a categorical dimension.

    In addition to supplying ``factors`` as a keyword argument to the
    ``FactorRange`` initializer, you may also instantiate with a sequence of
    positional arguments:

    .. code-block:: python

        FactorRange("foo", "bar") # equivalent to FactorRange(factors=["foo", "bar"])

    Users will normally supply categorical values directly:

    .. code-block:: python

        p.circle(x=["foo", "bar"], ...)

    BokehJS will create a mapping from ``"foo"`` and ``"bar"`` to a numerical
    coordinate system called *synthetic coordinates*. In the simplest cases,
    factors are separated by a distance of 1.0 in synthetic coordinates,
    however the exact mapping from factors to synthetic coordinates is
    affected by he padding properties as well as whether the number of levels
    the factors have.

    Users typically do not need to worry about the details of this mapping,
    however it can be useful to fine tune positions by adding offsets. When
    supplying factors as coordinates or values, it is possible to add an
    offset in the synthetic coordinate space by adding a final number value
    to a factor tuple. For example:

    .. code-block:: python

        p.circle(x=[("foo", 0.3), ...], ...)

    will position the first circle at an ``x`` position that is offset by
    adding 0.3 to the synthetic coordinate for ``"foo"``.

    """
    factors = FactorSeq(default=[], help='\n    A sequence of factors to define this categorical range.\n\n    Factors may have 1, 2, or 3 levels. For 1-level factors, each factor is\n    simply a string. For example:\n\n    .. code-block:: python\n\n        FactorRange(factors=["sales", "marketing", "engineering"])\n\n    defines a range with three simple factors that might represent different\n    units of a business.\n\n    For 2- and 3- level factors, each factor is a tuple of strings:\n\n    .. code-block:: python\n\n        FactorRange(factors=[\n            ["2016", "sales"], ["2016", "marketing"], ["2016", "engineering"],\n            ["2017", "sales"], ["2017", "marketing"], ["2017", "engineering"],\n        ])\n\n    defines a range with six 2-level factors that might represent the three\n    business units, grouped by year.\n\n    Note that factors and sub-factors *may only be strings*.\n\n    ')
    factor_padding = Float(default=0.0, help='\n    How much padding to add in between all lowest-level factors. When\n    ``factor_padding`` is non-zero, every factor in every group will have the\n    padding value applied.\n    ')
    subgroup_padding = Float(default=0.8, help="\n    How much padding to add in between mid-level groups of factors. This\n    property only applies when the overall factors have three levels. For\n    example with:\n\n    .. code-block:: python\n\n        FactorRange(factors=[\n            ['foo', 'A', '1'],  ['foo', 'A', '2'], ['foo', 'A', '3'],\n            ['foo', 'B', '2'],\n            ['bar', 'A', '1'],  ['bar', 'A', '2']\n        ])\n\n    This property dictates how much padding to add between the three factors\n    in the `['foo', 'A']` group, and between the two factors in the the\n    [`bar`]\n    ")
    group_padding = Float(default=1.4, help='\n    How much padding to add in between top-level groups of factors. This\n    property only applies when the overall range factors have either two or\n    three levels. For example, with:\n\n    .. code-block:: python\n\n        FactorRange(factors=[["foo", "1"], ["foo", "2"], ["bar", "1"]])\n\n    The top level groups correspond to ``"foo"` and ``"bar"``, and the\n    group padding will be applied between the factors ``["foo", "2"]`` and\n    ``["bar", "1"]``\n    ')
    range_padding = Float(default=0, help='\n    How much padding to add around the outside of computed range bounds.\n\n    When ``range_padding_units`` is set to ``"percent"``, the span of the\n    range span is expanded to make the range ``range_padding`` percent larger.\n\n    When ``range_padding_units`` is set to ``"absolute"``, the start and end\n    of the range span are extended by the amount ``range_padding``.\n    ')
    range_padding_units = Enum(PaddingUnits, default='percent', help='\n    Whether the ``range_padding`` should be interpreted as a percentage, or\n    as an absolute quantity. (default: ``"percent"``)\n    ')
    start = Readonly(Float, default=0, help='\n    The start of the range, in synthetic coordinates.\n\n    .. note::\n        Synthetic coordinates are only computed in the browser, based on the\n        factors and various padding properties. The value of ``start`` will only\n        be available in situations where bidirectional communication is\n        available (e.g. server, notebook).\n    ')
    end = Readonly(Float, default=0, help='\n    The end of the range, in synthetic coordinates.\n\n    .. note::\n        Synthetic coordinates are only computed in the browser, based on the\n        factors and various padding properties. The value of ``end`` will only\n        be available in situations where bidirectional communication is\n        available (e.g. server, notebook).\n    ')
    bounds = Nullable(MinMaxBounds(accept_datetime=False), help="\n    The bounds (in synthetic coordinates) that the range is allowed to go to.\n    Typically used to prevent the user from panning/zooming/etc away from the\n    data.\n\n    .. note::\n        Synthetic coordinates are only computed in the browser, based on the\n        factors and various padding properties. Some experimentation may be\n        required to arrive at bounds suitable for specific situations.\n\n    By default, the bounds will be None, allowing your plot to pan/zoom as far\n    as you want. If bounds are 'auto' they will be computed to be the same as\n    the start and end of the ``FactorRange``.\n    ")
    min_interval = Nullable(Float, help='\n    The level that the range is allowed to zoom in, expressed as the\n    minimum visible interval in synthetic coordinates. If set to ``None``\n    (default), the minimum interval is not bounded.\n\n    The default "width" of a category is 1.0 in synthetic coordinates.\n    However, the distance between factors is affected by the various\n    padding properties and whether or not factors are grouped.\n    ')
    max_interval = Nullable(Float, help='\n    The level that the range is allowed to zoom out, expressed as the\n    maximum visible interval in synthetic coordinates.. Note that ``bounds``\n    can impose an implicit constraint on the maximum interval as well.\n\n    The default "width" of a category is 1.0 in synthetic coordinates.\n    However, the distance between factors is affected by the various\n    padding properties and whether or not factors are grouped.\n    ')

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        if args and 'factors' in kwargs:
            raise ValueError("'factors' keyword cannot be used with positional arguments")
        elif args:
            kwargs['factors'] = list(args)
        super().__init__(**kwargs)

    @error(DUPLICATE_FACTORS)
    def _check_duplicate_factors(self):
        if False:
            while True:
                i = 10
        dupes = [item for (item, count) in Counter(self.factors).items() if count > 1]
        if dupes:
            return 'duplicate factors found: %s' % ', '.join((repr(x) for x in dupes))