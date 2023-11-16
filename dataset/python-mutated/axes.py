""" Guide renderers for various kinds of axes that can be added to
Bokeh plots

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.enums import Align, LabelOrientation
from ..core.has_props import abstract
from ..core.properties import Auto, Datetime, Dict, Either, Enum, Factor, Float, Include, Instance, InstanceDefault, Int, Null, Nullable, Override, Seq, String, TextLike, Tuple
from ..core.property_mixins import ScalarFillProps, ScalarLineProps, ScalarTextProps
from .formatters import BasicTickFormatter, CategoricalTickFormatter, DatetimeTickFormatter, LogTickFormatter, MercatorTickFormatter, TickFormatter
from .labeling import AllLabels, LabelingPolicy
from .renderers import GuideRenderer
from .tickers import BasicTicker, CategoricalTicker, DatetimeTicker, FixedTicker, LogTicker, MercatorTicker, Ticker
__all__ = ('Axis', 'CategoricalAxis', 'ContinuousAxis', 'DatetimeAxis', 'LinearAxis', 'LogAxis', 'MercatorAxis')

@abstract
class Axis(GuideRenderer):
    """ A base class that defines common properties for all axis types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    dimension = Either(Auto, Int, default='auto', help='\n    This allows to override the inferred dimensions in contexts that\n    support this. This property has no effect when an axes is used\n    as a frame axis.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    face = Either(Auto, Enum('front', 'back'))(default='auto', help='\n    The direction toward which the axis will face.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    bounds = Either(Auto, Tuple(Float, Float), Tuple(Datetime, Datetime), help='\n    Bounds for the rendered axis. If unset, the axis will span the\n    entire plot in the given dimension.\n    ')
    ticker = Instance(Ticker, help='\n    A Ticker to use for computing locations of axis components.\n\n    The property may also be passed a sequence of floating point numbers as\n    a shorthand for creating and configuring a ``FixedTicker``, e.g. the\n    following code\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure\n\n        p = figure()\n        p.xaxis.ticker = [10, 20, 37.4]\n\n    is equivalent to:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure\n        from bokeh.models import FixedTicker\n\n        p = figure()\n        p.xaxis.ticker = FixedTicker(ticks=[10, 20, 37.4])\n\n    ').accepts(Seq(Float), lambda ticks: FixedTicker(ticks=ticks))
    formatter = Instance(TickFormatter, help='\n    A ``TickFormatter`` to use for formatting the visual appearance\n    of ticks.\n    ')
    axis_label = Nullable(TextLike, help='\n    A text or LaTeX notation label for the axis, displayed parallel to the axis rule.\n    ')
    axis_label_standoff = Int(default=5, help='\n    The distance in pixels that the axis labels should be offset\n    from the tick labels.\n    ')
    axis_label_orientation = Either(Enum(LabelOrientation), Float)(default='parallel', help='\n    What direction the asix label text should be oriented. If a number\n    is supplied, the angle of the text is measured from horizontal.\n    ')
    axis_label_align = Enum(Align, default='center', help='\n    The alignment of axis label along the axis.\n    ')
    axis_label_props = Include(ScalarTextProps, prefix='axis_label', help='\n    The {prop} of the axis label.\n    ')
    axis_label_text_font_size = Override(default='13px')
    axis_label_text_font_style = Override(default='italic')
    major_label_standoff = Int(default=5, help='\n    The distance in pixels that the major tick labels should be\n    offset from the associated ticks.\n    ')
    major_label_orientation = Either(Enum(LabelOrientation), Float)(default='horizontal', help='\n    What direction the major label text should be oriented. If a number\n    is supplied, the angle of the text is measured from horizontal.\n    ')
    major_label_overrides = Dict(Either(Float, String), TextLike, default={}, help='\n    Provide explicit tick label values for specific tick locations that\n    override normal formatting.\n    ')
    major_label_policy = Instance(LabelingPolicy, default=InstanceDefault(AllLabels), help='\n    Allows to filter out labels, e.g. declutter labels to avoid overlap.\n    ')
    major_label_props = Include(ScalarTextProps, prefix='major_label', help='\n    The {prop} of the major tick labels.\n    ')
    major_label_text_align = Override(default='center')
    major_label_text_baseline = Override(default='alphabetic')
    major_label_text_font_size = Override(default='11px')
    axis_props = Include(ScalarLineProps, prefix='axis', help='\n    The {prop} of the axis line.\n    ')
    major_tick_props = Include(ScalarLineProps, prefix='major_tick', help='\n    The {prop} of the major ticks.\n    ')
    major_tick_in = Int(default=2, help='\n    The distance in pixels that major ticks should extend into the\n    main plot area.\n    ')
    major_tick_out = Int(default=6, help='\n    The distance in pixels that major ticks should extend out of the\n    main plot area.\n    ')
    minor_tick_props = Include(ScalarLineProps, prefix='minor_tick', help='\n    The {prop} of the minor ticks.\n    ')
    minor_tick_in = Int(default=0, help='\n    The distance in pixels that minor ticks should extend into the\n    main plot area.\n    ')
    minor_tick_out = Int(default=4, help='\n    The distance in pixels that major ticks should extend out of the\n    main plot area.\n    ')
    fixed_location = Either(Null, Float, Factor, help='\n    Set to specify a fixed coordinate location to draw the axis. The direction\n    of ticks and major labels is determined by the side panel that the axis\n    belongs to.\n\n    .. note::\n        Axes labels are suppressed when axes are positioned at fixed locations\n        inside the central plot area.\n    ')
    background_props = Include(ScalarFillProps, prefix='background', help='\n    The {prop} of the axis background.\n    ')
    background_fill_color = Override(default=None)

@abstract
class ContinuousAxis(Axis):
    """ A base class for all numeric, non-categorical axes types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)

class LinearAxis(ContinuousAxis):
    """ An axis that picks nice numbers for tick locations on a
    linear scale. Configured with a ``BasicTickFormatter`` by default.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    ticker = Override(default=InstanceDefault(BasicTicker))
    formatter = Override(default=InstanceDefault(BasicTickFormatter))

class LogAxis(ContinuousAxis):
    """ An axis that picks nice numbers for tick locations on a
    log scale. Configured with a ``LogTickFormatter`` by default.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    ticker = Override(default=InstanceDefault(LogTicker))
    formatter = Override(default=InstanceDefault(LogTickFormatter))

class CategoricalAxis(Axis):
    """ An axis that displays ticks and labels for categorical ranges.

    The ``CategoricalAxis`` can handle factor ranges with up to two levels of
    nesting, including drawing a separator line between top-level groups of
    factors.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    ticker = Override(default=InstanceDefault(CategoricalTicker))
    formatter = Override(default=InstanceDefault(CategoricalTickFormatter))
    separator_props = Include(ScalarLineProps, prefix='separator', help='\n    The {prop} of the separator line between top-level categorical groups.\n\n    This property always applies to factors in the outermost level of nesting.\n    ')
    separator_line_color = Override(default='lightgrey')
    separator_line_width = Override(default=2)
    group_props = Include(ScalarTextProps, prefix='group', help='\n    The {prop} of the group categorical labels.\n\n    This property always applies to factors in the outermost level of nesting.\n    If the list of categorical factors is flat (i.e. no nesting) then this\n    property has no effect.\n    ')
    group_label_orientation = Either(Enum(LabelOrientation), Float, default='parallel', help='\n    What direction the group label text should be oriented.\n\n    If a number is supplied, the angle of the text is measured from horizontal.\n\n    This property always applies to factors in the outermost level of nesting.\n    If the list of categorical factors is flat (i.e. no nesting) then this\n    property has no effect.\n    ')
    group_text_font_size = Override(default='11px')
    group_text_font_style = Override(default='bold')
    group_text_color = Override(default='grey')
    subgroup_props = Include(ScalarTextProps, prefix='subgroup', help='\n    The {prop} of the subgroup categorical labels.\n\n    This property always applies to factors in the middle level of nesting.\n    If the list of categorical factors is has only zero or one levels of nesting,\n    then this property has no effect.\n    ')
    subgroup_label_orientation = Either(Enum(LabelOrientation), Float, default='parallel', help='\n    What direction the subgroup label text should be oriented.\n\n    If a number is supplied, the angle of the text is measured from horizontal.\n\n    This property always applies to factors in the middle level of nesting.\n    If the list of categorical factors is has only zero or one levels of nesting,\n    then this property has no effect.\n    ')
    subgroup_text_font_size = Override(default='11px')
    subgroup_text_font_style = Override(default='bold')

class DatetimeAxis(LinearAxis):
    """ A ``LinearAxis`` that picks nice numbers for tick locations on
    a datetime scale. Configured with a ``DatetimeTickFormatter`` by
    default.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    ticker = Override(default=InstanceDefault(DatetimeTicker))
    formatter = Override(default=InstanceDefault(DatetimeTickFormatter))

class MercatorAxis(LinearAxis):
    """ An axis that picks nice numbers for tick locations on a
    Mercator scale. Configured with a ``MercatorTickFormatter`` by default.

    Args:
        dimension ('lat' or 'lon', optional) :
            Whether this axis will display latitude or longitude values.
            (default: 'lat')

    """

    def __init__(self, dimension='lat', *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        if isinstance(self.ticker, MercatorTicker):
            self.ticker.dimension = dimension
        if isinstance(self.formatter, MercatorTickFormatter):
            self.formatter.dimension = dimension
    ticker = Override(default=InstanceDefault(MercatorTicker))
    formatter = Override(default=InstanceDefault(MercatorTickFormatter))