""" A guide renderer for displaying grid lines on Bokeh plots.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.properties import Auto, Either, Float, Include, Instance, Int, Nullable, Override, Seq, Tuple
from ..core.property_mixins import ScalarFillProps, ScalarHatchProps, ScalarLineProps
from .axes import Axis
from .renderers import GuideRenderer
from .tickers import FixedTicker, Ticker
__all__ = ('Grid',)

class Grid(GuideRenderer):
    """ Display horizontal or vertical grid lines at locations
    given by a supplied ``Ticker``.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    dimension = Int(0, help='\n    Which dimension the Axis Grid lines will intersect. The\n    x-axis is dimension 0 (vertical Grid lines) and the y-axis\n    is dimension 1 (horizontal Grid lines).\n    ')
    bounds = Either(Auto, Tuple(Float, Float), help='\n    Bounds for the rendered grid lines. By default, a grid will look for a\n    corresponding axis to ask for bounds. If one cannot be found, the grid\n    will span the entire visible range.\n    ')
    cross_bounds = Either(Auto, Tuple(Float, Float), help='\n    Bounds for the rendered grid lines in the orthogonal direction. By default,\n    a grid will span the entire visible range.\n    ')
    axis = Nullable(Instance(Axis), help='\n    An Axis to delegate ticking to. If the ticker property is None, then the\n    Grid will use the ticker on the specified axis for computing where to draw\n    grid lines. Otherwise, it ticker is not None, it will take precedence over\n    any Axis.\n    ')
    ticker = Nullable(Instance(Ticker), help='\n    A Ticker to use for computing locations for the Grid lines.\n    ').accepts(Seq(Float), lambda ticks: FixedTicker(ticks=ticks))
    grid_props = Include(ScalarLineProps, prefix='grid', help='\n    The {prop} of the Grid lines.\n    ')
    grid_line_color = Override(default='#e5e5e5')
    minor_grid_props = Include(ScalarLineProps, prefix='minor_grid', help='\n    The {prop} of the minor Grid lines.\n    ')
    minor_grid_line_color = Override(default=None)
    band_fill_props = Include(ScalarFillProps, prefix='band', help='\n    The {prop} of alternating bands between Grid lines.\n    ')
    band_fill_alpha = Override(default=0)
    band_fill_color = Override(default=None)
    band_hatch_props = Include(ScalarHatchProps, prefix='band', help='\n    The {prop} of alternating bands between Grid lines.\n    ')
    level = Override(default='underlay')