from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..models import GMapPlot, LinearAxis, MercatorTicker, MercatorTickFormatter, Range1d
from ._figure import BaseFigureOptions
from ._plot import _get_num_minor_ticks
from ._tools import process_active_tools, process_tools_arg
from .glyph_api import GlyphAPI
DEFAULT_TOOLS = 'pan,wheel_zoom,reset,help'
__all__ = ('GMap', 'GMapFigureOptions', 'gmap')

class GMap(GMapPlot, GlyphAPI):
    """ A subclass of |Plot| that simplifies plot creation with default axes,
    grids, tools, etc.

    Args:
        google_api_key (str):
            Google requires an API key be supplied for maps to function. See:

            https://developers.google.com/maps/documentation/javascript/get-api-key

        map_options: (:class:`~bokeh.models.map_plots.GMapOptions`)
            Configuration specific to a Google Map

    In addition to all the Bokeh model property attributes documented below,
    the ``Figure`` initializer also accepts the following options, which can
    help simplify configuration:

    .. bokeh-options:: GMapFigureOptions
        :module: bokeh.plotting.gmap

    """

    def __init__(self, **kw) -> None:
        if False:
            i = 10
            return i + 15
        opts = GMapFigureOptions(kw)
        super().__init__(x_range=Range1d(), y_range=Range1d(), **kw)
        if opts.x_axis_location is not None:
            xf = MercatorTickFormatter(dimension='lon')
            xt = MercatorTicker(dimension='lon')
            xt.num_minor_ticks = _get_num_minor_ticks(LinearAxis, opts.x_minor_ticks)
            self.add_layout(LinearAxis(formatter=xf, ticker=xt, axis_label=opts.x_axis_label), opts.x_axis_location)
        if opts.y_axis_location is not None:
            yf = MercatorTickFormatter(dimension='lat')
            yt = MercatorTicker(dimension='lat')
            yt.num_minor_ticks = _get_num_minor_ticks(LinearAxis, opts.y_minor_ticks)
            self.add_layout(LinearAxis(formatter=yf, ticker=yt, axis_label=opts.y_axis_label), opts.y_axis_location)
        (tool_objs, tool_map) = process_tools_arg(self, opts.tools, opts.tooltips)
        self.add_tools(*tool_objs)
        process_active_tools(self.toolbar, tool_map, opts.active_drag, opts.active_inspect, opts.active_scroll, opts.active_tap, opts.active_multi)

    @property
    def plot(self):
        if False:
            return 10
        return self

    @property
    def coordinates(self):
        if False:
            while True:
                i = 10
        return None

def gmap(google_api_key, map_options, **kwargs) -> GMap:
    if False:
        return 10
    ' Create a new :class:`~bokeh.plotting.GMap` for plotting.\n\n    Args:\n        google_api_key (str):\n            Google requires an API key be supplied for maps to function. See:\n\n            https://developers.google.com/maps/documentation/javascript/get-api-key\n\n            The Google API key will be stored as a base64-encoded string in\n            the Bokeh Document JSON.\n\n        map_options: (:class:`~bokeh.models.map_plots.GMapOptions`)\n            Configuration specific to a Google Map\n\n    All other keyword arguments are passed to :class:`~bokeh.plotting.GMap`.\n\n    Returns:\n       :class:`~bokeh.plotting.GMap`\n\n    '
    return GMap(api_key=google_api_key, map_options=map_options, **kwargs)

class GMapFigureOptions(BaseFigureOptions):
    pass