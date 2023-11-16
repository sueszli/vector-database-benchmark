from __future__ import annotations
from typing import TYPE_CHECKING
import logging
log = logging.getLogger(__name__)
import numpy as np
from ..core.enums import HorizontalLocation, MarkerType, VerticalLocation
from ..core.properties import Auto, Datetime, Either, Enum, Float, Instance, InstanceDefault, Int, List, Nullable, Object, Seq, String, TextLike, TimeDelta, Tuple
from ..models import ColumnDataSource, CoordinateMapping, DataRange1d, GraphRenderer, Plot, Range, Scale, Tool
from ..models.dom import Template
from ..models.tools import Drag, GestureTool, InspectTool, Scroll, Tap
from ..transform import linear_cmap
from ..util.options import Options
from ._graph import get_graph_kwargs
from ._plot import get_range, get_scale, process_axis_and_grid
from ._stack import double_stack, single_stack
from ._tools import process_active_tools, process_tools_arg
from .contour import ContourRenderer, from_contour
from .glyph_api import _MARKER_SHORTCUTS, GlyphAPI
if TYPE_CHECKING:
    from numpy.typing import ArrayLike
DEFAULT_TOOLS = 'pan,wheel_zoom,box_zoom,save,reset,help'
__all__ = ('figure', 'markers')

class figure(Plot, GlyphAPI):
    """ Create a new figure for plotting.

    A subclass of |Plot| that simplifies plot creation with default axes, grids,
    tools, etc.

    Figure objects have many glyph methods that can be used to draw
    vectorized graphical glyphs:

    .. hlist::
        :columns: 3

        * :func:`~bokeh.plotting.figure.annular_wedge`
        * :func:`~bokeh.plotting.figure.annulus`
        * :func:`~bokeh.plotting.figure.arc`
        * :func:`~bokeh.plotting.figure.asterisk`
        * :func:`~bokeh.plotting.figure.bezier`
        * :func:`~bokeh.plotting.figure.circle`
        * :func:`~bokeh.plotting.figure.circle_cross`
        * :func:`~bokeh.plotting.figure.circle_dot`
        * :func:`~bokeh.plotting.figure.circle_x`
        * :func:`~bokeh.plotting.figure.circle_y`
        * :func:`~bokeh.plotting.figure.cross`
        * :func:`~bokeh.plotting.figure.dash`
        * :func:`~bokeh.plotting.figure.diamond`
        * :func:`~bokeh.plotting.figure.diamond_cross`
        * :func:`~bokeh.plotting.figure.diamond_dot`
        * :func:`~bokeh.plotting.figure.dot`
        * :func:`~bokeh.plotting.figure.ellipse`
        * :func:`~bokeh.plotting.figure.harea`
        * :func:`~bokeh.plotting.figure.harea_step`
        * :func:`~bokeh.plotting.figure.hbar`
        * :func:`~bokeh.plotting.figure.hex`
        * :func:`~bokeh.plotting.figure.hex_tile`
        * :func:`~bokeh.plotting.figure.image`
        * :func:`~bokeh.plotting.figure.image_rgba`
        * :func:`~bokeh.plotting.figure.image_url`
        * :func:`~bokeh.plotting.figure.inverted_triangle`
        * :func:`~bokeh.plotting.figure.line`
        * :func:`~bokeh.plotting.figure.multi_line`
        * :func:`~bokeh.plotting.figure.multi_polygons`
        * :func:`~bokeh.plotting.figure.oval`
        * :func:`~bokeh.plotting.figure.patch`
        * :func:`~bokeh.plotting.figure.patches`
        * :func:`~bokeh.plotting.figure.plus`
        * :func:`~bokeh.plotting.figure.quad`
        * :func:`~bokeh.plotting.figure.quadratic`
        * :func:`~bokeh.plotting.figure.ray`
        * :func:`~bokeh.plotting.figure.rect`
        * :func:`~bokeh.plotting.figure.segment`
        * :func:`~bokeh.plotting.figure.square`
        * :func:`~bokeh.plotting.figure.square_cross`
        * :func:`~bokeh.plotting.figure.square_dot`
        * :func:`~bokeh.plotting.figure.square_pin`
        * :func:`~bokeh.plotting.figure.square_x`
        * :func:`~bokeh.plotting.figure.star`
        * :func:`~bokeh.plotting.figure.star_dot`
        * :func:`~bokeh.plotting.figure.step`
        * :func:`~bokeh.plotting.figure.text`
        * :func:`~bokeh.plotting.figure.triangle`
        * :func:`~bokeh.plotting.figure.triangle_dot`
        * :func:`~bokeh.plotting.figure.triangle_pin`
        * :func:`~bokeh.plotting.figure.varea`
        * :func:`~bokeh.plotting.figure.varea_step`
        * :func:`~bokeh.plotting.figure.vbar`
        * :func:`~bokeh.plotting.figure.wedge`
        * :func:`~bokeh.plotting.figure.x`
        * :func:`~bokeh.plotting.figure.y`

    There is a scatter function that can be parameterized by marker type:

    * :func:`~bokeh.plotting.figure.scatter`

    There are also specialized methods for stacking bars:

    * bars: :func:`~bokeh.plotting.figure.hbar_stack`, :func:`~bokeh.plotting.figure.vbar_stack`
    * lines: :func:`~bokeh.plotting.figure.hline_stack`, :func:`~bokeh.plotting.figure.vline_stack`
    * areas: :func:`~bokeh.plotting.figure.harea_stack`, :func:`~bokeh.plotting.figure.varea_stack`

    As well as one specialized method for making simple hexbin plots:

    * :func:`~bokeh.plotting.figure.hexbin`

    In addition to all the ``figure`` property attributes, the following
    options are also accepted:

    .. bokeh-options:: FigureOptions
        :module: bokeh.plotting._figure

    """
    __view_model__ = 'Figure'

    def __init__(self, *arg, **kw) -> None:
        if False:
            for i in range(10):
                print('nop')
        opts = FigureOptions(kw)
        names = self.properties()
        for name in kw.keys():
            if name not in names:
                self._raise_attribute_error_with_matches(name, names | opts.properties())
        super().__init__(*arg, **kw)
        self.x_range = get_range(opts.x_range)
        self.y_range = get_range(opts.y_range)
        self.x_scale = get_scale(self.x_range, opts.x_axis_type)
        self.y_scale = get_scale(self.y_range, opts.y_axis_type)
        process_axis_and_grid(self, opts.x_axis_type, opts.x_axis_location, opts.x_minor_ticks, opts.x_axis_label, self.x_range, 0)
        process_axis_and_grid(self, opts.y_axis_type, opts.y_axis_location, opts.y_minor_ticks, opts.y_axis_label, self.y_range, 1)
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
            for i in range(10):
                print('nop')
        return None

    def subplot(self, *, x_source: Range | None=None, y_source: Range | None=None, x_scale: Scale | None=None, y_scale: Scale | None=None, x_target: Range, y_target: Range) -> GlyphAPI:
        if False:
            for i in range(10):
                print('nop')
        ' Create a new sub-coordinate system and expose a plotting API. '
        coordinates = CoordinateMapping(x_source=x_source, y_source=y_source, x_target=x_target, y_target=y_target)
        return GlyphAPI(self, coordinates)

    def hexbin(self, x, y, size, orientation='pointytop', palette='Viridis256', line_color=None, fill_color=None, aspect_scale=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Perform a simple equal-weight hexagonal binning.\n\n        A :class:`~bokeh.models.glyphs.HexTile` glyph will be added to display\n        the binning. The :class:`~bokeh.models.sources.ColumnDataSource` for\n        the glyph will have columns ``q``, ``r``, and ``count``, where ``q``\n        and ``r`` are `axial coordinates`_ for a tile, and ``count`` is the\n        associated bin count.\n\n        It is often useful to set ``match_aspect=True`` on the associated plot,\n        so that hexagonal tiles are all regular (i.e. not "stretched") in\n        screen space.\n\n        For more sophisticated use-cases, e.g. weighted binning or individually\n        scaling hex tiles, use :func:`hex_tile` directly, or consider a higher\n        level library such as HoloViews.\n\n        Args:\n            x (array[float]) :\n                A NumPy array of x-coordinates to bin into hexagonal tiles.\n\n            y (array[float]) :\n                A NumPy array of y-coordinates to bin into hexagonal tiles.\n\n            size (float) :\n                The size of the hexagonal tiling to use. The size is defined as\n                distance from the center of a hexagon to a corner.\n\n                In case the aspect scaling is not 1-1, then specifically `size`\n                is the distance from the center to the "top" corner with the\n                `"pointytop"` orientation, and the distance from the center to\n                a "side" corner with the "flattop" orientation.\n\n            orientation ("pointytop" or "flattop", optional) :\n                Whether the hexagonal tiles should be oriented with a pointed\n                corner on top, or a flat side on top. (default: "pointytop")\n\n            palette (str or seq[color], optional) :\n                A palette (or palette name) to use to colormap the bins according\n                to count. (default: \'Viridis256\')\n\n                If ``fill_color`` is supplied, it overrides this value.\n\n            line_color (color, optional) :\n                The outline color for hex tiles, or None (default: None)\n\n            fill_color (color, optional) :\n                An optional fill color for hex tiles, or None. If None, then\n                the ``palette`` will be used to color map the tiles by\n                count. (default: None)\n\n            aspect_scale (float) :\n                Match a plot\'s aspect ratio scaling.\n\n                When working with a plot with ``aspect_scale != 1``, this\n                parameter can be set to match the plot, in order to draw\n                regular hexagons (instead of "stretched" ones).\n\n                This is roughly equivalent to binning in "screen space", and\n                it may be better to use axis-aligned rectangular bins when\n                plot aspect scales are not one.\n\n        Any additional keyword arguments are passed to :func:`hex_tile`.\n\n        Returns:\n            (Glyphrender, DataFrame)\n                A tuple with the ``HexTile`` renderer generated to display the\n                binning, and a Pandas ``DataFrame`` with columns ``q``, ``r``,\n                and ``count``, where ``q`` and ``r`` are `axial coordinates`_\n                for a tile, and ``count`` is the associated bin count.\n\n        Example:\n\n            .. bokeh-plot::\n                :source-position: above\n\n                import numpy as np\n                from bokeh.models import HoverTool\n                from bokeh.plotting import figure, show\n\n                x = 2 + 2*np.random.standard_normal(500)\n                y = 2 + 2*np.random.standard_normal(500)\n\n                p = figure(match_aspect=True, tools="wheel_zoom,reset")\n                p.background_fill_color = \'#440154\'\n                p.grid.visible = False\n\n                p.hexbin(x, y, size=0.5, hover_color="pink", hover_alpha=0.8)\n\n                hover = HoverTool(tooltips=[("count", "@c"), ("(q,r)", "(@q, @r)")])\n                p.add_tools(hover)\n\n                show(p)\n\n        .. _axial coordinates: https://www.redblobgames.com/grids/hexagons/#coordinates-axial\n\n        '
        from ..util.hex import hexbin
        bins = hexbin(x, y, size, orientation, aspect_scale=aspect_scale)
        if fill_color is None:
            fill_color = linear_cmap('c', palette, 0, max(bins.counts))
        source = ColumnDataSource(data=dict(q=bins.q, r=bins.r, c=bins.counts))
        r = self.hex_tile(q='q', r='r', size=size, orientation=orientation, aspect_scale=aspect_scale, source=source, line_color=line_color, fill_color=fill_color, **kwargs)
        return (r, bins)

    def harea_stack(self, stackers, **kw):
        if False:
            for i in range(10):
                print('nop')
        " Generate multiple ``HArea`` renderers for levels stacked left\n        to right.\n\n        Args:\n            stackers (seq[str]) : a list of data source field names to stack\n                successively for ``x1`` and ``x2`` harea coordinates.\n\n                Additionally, the ``name`` of the renderer will be set to\n                the value of each successive stacker (this is useful with the\n                special hover variable ``$name``)\n\n        Any additional keyword arguments are passed to each call to ``harea``.\n        If a keyword value is a list or tuple, then each call will get one\n        value from the sequence.\n\n        Returns:\n            list[GlyphRenderer]\n\n        Examples:\n\n            Assuming a ``ColumnDataSource`` named ``source`` with columns\n            *2016* and *2017*, then the following call to ``harea_stack`` will\n            will create two ``HArea`` renderers that stack:\n\n            .. code-block:: python\n\n                p.harea_stack(['2016', '2017'], y='y', color=['blue', 'red'], source=source)\n\n            This is equivalent to the following two separate calls:\n\n            .. code-block:: python\n\n                p.harea(x1=stack(),       x2=stack('2016'),         y='y', color='blue', source=source, name='2016')\n                p.harea(x1=stack('2016'), x2=stack('2016', '2017'), y='y', color='red',  source=source, name='2017')\n\n        "
        result = []
        for kw in double_stack(stackers, 'x1', 'x2', **kw):
            result.append(self.harea(**kw))
        return result

    def hbar_stack(self, stackers, **kw):
        if False:
            while True:
                i = 10
        " Generate multiple ``HBar`` renderers for levels stacked left to right.\n\n        Args:\n            stackers (seq[str]) : a list of data source field names to stack\n                successively for ``left`` and ``right`` bar coordinates.\n\n                Additionally, the ``name`` of the renderer will be set to\n                the value of each successive stacker (this is useful with the\n                special hover variable ``$name``)\n\n        Any additional keyword arguments are passed to each call to ``hbar``.\n        If a keyword value is a list or tuple, then each call will get one\n        value from the sequence.\n\n        Returns:\n            list[GlyphRenderer]\n\n        Examples:\n\n            Assuming a ``ColumnDataSource`` named ``source`` with columns\n            *2016* and *2017*, then the following call to ``hbar_stack`` will\n            will create two ``HBar`` renderers that stack:\n\n            .. code-block:: python\n\n                p.hbar_stack(['2016', '2017'], y=10, width=0.9, color=['blue', 'red'], source=source)\n\n            This is equivalent to the following two separate calls:\n\n            .. code-block:: python\n\n                p.hbar(bottom=stack(),       top=stack('2016'),         y=10, width=0.9, color='blue', source=source, name='2016')\n                p.hbar(bottom=stack('2016'), top=stack('2016', '2017'), y=10, width=0.9, color='red',  source=source, name='2017')\n\n        "
        result = []
        for kw in double_stack(stackers, 'left', 'right', **kw):
            result.append(self.hbar(**kw))
        return result

    def _line_stack(self, x, y, **kw):
        if False:
            return 10
        " Generate multiple ``Line`` renderers for lines stacked vertically\n        or horizontally.\n\n        Args:\n            x (seq[str]) :\n\n            y (seq[str]) :\n\n        Additionally, the ``name`` of the renderer will be set to\n        the value of each successive stacker (this is useful with the\n        special hover variable ``$name``)\n\n        Any additional keyword arguments are passed to each call to ``hbar``.\n        If a keyword value is a list or tuple, then each call will get one\n        value from the sequence.\n\n        Returns:\n            list[GlyphRenderer]\n\n        Examples:\n\n            Assuming a ``ColumnDataSource`` named ``source`` with columns\n            *2016* and *2017*, then the following call to ``line_stack`` with\n            stackers for the y-coordinates will will create two ``Line``\n            renderers that stack:\n\n            .. code-block:: python\n\n                p.line_stack(['2016', '2017'], x='x', color=['blue', 'red'], source=source)\n\n            This is equivalent to the following two separate calls:\n\n            .. code-block:: python\n\n                p.line(y=stack('2016'),         x='x', color='blue', source=source, name='2016')\n                p.line(y=stack('2016', '2017'), x='x', color='red',  source=source, name='2017')\n\n        "
        if all((isinstance(val, (list, tuple)) for val in (x, y))):
            raise ValueError('Only one of x or y may be a list of stackers')
        result = []
        if isinstance(y, (list, tuple)):
            kw['x'] = x
            for kw in single_stack(y, 'y', **kw):
                result.append(self.line(**kw))
            return result
        if isinstance(x, (list, tuple)):
            kw['y'] = y
            for kw in single_stack(x, 'x', **kw):
                result.append(self.line(**kw))
            return result
        return [self.line(x, y, **kw)]

    def hline_stack(self, stackers, **kw):
        if False:
            return 10
        " Generate multiple ``Line`` renderers for lines stacked horizontally.\n\n        Args:\n            stackers (seq[str]) : a list of data source field names to stack\n                successively for ``x`` line coordinates.\n\n        Additionally, the ``name`` of the renderer will be set to\n        the value of each successive stacker (this is useful with the\n        special hover variable ``$name``)\n\n        Any additional keyword arguments are passed to each call to ``line``.\n        If a keyword value is a list or tuple, then each call will get one\n        value from the sequence.\n\n        Returns:\n            list[GlyphRenderer]\n\n        Examples:\n\n            Assuming a ``ColumnDataSource`` named ``source`` with columns\n            *2016* and *2017*, then the following call to ``hline_stack`` with\n            stackers for the x-coordinates will will create two ``Line``\n            renderers that stack:\n\n            .. code-block:: python\n\n                p.hline_stack(['2016', '2017'], y='y', color=['blue', 'red'], source=source)\n\n            This is equivalent to the following two separate calls:\n\n            .. code-block:: python\n\n                p.line(x=stack('2016'),         y='y', color='blue', source=source, name='2016')\n                p.line(x=stack('2016', '2017'), y='y', color='red',  source=source, name='2017')\n\n        "
        return self._line_stack(x=stackers, **kw)

    def varea_stack(self, stackers, **kw):
        if False:
            print('Hello World!')
        " Generate multiple ``VArea`` renderers for levels stacked bottom\n        to top.\n\n        Args:\n            stackers (seq[str]) : a list of data source field names to stack\n                successively for ``y1`` and ``y1`` varea coordinates.\n\n                Additionally, the ``name`` of the renderer will be set to\n                the value of each successive stacker (this is useful with the\n                special hover variable ``$name``)\n\n        Any additional keyword arguments are passed to each call to ``varea``.\n        If a keyword value is a list or tuple, then each call will get one\n        value from the sequence.\n\n        Returns:\n            list[GlyphRenderer]\n\n        Examples:\n\n            Assuming a ``ColumnDataSource`` named ``source`` with columns\n            *2016* and *2017*, then the following call to ``varea_stack`` will\n            will create two ``VArea`` renderers that stack:\n\n            .. code-block:: python\n\n                p.varea_stack(['2016', '2017'], x='x', color=['blue', 'red'], source=source)\n\n            This is equivalent to the following two separate calls:\n\n            .. code-block:: python\n\n                p.varea(y1=stack(),       y2=stack('2016'),         x='x', color='blue', source=source, name='2016')\n                p.varea(y1=stack('2016'), y2=stack('2016', '2017'), x='x', color='red',  source=source, name='2017')\n\n        "
        result = []
        for kw in double_stack(stackers, 'y1', 'y2', **kw):
            result.append(self.varea(**kw))
        return result

    def vbar_stack(self, stackers, **kw):
        if False:
            for i in range(10):
                print('nop')
        " Generate multiple ``VBar`` renderers for levels stacked bottom\n        to top.\n\n        Args:\n            stackers (seq[str]) : a list of data source field names to stack\n                successively for ``left`` and ``right`` bar coordinates.\n\n                Additionally, the ``name`` of the renderer will be set to\n                the value of each successive stacker (this is useful with the\n                special hover variable ``$name``)\n\n        Any additional keyword arguments are passed to each call to ``vbar``.\n        If a keyword value is a list or tuple, then each call will get one\n        value from the sequence.\n\n        Returns:\n            list[GlyphRenderer]\n\n        Examples:\n\n            Assuming a ``ColumnDataSource`` named ``source`` with columns\n            *2016* and *2017*, then the following call to ``vbar_stack`` will\n            will create two ``VBar`` renderers that stack:\n\n            .. code-block:: python\n\n                p.vbar_stack(['2016', '2017'], x=10, width=0.9, color=['blue', 'red'], source=source)\n\n            This is equivalent to the following two separate calls:\n\n            .. code-block:: python\n\n                p.vbar(bottom=stack(),       top=stack('2016'),         x=10, width=0.9, color='blue', source=source, name='2016')\n                p.vbar(bottom=stack('2016'), top=stack('2016', '2017'), x=10, width=0.9, color='red',  source=source, name='2017')\n\n        "
        result = []
        for kw in double_stack(stackers, 'bottom', 'top', **kw):
            result.append(self.vbar(**kw))
        return result

    def vline_stack(self, stackers, **kw):
        if False:
            i = 10
            return i + 15
        " Generate multiple ``Line`` renderers for lines stacked vertically.\n\n        Args:\n            stackers (seq[str]) : a list of data source field names to stack\n                successively for ``y`` line coordinates.\n\n        Additionally, the ``name`` of the renderer will be set to\n        the value of each successive stacker (this is useful with the\n        special hover variable ``$name``)\n\n        Any additional keyword arguments are passed to each call to ``line``.\n        If a keyword value is a list or tuple, then each call will get one\n        value from the sequence.\n\n        Returns:\n            list[GlyphRenderer]\n\n        Examples:\n\n            Assuming a ``ColumnDataSource`` named ``source`` with columns\n            *2016* and *2017*, then the following call to ``vline_stack`` with\n            stackers for the y-coordinates will will create two ``Line``\n            renderers that stack:\n\n            .. code-block:: python\n\n                p.vline_stack(['2016', '2017'], x='x', color=['blue', 'red'], source=source)\n\n            This is equivalent to the following two separate calls:\n\n            .. code-block:: python\n\n                p.line(y=stack('2016'),         x='x', color='blue', source=source, name='2016')\n                p.line(y=stack('2016', '2017'), x='x', color='red',  source=source, name='2017')\n\n        "
        return self._line_stack(y=stackers, **kw)

    def graph(self, node_source, edge_source, layout_provider, **kwargs):
        if False:
            print('Hello World!')
        ' Creates a network graph using the given node, edge and layout provider.\n\n        Args:\n            node_source (:class:`~bokeh.models.sources.ColumnDataSource`) : a user-supplied data source\n                for the graph nodes. An attempt will be made to convert the object to\n                :class:`~bokeh.models.sources.ColumnDataSource` if needed. If none is supplied, one is created\n                for the user automatically.\n\n            edge_source (:class:`~bokeh.models.sources.ColumnDataSource`) : a user-supplied data source\n                for the graph edges. An attempt will be made to convert the object to\n                :class:`~bokeh.models.sources.ColumnDataSource` if needed. If none is supplied, one is created\n                for the user automatically.\n\n            layout_provider (:class:`~bokeh.models.graphs.LayoutProvider`) : a ``LayoutProvider`` instance to\n                provide the graph coordinates in Cartesian space.\n\n            **kwargs: |line properties| and |fill properties|\n\n        '
        kw = get_graph_kwargs(node_source, edge_source, **kwargs)
        graph_renderer = GraphRenderer(layout_provider=layout_provider, **kw)
        self.renderers.append(graph_renderer)
        return graph_renderer

    def contour(self, x: ArrayLike | None=None, y: ArrayLike | None=None, z: ArrayLike | np.ma.MaskedArray | None=None, levels: ArrayLike | None=None, **visuals) -> ContourRenderer:
        if False:
            i = 10
            return i + 15
        ' Creates a contour plot of filled polygons and/or contour lines.\n\n        Filled contour polygons are calculated if ``fill_color`` is set,\n        contour lines if ``line_color`` is set.\n\n        Args:\n            x (array-like[float] of shape (ny, nx) or (nx,), optional) :\n                The x-coordinates of the ``z`` values. May be 2D with the same\n                shape as ``z.shape``, or 1D with length ``nx = z.shape[1]``.\n                If not specified are assumed to be ``np.arange(nx)``. Must be\n                ordered monotonically.\n\n            y (array-like[float] of shape (ny, nx) or (ny,), optional) :\n                The y-coordinates of the ``z`` values. May be 2D with the same\n                shape as ``z.shape``, or 1D with length ``ny = z.shape[0]``.\n                If not specified are assumed to be ``np.arange(ny)``. Must be\n                ordered monotonically.\n\n            z (array-like[float] of shape (ny, nx)) :\n                A 2D NumPy array of gridded values to calculate the contours\n                of.  May be a masked array, and any invalid values (``np.inf``\n                or ``np.nan``) will also be masked out.\n\n            levels (array-like[float]) :\n                The z-levels to calculate the contours at, must be increasing.\n                Contour lines are calculated at each level and filled contours\n                are calculated between each adjacent pair of levels so the\n                number of sets of contour lines is ``len(levels)`` and the\n                number of sets of filled contour polygons is ``len(levels)-1``.\n\n            **visuals: |fill properties|, |hatch properties| and |line properties|\n                Fill and hatch properties are used for filled contours, line\n                properties for line contours. If using vectorized properties\n                then the correct number must be used, ``len(levels)`` for line\n                properties and ``len(levels)-1`` for fill and hatch properties.\n\n                ``fill_color`` and ``line_color`` are more flexible in that\n                they will accept longer sequences and interpolate them to the\n                required number using :func:`~bokeh.palettes.linear_palette`,\n                and also accept palette collections (dictionaries mapping from\n                integer length to color sequence) such as\n                `bokeh.palettes.Cividis`.\n\n        '
        contour_renderer = from_contour(x, y, z, levels, **visuals)
        self.renderers.append(contour_renderer)
        return contour_renderer

def markers():
    if False:
        while True:
            i = 10
    ' Prints a list of valid marker types for scatter()\n\n    Returns:\n        None\n    '
    print('Available markers: \n\n - ' + '\n - '.join(list(MarkerType)))
    print()
    print('Shortcuts: \n\n' + '\n'.join((f' {short!r}: {name}' for (short, name) in _MARKER_SHORTCUTS.items())))

class BaseFigureOptions(Options):
    tools = Either(String, Seq(Either(String, Instance(Tool))), default=DEFAULT_TOOLS, help='\n    Tools the plot should start with.\n    ')
    x_minor_ticks = Either(Auto, Int, default='auto', help='\n    Number of minor ticks between adjacent x-axis major ticks.\n    ')
    y_minor_ticks = Either(Auto, Int, default='auto', help='\n    Number of minor ticks between adjacent y-axis major ticks.\n    ')
    x_axis_location = Nullable(Enum(VerticalLocation), default='below', help='\n    Where the x-axis should be located.\n    ')
    y_axis_location = Nullable(Enum(HorizontalLocation), default='left', help='\n    Where the y-axis should be located.\n    ')
    x_axis_label = Nullable(TextLike, default='', help='\n    A label for the x-axis.\n    ')
    y_axis_label = Nullable(TextLike, default='', help='\n    A label for the y-axis.\n    ')
    active_drag = Nullable(Either(Auto, String, Instance(Drag)), default='auto', help='\n    Which drag tool should initially be active.\n    ')
    active_inspect = Nullable(Either(Auto, String, Instance(InspectTool), Seq(Instance(InspectTool))), default='auto', help='\n    Which drag tool should initially be active.\n    ')
    active_scroll = Nullable(Either(Auto, String, Instance(Scroll)), default='auto', help='\n    Which scroll tool should initially be active.\n    ')
    active_tap = Nullable(Either(Auto, String, Instance(Tap)), default='auto', help='\n    Which tap tool should initially be active.\n    ')
    active_multi = Nullable(Either(Auto, String, Instance(GestureTool)), default='auto', help='\n    Specify an active multi-gesture tool, for instance an edit tool or a range tool.\n    ')
    tooltips = Nullable(Either(Instance(Template), String, List(Tuple(String, String))), help='\n    An optional argument to configure tooltips for the Figure. This argument\n    accepts the same values as the ``HoverTool.tooltips`` property. If a hover\n    tool is specified in the ``tools`` argument, this value will override that\n    hover tools ``tooltips`` value. If no hover tool is specified in the\n    ``tools`` argument, then passing tooltips here will cause one to be created\n    and added.\n    ')
RangeLike = Either(Instance(Range), Either(Tuple(Float, Float), Tuple(Datetime, Datetime), Tuple(TimeDelta, TimeDelta)), Seq(String), Object('pandas.Series'), Object('pandas.core.groupby.GroupBy'))
AxisType = Nullable(Either(Auto, Enum('linear', 'log', 'datetime', 'mercator')))

class FigureOptions(BaseFigureOptions):
    x_range = RangeLike(default=InstanceDefault(DataRange1d), help='\n    Customize the x-range of the plot.\n    ')
    y_range = RangeLike(default=InstanceDefault(DataRange1d), help='\n    Customize the y-range of the plot.\n    ')
    x_axis_type = AxisType(default='auto', help='\n    The type of the x-axis.\n    ')
    y_axis_type = AxisType(default='auto', help='\n    The type of the y-axis.\n    ')
_color_fields = {'color', 'fill_color', 'line_color'}
_alpha_fields = {'alpha', 'fill_alpha', 'line_alpha'}