from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
if TYPE_CHECKING:
    from ..models.coordinates import CoordinateMapping
    from ..models.plots import Plot
    from ..models.renderers import GlyphRenderer
__all__ = ('GlyphAPI',)

class GlyphAPI:
    """ """

    @property
    def plot(self) -> Plot | None:
        if False:
            while True:
                i = 10
        return self._parent

    @property
    def coordinates(self) -> CoordinateMapping | None:
        if False:
            i = 10
            return i + 15
        return self._coordinates

    def __init__(self, parent: Plot | None=None, coordinates: CoordinateMapping | None=None) -> None:
        if False:
            return 10
        self._parent = parent
        self._coordinates = coordinates

    @glyph_method(glyphs.AnnularWedge)
    def annular_wedge(self, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        pass

    @glyph_method(glyphs.Annulus)
    def annulus(self, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.annulus(x=[1, 2, 3], y=[1, 2, 3], color="#7FC97F",\n                     inner_radius=0.2, outer_radius=0.5)\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Arc)
    def arc(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        pass

    @marker_method()
    def asterisk(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.asterisk(x=[1,2,3], y=[1,2,3], size=20, color="#F0027F")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Bezier)
    def bezier(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        pass

    @glyph_method(glyphs.Circle)
    def _circle(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        pass

    def circle(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        ' Configure and add :class:`~bokeh.models.glyphs.Circle` glyphs to this figure.\n\n        Args:\n            x (str or seq[float]) : values or field names of center x coordinates\n\n            y (str or seq[float]) : values or field names of center y coordinates\n\n            radius (str or list[float]) : values or field names of radii in |data units|\n\n            color (color value, optional): shorthand to set both fill and line color\n\n            source (:class:`~bokeh.models.sources.ColumnDataSource`) : a user-supplied data source.\n                An attempt will be made to convert the object to :class:`~bokeh.models.sources.ColumnDataSource`\n                if needed. If none is supplied, one is created for the user automatically.\n\n            **kwargs: |line properties| and |fill properties|\n\n        Examples:\n\n            .. code-block:: python\n\n                from bokeh.plotting import figure, show\n\n                plot = figure(width=300, height=300)\n                plot.circle(x=[1, 2, 3], y=[1, 2, 3], radius=0.2)\n\n                show(plot)\n\n        '
        if 'size' in kwargs:
            if 'radius' in kwargs:
                raise ValueError('Can only provide one of size or radius')
            deprecated((3, 3, 0), 'circle() method with size value', 'scatter(size=...) instead')
            return self.scatter(*args, **kwargs)
        else:
            return self._circle(*args, **kwargs)

    @glyph_method(glyphs.Block)
    def block(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.block(x=[1, 2, 3], y=[1,2,3], width=0.5, height=1, , color="#CAB2D6")\n\n        show(plot)\n\n'

    @marker_method()
    def circle_cross(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.circle_cross(x=[1,2,3], y=[4,5,6], size=20,\n                          color="#FB8072", fill_alpha=0.2, line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def circle_dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.circle_dot(x=[1,2,3], y=[4,5,6], size=20,\n                        color="#FB8072", fill_color=None)\n\n        show(plot)\n\n'

    @marker_method()
    def circle_x(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.circle_x(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                      color="#DD1C77", fill_alpha=0.2)\n\n        show(plot)\n\n'

    @marker_method()
    def circle_y(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.circle_y(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                      color="#DD1C77", fill_alpha=0.2)\n\n        show(plot)\n\n'

    @marker_method()
    def cross(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.cross(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                   color="#E6550D", line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def dash(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.dash(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                  color="#99D594", line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def diamond(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.diamond(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                     color="#1C9099", line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def diamond_cross(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.diamond_cross(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                           color="#386CB0", fill_color=None, line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def diamond_dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.diamond_dot(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                         color="#386CB0", fill_color=None)\n\n        show(plot)\n\n'

    @marker_method()
    def dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.dot(x=[1, 2, 3], y=[1, 2, 3], size=20, color="#386CB0")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.HArea)
    def harea(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.harea(x1=[0, 0, 0], x2=[1, 4, 2], y=[1, 2, 3],\n                   fill_color="#99D594")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.HAreaStep)
    def harea_step(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.harea_step(x1=[1, 2, 3], x2=[0, 0, 0], y=[1, 4, 2],\n                        step_mode="after", fill_color="#99D594")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.HBar)
    def hbar(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.hbar(y=[1, 2, 3], height=0.5, left=0, right=[1,2,3], color="#CAB2D6")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.HSpan)
    def hspan(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300, x_range=(0, 1))\n        plot.hspan(y=[1, 2, 3], color="#CAB2D6")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.HStrip)
    def hstrip(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300, x_range=(0, 1))\n        plot.hstrip(y0=[1, 2, 5], y1=[3, 4, 8], color="#CAB2D6")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Ellipse)
    def ellipse(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.ellipse(x=[1, 2, 3], y=[1, 2, 3], width=30, height=20,\n                     color="#386CB0", fill_color=None, line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def hex(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.hex(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,30], color="#74ADD1")\n\n        show(plot)\n\n'

    @marker_method()
    def hex_dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.hex_dot(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,30],\n                     color="#74ADD1", fill_color=None)\n\n        show(plot)\n\n'

    @glyph_method(glyphs.HexTile)
    def hex_tile(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300, match_aspect=True)\n        plot.hex_tile(r=[0, 0, 1], q=[1, 2, 2], fill_color="#74ADD1")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Image)
    def image(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\n.. note::\n    If both ``palette`` and ``color_mapper`` are passed, a ``ValueError``\n    exception will be raised. If neither is passed, then the ``Greys9``\n    palette will be used as a default.\n\n'

    @glyph_method(glyphs.ImageRGBA)
    def image_rgba(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\n.. note::\n    The ``image_rgba`` method accepts images as a two-dimensional array of RGBA\n    values (encoded as 32-bit integers).\n\n'

    @glyph_method(glyphs.ImageStack)
    def image_stack(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        pass

    @glyph_method(glyphs.ImageURL)
    def image_url(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        pass

    @marker_method()
    def inverted_triangle(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.inverted_triangle(x=[1, 2, 3], y=[1, 2, 3], size=20, color="#DE2D26")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Line)
    def line(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        p = figure(title="line", width=300, height=300)\n        p.line(x=[1, 2, 3, 4, 5], y=[6, 7, 2, 4, 5])\n\n        show(p)\n\n'

    @glyph_method(glyphs.MultiLine)
    def multi_line(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\n.. note::\n    For this glyph, the data is not simply an array of scalars, it is an\n    "array of arrays".\n\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        p = figure(width=300, height=300)\n        p.multi_line(xs=[[1, 2, 3], [2, 3, 4]], ys=[[6, 7, 2], [4, 5, 7]],\n                    color=[\'red\',\'green\'])\n\n        show(p)\n\n'

    @glyph_method(glyphs.MultiPolygons)
    def multi_polygons(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        "\n.. note::\n    For this glyph, the data is not simply an array of scalars, it is a\n    nested array.\n\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        p = figure(width=300, height=300)\n        p.multi_polygons(xs=[[[[1, 1, 2, 2]]], [[[1, 1, 3], [1.5, 1.5, 2]]]],\n                        ys=[[[[4, 3, 3, 4]]], [[[1, 3, 1], [1.5, 2, 1.5]]]],\n                        color=['red', 'green'])\n        show(p)\n\n"

    @glyph_method(glyphs.Patch)
    def patch(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        p = figure(width=300, height=300)\n        p.patch(x=[1, 2, 3, 2], y=[6, 7, 2, 2], color="#99d8c9")\n\n        show(p)\n\n'

    @glyph_method(glyphs.Patches)
    def patches(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\n.. note::\n    For this glyph, the data is not simply an array of scalars, it is an\n    "array of arrays".\n\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        p = figure(width=300, height=300)\n        p.patches(xs=[[1,2,3],[4,5,6,5]], ys=[[1,2,1],[4,5,5,4]],\n                  color=["#43a2ca", "#a8ddb5"])\n\n        show(p)\n\n'

    @marker_method()
    def plus(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.plus(x=[1, 2, 3], y=[1, 2, 3], size=20, color="#DE2D26")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Quad)
    def quad(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.quad(top=[2, 3, 4], bottom=[1, 2, 3], left=[1, 2, 3],\n                  right=[1.2, 2.5, 3.7], color="#B3DE69")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Quadratic)
    def quadratic(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        pass

    @glyph_method(glyphs.Ray)
    def ray(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.ray(x=[1, 2, 3], y=[1, 2, 3], length=45, angle=-0.7, color="#FB8072",\n                line_width=2)\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Rect)
    def rect(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.rect(x=[1, 2, 3], y=[1, 2, 3], width=10, height=20, color="#CAB2D6",\n                  width_units="screen", height_units="screen")\n\n        show(plot)\n\n    .. warning::\n        ``Rect`` glyphs are not well defined on logarithmic scales. Use\n        :class:`~bokeh.models.Block` or :class:`~bokeh.models.Quad` glyphs\n        instead.\n\n'

    @glyph_method(glyphs.Step)
    def step(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.step(x=[1, 2, 3, 4, 5], y=[1, 2, 3, 2, 5], color="#FB8072")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Segment)
    def segment(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.segment(x0=[1, 2, 3], y0=[1, 2, 3],\n                     x1=[1, 2, 3], y1=[1.2, 2.5, 3.7],\n                     color="#F4A582", line_width=3)\n\n        show(plot)\n\n'

    @marker_method()
    def square(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.square(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,30], color="#74ADD1")\n\n        show(plot)\n\n'

    @marker_method()
    def square_cross(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.square_cross(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                          color="#7FC97F",fill_color=None, line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def square_dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.square_dot(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                        color="#7FC97F", fill_color=None)\n\n        show(plot)\n\n'

    @marker_method()
    def square_pin(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.square_pin(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                        color="#7FC97F",fill_color=None, line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def square_x(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.square_x(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                      color="#FDAE6B",fill_color=None, line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def star(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.star(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                  color="#1C9099", line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def star_dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.star_dot(x=[1, 2, 3], y=[1, 2, 3], size=20,\n                      color="#386CB0", fill_color=None, line_width=2)\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Text)
    def text(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\n.. note::\n    The location and angle of the text relative to the ``x``, ``y`` coordinates\n    is indicated by the alignment and baseline text properties.\n\n'

    @marker_method()
    def triangle(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.triangle(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                      color="#99D594", line_width=2)\n\n        show(plot)\n\n'

    @marker_method()
    def triangle_dot(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.triangle_dot(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                          color="#99D594", fill_color=None)\n\n        show(plot)\n\n'

    @marker_method()
    def triangle_pin(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.triangle_pin(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],\n                      color="#99D594", line_width=2)\n\n        show(plot)\n\n'

    @glyph_method(glyphs.VArea)
    def varea(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            while True:
                i = 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.varea(x=[1, 2, 3], y1=[0, 0, 0], y2=[1, 4, 2],\n                   fill_color="#99D594")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.VAreaStep)
    def varea_step(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.varea_step(x=[1, 2, 3], y1=[0, 0, 0], y2=[1, 4, 2],\n                        step_mode="after", fill_color="#99D594")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.VBar)
    def vbar(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.vbar(x=[1, 2, 3], width=0.5, bottom=0, top=[1,2,3], color="#CAB2D6")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.VSpan)
    def vspan(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300, y_range=(0, 1))\n        plot.vspan(x=[1, 2, 3], color="#CAB2D6")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.VStrip)
    def vstrip(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300, y_range=(0, 1))\n        plot.vstrip(x0=[1, 2, 5], x1=[3, 4, 8], color="#CAB2D6")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Wedge)
    def wedge(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.wedge(x=[1, 2, 3], y=[1, 2, 3], radius=15, start_angle=0.6,\n                   end_angle=4.1, radius_units="screen", color="#2b8cbe")\n\n        show(plot)\n\n'

    @marker_method()
    def x(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            i = 10
            return i + 15
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.x(x=[1, 2, 3], y=[1, 2, 3], size=[10, 20, 25], color="#fa9fb5")\n\n        show(plot)\n\n'

    @marker_method()
    def y(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            return 10
        '\nExamples:\n\n    .. code-block:: python\n\n        from bokeh.plotting import figure, output_file, show\n\n        plot = figure(width=300, height=300)\n        plot.y(x=[1, 2, 3], y=[1, 2, 3], size=20, color="#DE2D26")\n\n        show(plot)\n\n'

    @glyph_method(glyphs.Scatter)
    def _scatter(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            for i in range(10):
                print('nop')
        pass

    def scatter(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
        if False:
            print('Hello World!')
        ' Creates a scatter plot of the given x and y items.\n\n        Args:\n            x (str or seq[float]) : values or field names of center x coordinates\n\n            y (str or seq[float]) : values or field names of center y coordinates\n\n            size (str or list[float]) : values or field names of sizes in |screen units|\n\n            marker (str, or list[str]): values or field names of marker types\n\n            color (color value, optional): shorthand to set both fill and line color\n\n            source (:class:`~bokeh.models.sources.ColumnDataSource`) : a user-supplied data source.\n                An attempt will be made to convert the object to :class:`~bokeh.models.sources.ColumnDataSource`\n                if needed. If none is supplied, one is created for the user automatically.\n\n            **kwargs: |line properties| and |fill properties|\n\n        Examples:\n\n            >>> p.scatter([1,2,3],[4,5,6], marker="square", fill_color="red")\n            >>> p.scatter("data1", "data2", marker="mtype", source=data_source, ...)\n\n        .. note::\n            ``Scatter`` markers with multiple marker types may be drawn in a\n            different order when using the WebGL output backend. This is an explicit\n            trade-off made in the interests of performance.\n\n        '
        marker_type = kwargs.pop('marker', 'circle')
        if isinstance(marker_type, str) and marker_type in _MARKER_SHORTCUTS:
            marker_type = _MARKER_SHORTCUTS[marker_type]
        if marker_type == 'circle' and 'radius' in kwargs:
            if 'size' in kwargs:
                raise ValueError('Can only provide one of size or radius')
            deprecated((3, 3, 0), 'scatter(radius=...)', 'circle(radius=...) instead')
            return self.circle(*args, **kwargs)
        else:
            return self._scatter(*args, marker=marker_type, **kwargs)
_MARKER_SHORTCUTS = {'*': 'asterisk', '+': 'cross', 'o': 'circle', 'o+': 'circle_cross', 'o.': 'circle_dot', 'ox': 'circle_x', 'oy': 'circle_y', '-': 'dash', '.': 'dot', 'v': 'inverted_triangle', '^': 'triangle', '^.': 'triangle_dot'}