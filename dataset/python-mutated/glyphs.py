""" Display a variety of visual shapes whose attributes can be associated
with data columns from ``ColumnDataSources``.



The full list of glyphs is below:

.. toctree::
   :maxdepth: 1
   :glob:

   glyphs/*

All glyphs share a minimal common interface through the base class ``Glyph``:

.. bokeh-model:: Glyph
    :module: bokeh.models.glyphs

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.enums import Direction, ImageOrigin, Palette, StepMode, enumeration
from ..core.has_props import abstract
from ..core.properties import AngleSpec, Bool, DataSpec, DistanceSpec, Enum, Float, Include, Instance, InstanceDefault, Int, MarkerSpec, NullDistanceSpec, NumberSpec, Override, Size, SizeSpec, String, StringSpec, field, value
from ..core.property_aliases import Anchor, BorderRadius, Padding, TextAnchor
from ..core.property_mixins import FillProps, HatchProps, ImageProps, LineProps, ScalarFillProps, ScalarHatchProps, ScalarLineProps, TextProps
from .glyph import ConnectedXYGlyph, FillGlyph, Glyph, HatchGlyph, LineGlyph, TextGlyph, XYGlyph
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
__all__ = ('AnnularWedge', 'Annulus', 'Arc', 'Bezier', 'Block', 'Circle', 'ConnectedXYGlyph', 'Ellipse', 'Glyph', 'HArea', 'HAreaStep', 'HBar', 'HSpan', 'HStrip', 'HexTile', 'Image', 'ImageRGBA', 'ImageStack', 'ImageURL', 'Line', 'Marker', 'MultiLine', 'MultiPolygons', 'Patch', 'Patches', 'Quad', 'Quadratic', 'Ray', 'Rect', 'Scatter', 'Segment', 'Step', 'Text', 'VArea', 'VAreaStep', 'VBar', 'VSpan', 'VStrip', 'Wedge', 'XYGlyph')

@abstract
class Marker(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Base class for glyphs that are simple markers with line and
    fill properties, located at an (x, y) location with a specified
    size.

    .. note::
        For simplicity, all markers have both line and fill properties
        declared, however some marker types (`asterisk`, `cross`, `x`)
        only draw lines. For these markers, the fill values are simply
        ignored.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    _args = ('x', 'y', 'size', 'angle')
    x = NumberSpec(default=field('x'), help='\n    The x-axis coordinates for the center of the markers.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-axis coordinates for the center of the markers.\n    ')
    hit_dilation = Size(default=1.0, help='\n    The factor by which to dilate the hit radius\n    which is responsible for defining the range in which a\n    marker responds to interactions with the Hover and Tap\n    tools.\n    ')
    size = SizeSpec(default=4, help='\n    The size (diameter) values for the markers in screen space units.\n    ')
    angle = AngleSpec(default=0.0, help='\n    The angles to rotate the markers.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the markers.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the markers.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the markers.\n    ')

@abstract
class LRTBGlyph(LineGlyph, FillGlyph, HatchGlyph):
    """ Base class for axis-aligned rectangles. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    border_radius = BorderRadius(default=0, help='\n    Allows the box to have rounded corners.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')

class AnnularWedge(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render annular wedges.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/AnnularWedge.py'
    _args = ('x', 'y', 'inner_radius', 'outer_radius', 'start_angle', 'end_angle', 'direction')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the center of the annular wedges.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the center of the annular wedges.\n    ')
    inner_radius = DistanceSpec(default=field('inner_radius'), help='\n    The inner radii of the annular wedges.\n    ')
    outer_radius = DistanceSpec(default=field('outer_radius'), help='\n    The outer radii of the annular wedges.\n    ')
    start_angle = AngleSpec(default=field('start_angle'), help='\n    The angles to start the annular wedges, as measured from the horizontal.\n    ')
    end_angle = AngleSpec(default=field('end_angle'), help='\n    The angles to end the annular wedges, as measured from the horizontal.\n    ')
    direction = Enum(Direction, default=Direction.anticlock, help='\n    Which direction to stroke between the start and end angles.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the annular wedges.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the annular wedges.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the annular wedges.\n    ')

class Annulus(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render annuli.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Annulus.py'
    _args = ('x', 'y', 'inner_radius', 'outer_radius')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the center of the annuli.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the center of the annuli.\n    ')
    inner_radius = DistanceSpec(default=field('inner_radius'), help='\n    The inner radii of the annuli.\n    ')
    outer_radius = DistanceSpec(default=field('outer_radius'), help='\n    The outer radii of the annuli.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the annuli.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the annuli.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the annuli.\n    ')

class Arc(XYGlyph, LineGlyph):
    """ Render arcs.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Arc.py'
    _args = ('x', 'y', 'radius', 'start_angle', 'end_angle', 'direction')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the center of the arcs.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the center of the arcs.\n    ')
    radius = DistanceSpec(default=field('radius'), help='\n    Radius of the arc.\n    ')
    start_angle = AngleSpec(default=field('start_angle'), help='\n    The angles to start the arcs, as measured from the horizontal.\n    ')
    end_angle = AngleSpec(default=field('end_angle'), help='\n    The angles to end the arcs, as measured from the horizontal.\n    ')
    direction = Enum(Direction, default='anticlock', help='\n    Which direction to stroke between the start and end angles.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the arcs.\n    ')

class Bezier(LineGlyph):
    """ Render Bezier curves.

    For more information consult the `Wikipedia article for Bezier curve`_.

    .. _Wikipedia article for Bezier curve: http://en.wikipedia.org/wiki/Bezier_curve

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Bezier.py'
    _args = ('x0', 'y0', 'x1', 'y1', 'cx0', 'cy0', 'cx1', 'cy1')
    x0 = NumberSpec(default=field('x0'), help='\n    The x-coordinates of the starting points.\n    ')
    y0 = NumberSpec(default=field('y0'), help='\n    The y-coordinates of the starting points.\n    ')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates of the ending points.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates of the ending points.\n    ')
    cx0 = NumberSpec(default=field('cx0'), help='\n    The x-coordinates of first control points.\n    ')
    cy0 = NumberSpec(default=field('cy0'), help='\n    The y-coordinates of first control points.\n    ')
    cx1 = NumberSpec(default=field('cx1'), help='\n    The x-coordinates of second control points.\n    ')
    cy1 = NumberSpec(default=field('cy1'), help='\n    The y-coordinates of second control points.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the Bezier curves.\n    ')

class Block(LRTBGlyph):
    """ Render rectangular regions, given a corner coordinate, width, and height.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Block.py'
    _args = ('x', 'y', 'width', 'height')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the centers of the blocks.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the centers of the blocks.\n    ')
    width = NumberSpec(default=1, help='\n    The widths of the blocks.\n    ')
    height = NumberSpec(default=1, help='\n    The heights of the blocks.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the blocks.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the blocks.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the blocks.\n    ')

class Circle(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render circle markers. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Circle.py'
    _args = ('x', 'y', 'radius')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the center of the annuli.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the center of the annuli.\n    ')
    radius = DistanceSpec(default=field('radius'), help='\n    The radius values for circle markers (in |data units|, by default).\n\n    .. warning::\n        Note that ``Circle`` glyphs are always drawn as circles on the screen,\n        even in cases where the data space aspect ratio is not 1-1. In all\n        cases where radius values are specified, the "distance" for the radius\n        is measured along the dimension specified by ``radius_dimension``. If\n        the aspect ratio is very large or small, the drawn circles may appear\n        much larger or smaller than expected. See :bokeh-issue:`626` for more\n        information.\n\n    ')
    radius_dimension = Enum(enumeration('x', 'y', 'max', 'min'), help='\n    What dimension to measure circle radii along.\n\n    When the data space aspect ratio is not 1-1, then the size of the drawn\n    circles depends on what direction is used to measure the "distance" of\n    the radius. This property allows that direction to be controlled.\n\n    Setting this dimension to \'max\' will calculate the radius on both the x\n    and y dimensions and use the maximum of the two, \'min\' selects the minimum.\n    ')
    hit_dilation = Size(default=1.0, help='\n    The factor by which to dilate the hit radius which is responsible for\n    defining the range in which a glyph responds to interactions with the\n    hover and tap tools.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the markers.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the markers.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the markers.\n    ')

class Ellipse(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render ellipses.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Ellipse.py'
    _args = ('x', 'y', 'width', 'height', 'angle')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the centers of the ellipses.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the centers of the ellipses.\n    ')
    width = DistanceSpec(default=field('width'), help='\n    The widths of each ellipse.\n    ')
    height = DistanceSpec(default=field('height'), help='\n    The heights of each ellipse.\n    ')
    angle = AngleSpec(default=0.0, help='\n    The angle the ellipses are rotated from horizontal. [rad]\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the ellipses.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the ellipses.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the ellipses.\n    ')

class HArea(LineGlyph, FillGlyph, HatchGlyph):
    """ Render a horizontally directed area between two equal length sequences
    of x-coordinates with the same y-coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HArea.py'
    _args = ('x1', 'x2', 'y')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates for the points of one side of the area.\n    ')
    x2 = NumberSpec(default=field('x2'), help='\n    The x-coordinates for the points of the other side of the area.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates for the points of the area.\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the horizontal directed area.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the horizontal directed area.\n    ')

class HAreaStep(FillGlyph, HatchGlyph):
    """ Render a horizontally directed area between two equal length sequences
    of x-coordinates with the same y-coordinates using step lines.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HAreaStep.py'
    _args = ('x1', 'x2', 'y')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates for the points of one side of the area.\n    ')
    x2 = NumberSpec(default=field('x2'), help='\n    The x-coordinates for the points of the other side of the area.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates for the points of the area.\n    ')
    step_mode = Enum(StepMode, default='before', help='\n    Where the step "level" should be drawn in relation to the x and y\n    coordinates. The parameter can assume one of three values:\n\n    * ``before``: (default) Draw step levels before each y-coordinate (no step before the first point)\n    * ``after``:  Draw step levels after each y-coordinate (no step after the last point)\n    * ``center``: Draw step levels centered on each y-coordinate\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the horizontal directed area.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the horizontal directed area.\n    ')

class HBar(LRTBGlyph):
    """ Render horizontal bars, given a center coordinate, ``height`` and
    (``left``, ``right``) coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HBar.py'
    _args = ('y', 'height', 'right', 'left')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the centers of the horizontal bars.\n    ')
    height = NumberSpec(default=1, help='\n    The heights of the vertical bars.\n    ')
    left = NumberSpec(default=0, help='\n    The x-coordinates of the left edges.\n    ')
    right = NumberSpec(default=field('right'), help='\n    The x-coordinates of the right edges.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the horizontal bars.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the horizontal bars.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the horizontal bars.\n    ')

class HexTile(LineGlyph, FillGlyph, HatchGlyph):
    """ Render horizontal tiles on a regular hexagonal grid.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HexTile.py'
    _args = ('q', 'r')
    size = Float(1.0, help='\n    The radius (in |data units|) of the hex tiling.\n\n    The radius is always measured along the cartesian y-axis for "pointy_top"\n    orientation, and along the cartesian x-axis for "flat_top" orientation. If\n    the aspect ratio of the underlying cartesian system is not 1-1, then the\n    tiles may be "squished" in one direction. To ensure that the tiles are\n    always regular hexagons, consider setting the ``match_aspect`` property of\n    the plot to True.\n    ')
    aspect_scale = Float(default=1.0, help="\n    Match a plot's aspect ratio scaling.\n\n    Use this parameter to match the aspect ratio scaling of a plot when using\n    :class:`~bokeh.models.Plot.aspect_scale` with a value other than ``1.0``.\n\n    ")
    r = NumberSpec(default=field('r'), help='\n    The "row" axial coordinates of the tile centers.\n    ')
    q = NumberSpec(default=field('q'), help='\n    The "column" axial coordinates of the tile centers.\n    ')
    scale = NumberSpec(1.0, help='\n    A scale factor for individual tiles.\n    ')
    orientation = String(default='pointytop', help='\n    The orientation of the hex tiles.\n\n    Use ``"pointytop"`` to orient the tile so that a pointed corner is at the top. Use\n    ``"flattop"`` to orient the tile so that a flat side is at the top.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the hex tiles.\n    ')
    line_color = Override(default=None)
    fill_props = Include(FillProps, help='\n    The {prop} values for the hex tiles.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the hex tiles.\n    ')

@abstract
class ImageBase(XYGlyph):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates to locate the image anchors.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates to locate the image anchors.\n    ')
    dw = DistanceSpec(default=field('dw'), help='\n    The widths of the plot regions that the images will occupy.\n\n    .. note::\n        This is not the number of pixels that an image is wide.\n        That number is fixed by the image itself.\n    ')
    dh = DistanceSpec(default=field('dh'), help='\n    The height of the plot region that the image will occupy.\n\n    .. note::\n        This is not the number of pixels that an image is tall.\n        That number is fixed by the image itself.\n    ')
    image_props = Include(ImageProps, help='\n    The {prop} values for the images.\n    ')
    dilate = Bool(False, help='\n    Whether to always round fractional pixel locations in such a way\n    as to make the images bigger.\n\n    This setting may be useful if pixel rounding errors are causing\n    images to have a gap between them, when they should appear flush.\n    ')
    origin = Enum(ImageOrigin, default='bottom_left', help='\n    Defines the coordinate space of an image.\n    ')
    anchor = Anchor(default='bottom_left', help='\n    Position of the image should be anchored at the `x`, `y` coordinates.\n    ')

class Image(ImageBase):
    """ Render images given as scalar data together with a color mapper.

    In addition to the defined model properties, ``Image`` also can accept
    a keyword argument ``palette`` in place of an explicit ``color_mapper``.
    The value should be a list of colors, or the name of one of the built-in
    palettes in ``bokeh.palettes``. This palette will be used to automatically
    construct a ``ColorMapper`` model for the ``color_mapper`` property.

    If both ``palette`` and ``color_mapper`` are passed, a ``ValueError``
    exception will be raised. If neither is passed, then the ``Greys9``
    palette will be used as a default.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        if 'palette' in kwargs and 'color_mapper' in kwargs:
            raise ValueError("only one of 'palette' and 'color_mapper' may be specified")
        elif 'color_mapper' not in kwargs:
            palette = kwargs.pop('palette', 'Greys9')
            mapper = LinearColorMapper(palette)
            kwargs['color_mapper'] = mapper
        super().__init__(*args, **kwargs)
    _args = ('image', 'x', 'y', 'dw', 'dh', 'dilate')
    _extra_kws = {'palette': ('str or list[color value]', 'a palette to construct a value for the color mapper property from')}
    image = NumberSpec(default=field('image'), help='\n    The arrays of scalar data for the images to be colormapped.\n    ')
    color_mapper = Instance(ColorMapper, default=InstanceDefault(LinearColorMapper, palette='Greys9'), help='\n    A ``ColorMapper`` to use to map the scalar data from ``image``\n    into RGBA values for display.\n\n    The name of a palette from ``bokeh.palettes`` may also be set, in which\n    case a ``LinearColorMapper`` configured with the named palette wil be used.\n\n    .. note::\n        The color mapping step happens on the client.\n    ').accepts(Enum(Palette), lambda pal: LinearColorMapper(palette=pal))

class ImageRGBA(ImageBase):
    """ Render images given as RGBA data.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    _args = ('image', 'x', 'y', 'dw', 'dh', 'dilate')
    image = NumberSpec(default=field('image'), help='\n    The arrays of RGBA data for the images.\n    ')

class ImageStack(ImageBase):
    """ Render images given as 3D stacked arrays by flattening each stack into
    an RGBA image using a ``StackColorMapper``.

    The 3D arrays have shape (ny, nx, nstack) where ``nstack` is the number of
    stacks. The ``color_mapper`` produces an RGBA value for each of the
    (ny, nx) pixels by combining array values in the ``nstack`` direction.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    _args = ('image', 'x', 'y', 'dw', 'dh', 'dilate')
    image = NumberSpec(default=field('image'), help='\n    The 3D arrays of data for the images.\n    ')
    color_mapper = Instance(StackColorMapper, help='\n    ``ScalarColorMapper`` used to map the scalar data from ``image``\n    into RGBA values for display.\n\n    .. note::\n        The color mapping step happens on the client.\n    ')

class ImageURL(XYGlyph):
    """ Render images loaded from given URLs.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/ImageURL.py'
    _args = ('url', 'x', 'y', 'w', 'h', 'angle', 'dilate')
    url = StringSpec(default=field('url'), help='\n    The URLs to retrieve images from.\n\n    .. note::\n        The actual retrieving and loading of the images happens on\n        the client.\n    ')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates to locate the image anchors.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates to locate the image anchors.\n    ')
    w = NullDistanceSpec(help='\n    The width of the plot region that the image will occupy in data space.\n\n    The default value is ``None``, in which case the image will be displayed\n    at its actual image size (regardless of the units specified here).\n    ')
    h = NullDistanceSpec(help='\n    The height of the plot region that the image will occupy in data space.\n\n    The default value is ``None``, in which case the image will be displayed\n    at its actual image size (regardless of the units specified here).\n    ')
    angle = AngleSpec(default=0, help='\n    The angles to rotate the images, as measured from the horizontal.\n    ')
    global_alpha = NumberSpec(1.0, help='\n    An overall opacity that each image is rendered with (in addition\n    to any inherent alpha values in the image itself).\n    ')
    dilate = Bool(False, help='\n    Whether to always round fractional pixel locations in such a way\n    as to make the images bigger.\n\n    This setting may be useful if pixel rounding errors are causing\n    images to have a gap between them, when they should appear flush.\n    ')
    anchor = Anchor(default='top_left', help='\n    Position of the image should be anchored at the `x`, `y` coordinates.\n    ')
    retry_attempts = Int(0, help='\n    Number of attempts to retry loading the images from the specified URL.\n    Default is zero.\n    ')
    retry_timeout = Int(0, help='\n    Timeout (in ms) between retry attempts to load the image from the\n    specified URL. Default is zero ms.\n    ')

class Line(ConnectedXYGlyph, LineGlyph):
    """ Render a single line.

    The ``Line`` glyph is different from most other glyphs in that the vector
    of values only produces one glyph on the Plot.

    .. note::
        Due to limitations in the underlying HTML canvas, it is possible that a
        line is not drawn when one or more of its coordinates is very far outside
        the viewport. This behavior is different for different browsers. See
        :bokeh-issue:`11498` for more information.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    _args = ('x', 'y')
    __example__ = 'examples/reference/models/Line.py'
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates for the points of the line.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates for the points of the line.\n    ')
    line_props = Include(ScalarLineProps, help='\n    The {prop} values for the line.\n    ')

class MultiLine(LineGlyph):
    """ Render several lines.

    The data for the ``MultiLine`` glyph is different in that the vector of
    values is not a vector of scalars. Rather, it is a "list of lists".

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/MultiLine.py'
    _args = ('xs', 'ys')
    xs = NumberSpec(default=field('xs'), help='\n    The x-coordinates for all the lines, given as a "list of lists".\n    ')
    ys = NumberSpec(default=field('ys'), help='\n    The y-coordinates for all the lines, given as a "list of lists".\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the lines.\n    ')

class MultiPolygons(LineGlyph, FillGlyph, HatchGlyph):
    """ Render several MultiPolygon.

    Modeled on geoJSON - the data for the ``MultiPolygons`` glyph is
    different in that the vector of values is not a vector of scalars.
    Rather, it is a "list of lists of lists of lists".

    During box selection only multi-polygons entirely contained in the
    selection box will be included.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/MultiPolygons.py'
    _args = ('xs', 'ys')
    xs = NumberSpec(default=field('xs'), help='\n    The x-coordinates for all the patches, given as a nested list.\n\n    .. note::\n        Each item in ``MultiPolygons`` represents one MultiPolygon and each\n        MultiPolygon is comprised of ``n`` Polygons. Each Polygon is made of\n        one exterior ring optionally followed by ``m`` interior rings (holes).\n    ')
    ys = NumberSpec(default=field('ys'), help='\n    The y-coordinates for all the patches, given as a "list of lists".\n\n    .. note::\n        Each item in ``MultiPolygons`` represents one MultiPolygon and each\n        MultiPolygon is comprised of ``n`` Polygons. Each Polygon is made of\n        one exterior ring optionally followed by ``m`` interior rings (holes).\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the multipolygons.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the multipolygons.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the multipolygons.\n    ')

class Patch(ConnectedXYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render a single patch.

    The ``Patch`` glyph is different from most other glyphs in that the vector
    of values only produces one glyph on the Plot.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Patch.py'
    _args = ('x', 'y')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates for the points of the patch.\n\n    .. note::\n        A patch may comprise multiple polygons. In this case the\n        x-coordinates for each polygon should be separated by NaN\n        values in the sequence.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates for the points of the patch.\n\n    .. note::\n        A patch may comprise multiple polygons. In this case the\n        y-coordinates for each polygon should be separated by NaN\n        values in the sequence.\n    ')
    line_props = Include(ScalarLineProps, help='\n    The {prop} values for the patch.\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the patch.\n    ')
    hatch_props = Include(ScalarHatchProps, help='\n    The {prop} values for the patch.\n    ')

class Patches(LineGlyph, FillGlyph, HatchGlyph):
    """ Render several patches.

    The data for the ``Patches`` glyph is different in that the vector of
    values is not a vector of scalars. Rather, it is a "list of lists".

    During box selection only patches entirely contained in the
    selection box will be included.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Patches.py'
    _args = ('xs', 'ys')
    xs = NumberSpec(default=field('xs'), help='\n    The x-coordinates for all the patches, given as a "list of lists".\n\n    .. note::\n        Individual patches may comprise multiple polygons. In this case\n        the x-coordinates for each polygon should be separated by NaN\n        values in the sublists.\n    ')
    ys = NumberSpec(default=field('ys'), help='\n    The y-coordinates for all the patches, given as a "list of lists".\n\n    .. note::\n        Individual patches may comprise multiple polygons. In this case\n        the y-coordinates for each polygon should be separated by NaN\n        values in the sublists.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the patches.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the patches.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the patches.\n    ')

class Quad(LRTBGlyph):
    """ Render axis-aligned quads.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Quad.py'
    _args = ('left', 'right', 'top', 'bottom')
    left = NumberSpec(default=field('left'), help='\n    The x-coordinates of the left edges.\n    ')
    right = NumberSpec(default=field('right'), help='\n    The x-coordinates of the right edges.\n    ')
    bottom = NumberSpec(default=field('bottom'), help='\n    The y-coordinates of the bottom edges.\n    ')
    top = NumberSpec(default=field('top'), help='\n    The y-coordinates of the top edges.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the quads.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the quads.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the quads.\n    ')

class Quadratic(LineGlyph):
    """ Render parabolas.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Quadratic.py'
    _args = ('x0', 'y0', 'x1', 'y1', 'cx', 'cy')
    x0 = NumberSpec(default=field('x0'), help='\n    The x-coordinates of the starting points.\n    ')
    y0 = NumberSpec(default=field('y0'), help='\n    The y-coordinates of the starting points.\n    ')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates of the ending points.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates of the ending points.\n    ')
    cx = NumberSpec(default=field('cx'), help='\n    The x-coordinates of the control points.\n    ')
    cy = NumberSpec(default=field('cy'), help='\n    The y-coordinates of the control points.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the parabolas.\n    ')

class Ray(XYGlyph, LineGlyph):
    """ Render rays.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Ray.py'
    _args = ('x', 'y', 'length', 'angle')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates to start the rays.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates to start the rays.\n    ')
    angle = AngleSpec(default=0, help='\n    The angles in radians to extend the rays, as measured from the horizontal.\n    ')
    length = DistanceSpec(default=0, help='\n    The length to extend the ray. Note that this ``length`` defaults\n    to |data units| (measured in the x-direction).\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the rays.\n    ')

class Rect(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render rectangles, characterised by center position (x and y), width,
    height, and angle of rotation.

    .. warning::
        ``Rect`` glyphs are not well defined on logarithmic scales. Use
        :class:`~bokeh.models.Block` or :class:`~bokeh.models.Quad` glyphs
        instead.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Rect.py'
    _args = ('x', 'y', 'width', 'height', 'angle', 'dilate')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the centers of the rectangles.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the centers of the rectangles.\n    ')
    width = DistanceSpec(default=field('width'), help='\n    The overall widths of the rectangles.\n    ')
    height = DistanceSpec(default=field('height'), help='\n    The overall heights of the rectangles.\n    ')
    angle = AngleSpec(default=0.0, help='\n    The angles to rotate the rectangles, as measured from the horizontal.\n    ')
    border_radius = BorderRadius(default=0, help='\n    Allows the box to have rounded corners.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    dilate = Bool(False, help='\n    Whether to always round fractional pixel locations in such a way\n    as to make the rectangles bigger.\n\n    This setting may be useful if pixel rounding errors are causing\n    rectangles to have a gap between them, when they should appear\n    flush.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the rectangles.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the rectangles.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the rectangles.\n    ')

class Scatter(Marker):
    """ Render scatter markers selected from a predefined list of designs.

    Use ``Scatter`` to draw any of Bokeh's built-in marker types:
    ``asterisk``, ``circle``, ``circle_cross``, ``circle_dot``, ``circle_x``,
    ``circle_y``, ``cross``, ``dash``, ``diamond``, ``diamond_cross``,
    ``diamond_dot``, ``dot``, ``hex``, ``hex_dot``, ``inverted_triangle``,
    ``plus``, ``square``, ``square_cross``, ``square_dot``, ``square_pin``,
    ``square_x``, ``star``, ``star_dot``, ``triangle``, ``triangle_dot``,
    ``triangle_pin``, ``x``, or ``y``. This collection is available in
    :class:`~bokeh.core.enums.MarkerType`.

    Bokeh's built-in markers consist of a set of base markers, most of which can
    be combined with different kinds of additional visual features:

    .. bokeh-plot:: __REPO__/examples/basic/scatters/markertypes.py
        :source-position: none

    You can select marker types in two ways:

    * To draw the **same marker for all values**, use the ``marker`` attribute
      to specify the name of a specific marker. For example:

      .. code-block:: python

          glyph = Scatter(x="x", y="y", size="sizes", marker="square")
          plot.add_glyph(source, glyph)

      This will render square markers for all points.

    * Alternatively, to use **marker types specified in a data source column**,
      assign the column name to the ``marker`` attribute. For example:

      .. code-block:: python

          # source.data['markers'] = ["circle", "square", "circle", ... ]

          glyph = Scatter(x="x", y="y", size="sizes", marker="markers")
          plot.add_glyph(source, glyph)

    .. note::
        When you draw ``circle`` markers with ``Scatter``, you can only assign a
        size in |screen units| (by passing a number of pixels to the ``size``
        property). In case you want to define the radius of circles in
        |data units|, use the :class:`~bokeh.models.glyphs.Circle` glyph instead
        of the ``Scatter`` glyph.

    .. note::
        ``Scatter`` markers with multiple marker types may be drawn in a
        different order when using the WebGL output backend. This is an explicit
        trade-off made in the interests of performance.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Scatter.py'
    _args = ('x', 'y', 'size', 'angle', 'marker')
    marker = MarkerSpec(default='circle', help='\n    Which marker to render. This can be the name of any built in marker,\n    e.g. "circle", or a reference to a data column containing such names.\n    ')

class Segment(LineGlyph):
    """ Render segments.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Segment.py'
    _args = ('x0', 'y0', 'x1', 'y1')
    x0 = NumberSpec(default=field('x0'), help='\n    The x-coordinates of the starting points.\n    ')
    y0 = NumberSpec(default=field('y0'), help='\n    The y-coordinates of the starting points.\n    ')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates of the ending points.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates of the ending points.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the segments.\n    ')

class Step(XYGlyph, LineGlyph):
    """ Render step lines.

    Step levels can be draw before, after, or centered on each point, according
    to the value of the ``mode`` property.

    The x-coordinates are assumed to be (and must be) sorted in ascending order
    for steps to be properly rendered.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Step.py'
    _args = ('x', 'y')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates for the steps.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates for the steps.\n    ')
    line_props = Include(ScalarLineProps, help='\n    The {prop} values for the steps.\n    ')
    mode = Enum(StepMode, default='before', help='\n    Where the step "level" should be drawn in relation to the x and y\n    coordinates. The parameter can assume one of three values:\n\n    * ``before``: (default) Draw step levels before each x-coordinate (no step before the first point)\n    * ``after``:  Draw step levels after each x-coordinate (no step after the last point)\n    * ``center``: Draw step levels centered on each x-coordinate\n    ')

class Text(XYGlyph, TextGlyph):
    """ Render text.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Text.py'
    _args = ('x', 'y', 'text', 'angle', 'x_offset', 'y_offset')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates to locate the text anchors.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates to locate the text anchors.\n    ')
    text = StringSpec(default=field('text'), help='\n    The text values to render.\n    ')
    angle = AngleSpec(default=0, help='\n    The angles to rotate the text, as measured from the horizontal.\n    ')
    x_offset = NumberSpec(default=0, help='\n    Offset values in pixels to apply to the x-coordinates.\n\n    This is useful, for instance, if it is desired to "float" text a fixed\n    distance in |screen units| from a given data position.\n    ')
    y_offset = NumberSpec(default=0, help='\n    Offset values in pixels to apply to the y-coordinates.\n\n    This is useful, for instance, if it is desired to "float" text a fixed\n    distance in |screen units| from a given data position.\n    ')
    anchor = DataSpec(TextAnchor, default=value('auto'), help='\n    Position within the bounding box of this glyph to which ``x`` and ``y``\n    coordinates are anchored to. This can be a named anchor point like\n    ``top_left`` or ``center``, or a percentage from from left to right\n    and top to bottom, or a combination of those, independently in width\n    and height. If set to ``auto``, then anchor point will be determined\n    from text ``align`` and ``baseline``.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    padding = Padding(default=0, help='\n    Extra space between the text of a glyphs and its bounding box (border).\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    border_radius = BorderRadius(default=0, help='\n    Allows the box to have rounded corners. For the best results, it\n    should be used in combination with ``padding``.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')
    text_props = Include(TextProps, help='\n    The {prop} values for the text.\n    ')
    background_fill_props = Include(FillProps, prefix='background', help='\n    The {prop} values for the text bounding box.\n    ')
    background_hatch_props = Include(HatchProps, prefix='background', help='\n    The {prop} values for the text bounding box.\n    ')
    border_line_props = Include(LineProps, prefix='border', help='\n    The {prop} values for the text bounding box.\n    ')
    background_fill_color = Override(default=None)
    background_hatch_color = Override(default=None)
    border_line_color = Override(default=None)

class VArea(FillGlyph, HatchGlyph):
    """ Render a vertically directed area between two equal length sequences
    of y-coordinates with the same x-coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VArea.py'
    _args = ('x', 'y1', 'y2')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates for the points of the area.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates for the points of one side of the area.\n    ')
    y2 = NumberSpec(default=field('y2'), help='\n    The y-coordinates for the points of the other side of the area.\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the vertical directed area.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the vertical directed area.\n    ')

class VAreaStep(FillGlyph, HatchGlyph):
    """ Render a vertically directed area between two equal length sequences
    of y-coordinates with the same x-coordinates using step lines.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VAreaStep.py'
    _args = ('x', 'y1', 'y2')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates for the points of the area.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates for the points of one side of the area.\n    ')
    y2 = NumberSpec(default=field('y2'), help='\n    The y-coordinates for the points of the other side of the area.\n    ')
    step_mode = Enum(StepMode, default='before', help='\n    Where the step "level" should be drawn in relation to the x and y\n    coordinates. The parameter can assume one of three values:\n\n    * ``before``: (default) Draw step levels before each x-coordinate (no step before the first point)\n    * ``after``:  Draw step levels after each x-coordinate (no step after the last point)\n    * ``center``: Draw step levels centered on each x-coordinate\n    ')
    fill_props = Include(ScalarFillProps, help='\n    The {prop} values for the vertical directed area.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the vertical directed area.\n    ')

class VBar(LRTBGlyph):
    """ Render vertical bars, given a center coordinate, width and (top, bottom) coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VBar.py'
    _args = ('x', 'width', 'top', 'bottom')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the centers of the vertical bars.\n    ')
    width = NumberSpec(default=1, help='\n    The widths of the vertical bars.\n    ')
    bottom = NumberSpec(default=0, help='\n    The y-coordinates of the bottom edges.\n    ')
    top = NumberSpec(default=field('top'), help='\n    The y-coordinates of the top edges.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the vertical bars.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the vertical bars.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the vertical bars.\n    ')

class Wedge(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render wedges.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Wedge.py'
    _args = ('x', 'y', 'radius', 'start_angle', 'end_angle', 'direction')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the points of the wedges.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the points of the wedges.\n    ')
    radius = DistanceSpec(default=field('radius'), help='\n    Radii of the wedges.\n    ')
    start_angle = AngleSpec(default=field('start_angle'), help='\n    The angles to start the wedges, as measured from the horizontal.\n    ')
    end_angle = AngleSpec(default=field('end_angle'), help='\n    The angles to end the wedges, as measured from the horizontal.\n    ')
    direction = Enum(Direction, default='anticlock', help='\n    Which direction to stroke between the start and end angles.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the wedges.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the wedges.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the wedges.\n    ')

class HSpan(LineGlyph):
    """ Horizontal lines of infinite width. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HSpan.py'
    _args = 'y'
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the spans.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the spans.\n    ')

class VSpan(LineGlyph):
    """ Vertical lines of infinite height. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VSpan.py'
    _args = 'x'
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the spans.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the spans.\n    ')

class HStrip(LineGlyph, FillGlyph, HatchGlyph):
    """ Horizontal strips of infinite width. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HStrip.py'
    _args = ('y0', 'y1')
    y0 = NumberSpec(default=field('y0'), help='\n    The y-coordinates of the coordinates of one side of the strips.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates of the coordinates of the other side of the strips.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the strips.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the strips.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the strips.\n    ')

class VStrip(LineGlyph, FillGlyph, HatchGlyph):
    """ Vertical strips of infinite height. """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VStrip.py'
    _args = ('x0', 'x1')
    x0 = NumberSpec(default=field('x0'), help='\n    The x-coordinates of the coordinates of one side of the strips.\n    ')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates of the coordinates of the other side of the strips.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the strips.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the strips.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the strips.\n    ')