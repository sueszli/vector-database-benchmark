"""
A collection of functions and objects for creating or placing inset axes.
"""
from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes

@_api.deprecated('3.8', alternative='Axes.inset_axes')
class InsetPosition:

    @_docstring.dedent_interpd
    def __init__(self, parent, lbwh):
        if False:
            i = 10
            return i + 15
        '\n        An object for positioning an inset axes.\n\n        This is created by specifying the normalized coordinates in the axes,\n        instead of the figure.\n\n        Parameters\n        ----------\n        parent : `~matplotlib.axes.Axes`\n            Axes to use for normalizing coordinates.\n\n        lbwh : iterable of four floats\n            The left edge, bottom edge, width, and height of the inset axes, in\n            units of the normalized coordinate of the *parent* axes.\n\n        See Also\n        --------\n        :meth:`matplotlib.axes.Axes.set_axes_locator`\n\n        Examples\n        --------\n        The following bounds the inset axes to a box with 20%% of the parent\n        axes height and 40%% of the width. The size of the axes specified\n        ([0, 0, 1, 1]) ensures that the axes completely fills the bounding box:\n\n        >>> parent_axes = plt.gca()\n        >>> ax_ins = plt.axes([0, 0, 1, 1])\n        >>> ip = InsetPosition(parent_axes, [0.5, 0.1, 0.4, 0.2])\n        >>> ax_ins.set_axes_locator(ip)\n        '
        self.parent = parent
        self.lbwh = lbwh

    def __call__(self, ax, renderer):
        if False:
            while True:
                i = 10
        bbox_parent = self.parent.get_position(original=False)
        trans = BboxTransformTo(bbox_parent)
        bbox_inset = Bbox.from_bounds(*self.lbwh)
        bb = TransformedBbox(bbox_inset, trans)
        return bb

class AnchoredLocatorBase(AnchoredOffsetbox):

    def __init__(self, bbox_to_anchor, offsetbox, loc, borderpad=0.5, bbox_transform=None):
        if False:
            while True:
                i = 10
        super().__init__(loc, pad=0.0, child=None, borderpad=borderpad, bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform)

    def draw(self, renderer):
        if False:
            while True:
                i = 10
        raise RuntimeError('No draw method should be called')

    def __call__(self, ax, renderer):
        if False:
            return 10
        if renderer is None:
            renderer = ax.figure._get_renderer()
        self.axes = ax
        bbox = self.get_window_extent(renderer)
        (px, py) = self.get_offset(bbox.width, bbox.height, 0, 0, renderer)
        bbox_canvas = Bbox.from_bounds(px, py, bbox.width, bbox.height)
        tr = ax.figure.transSubfigure.inverted()
        return TransformedBbox(bbox_canvas, tr)

class AnchoredSizeLocator(AnchoredLocatorBase):

    def __init__(self, bbox_to_anchor, x_size, y_size, loc, borderpad=0.5, bbox_transform=None):
        if False:
            while True:
                i = 10
        super().__init__(bbox_to_anchor, None, loc, borderpad=borderpad, bbox_transform=bbox_transform)
        self.x_size = Size.from_any(x_size)
        self.y_size = Size.from_any(y_size)

    def get_bbox(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        bbox = self.get_bbox_to_anchor()
        dpi = renderer.points_to_pixels(72.0)
        (r, a) = self.x_size.get_size(renderer)
        width = bbox.width * r + a * dpi
        (r, a) = self.y_size.get_size(renderer)
        height = bbox.height * r + a * dpi
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize
        return Bbox.from_bounds(0, 0, width, height).padded(pad)

class AnchoredZoomLocator(AnchoredLocatorBase):

    def __init__(self, parent_axes, zoom, loc, borderpad=0.5, bbox_to_anchor=None, bbox_transform=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent_axes = parent_axes
        self.zoom = zoom
        if bbox_to_anchor is None:
            bbox_to_anchor = parent_axes.bbox
        super().__init__(bbox_to_anchor, None, loc, borderpad=borderpad, bbox_transform=bbox_transform)

    def get_bbox(self, renderer):
        if False:
            return 10
        bb = self.parent_axes.transData.transform_bbox(self.axes.viewLim)
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize
        return Bbox.from_bounds(0, 0, abs(bb.width * self.zoom), abs(bb.height * self.zoom)).padded(pad)

class BboxPatch(Patch):

    @_docstring.dedent_interpd
    def __init__(self, bbox, **kwargs):
        if False:
            return 10
        '\n        Patch showing the shape bounded by a Bbox.\n\n        Parameters\n        ----------\n        bbox : `~matplotlib.transforms.Bbox`\n            Bbox to use for the extents of this patch.\n\n        **kwargs\n            Patch properties. Valid arguments include:\n\n            %(Patch:kwdoc)s\n        '
        if 'transform' in kwargs:
            raise ValueError('transform should not be set')
        kwargs['transform'] = IdentityTransform()
        super().__init__(**kwargs)
        self.bbox = bbox

    def get_path(self):
        if False:
            i = 10
            return i + 15
        (x0, y0, x1, y1) = self.bbox.extents
        return Path._create_closed([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

class BboxConnector(Patch):

    @staticmethod
    def get_bbox_edge_pos(bbox, loc):
        if False:
            while True:
                i = 10
        '\n        Return the ``(x, y)`` coordinates of corner *loc* of *bbox*; parameters\n        behave as documented for the `.BboxConnector` constructor.\n        '
        (x0, y0, x1, y1) = bbox.extents
        if loc == 1:
            return (x1, y1)
        elif loc == 2:
            return (x0, y1)
        elif loc == 3:
            return (x0, y0)
        elif loc == 4:
            return (x1, y0)

    @staticmethod
    def connect_bbox(bbox1, bbox2, loc1, loc2=None):
        if False:
            print('Hello World!')
        '\n        Construct a `.Path` connecting corner *loc1* of *bbox1* to corner\n        *loc2* of *bbox2*, where parameters behave as documented as for the\n        `.BboxConnector` constructor.\n        '
        if isinstance(bbox1, Rectangle):
            bbox1 = TransformedBbox(Bbox.unit(), bbox1.get_transform())
        if isinstance(bbox2, Rectangle):
            bbox2 = TransformedBbox(Bbox.unit(), bbox2.get_transform())
        if loc2 is None:
            loc2 = loc1
        (x1, y1) = BboxConnector.get_bbox_edge_pos(bbox1, loc1)
        (x2, y2) = BboxConnector.get_bbox_edge_pos(bbox2, loc2)
        return Path([[x1, y1], [x2, y2]])

    @_docstring.dedent_interpd
    def __init__(self, bbox1, bbox2, loc1, loc2=None, **kwargs):
        if False:
            return 10
        "\n        Connect two bboxes with a straight line.\n\n        Parameters\n        ----------\n        bbox1, bbox2 : `~matplotlib.transforms.Bbox`\n            Bounding boxes to connect.\n\n        loc1, loc2 : {1, 2, 3, 4}\n            Corner of *bbox1* and *bbox2* to draw the line. Valid values are::\n\n                'upper right'  : 1,\n                'upper left'   : 2,\n                'lower left'   : 3,\n                'lower right'  : 4\n\n            *loc2* is optional and defaults to *loc1*.\n\n        **kwargs\n            Patch properties for the line drawn. Valid arguments include:\n\n            %(Patch:kwdoc)s\n        "
        if 'transform' in kwargs:
            raise ValueError('transform should not be set')
        kwargs['transform'] = IdentityTransform()
        kwargs.setdefault('fill', bool({'fc', 'facecolor', 'color'}.intersection(kwargs)))
        super().__init__(**kwargs)
        self.bbox1 = bbox1
        self.bbox2 = bbox2
        self.loc1 = loc1
        self.loc2 = loc2

    def get_path(self):
        if False:
            i = 10
            return i + 15
        return self.connect_bbox(self.bbox1, self.bbox2, self.loc1, self.loc2)

class BboxConnectorPatch(BboxConnector):

    @_docstring.dedent_interpd
    def __init__(self, bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, **kwargs):
        if False:
            return 10
        "\n        Connect two bboxes with a quadrilateral.\n\n        The quadrilateral is specified by two lines that start and end at\n        corners of the bboxes. The four sides of the quadrilateral are defined\n        by the two lines given, the line between the two corners specified in\n        *bbox1* and the line between the two corners specified in *bbox2*.\n\n        Parameters\n        ----------\n        bbox1, bbox2 : `~matplotlib.transforms.Bbox`\n            Bounding boxes to connect.\n\n        loc1a, loc2a, loc1b, loc2b : {1, 2, 3, 4}\n            The first line connects corners *loc1a* of *bbox1* and *loc2a* of\n            *bbox2*; the second line connects corners *loc1b* of *bbox1* and\n            *loc2b* of *bbox2*.  Valid values are::\n\n                'upper right'  : 1,\n                'upper left'   : 2,\n                'lower left'   : 3,\n                'lower right'  : 4\n\n        **kwargs\n            Patch properties for the line drawn:\n\n            %(Patch:kwdoc)s\n        "
        if 'transform' in kwargs:
            raise ValueError('transform should not be set')
        super().__init__(bbox1, bbox2, loc1a, loc2a, **kwargs)
        self.loc1b = loc1b
        self.loc2b = loc2b

    def get_path(self):
        if False:
            while True:
                i = 10
        path1 = self.connect_bbox(self.bbox1, self.bbox2, self.loc1, self.loc2)
        path2 = self.connect_bbox(self.bbox2, self.bbox1, self.loc2b, self.loc1b)
        path_merged = [*path1.vertices, *path2.vertices, path1.vertices[0]]
        return Path(path_merged)

def _add_inset_axes(parent_axes, axes_class, axes_kwargs, axes_locator):
    if False:
        i = 10
        return i + 15
    'Helper function to add an inset axes and disable navigation in it.'
    if axes_class is None:
        axes_class = HostAxes
    if axes_kwargs is None:
        axes_kwargs = {}
    inset_axes = axes_class(parent_axes.figure, parent_axes.get_position(), **{'navigate': False, **axes_kwargs, 'axes_locator': axes_locator})
    return parent_axes.figure.add_axes(inset_axes)

@_docstring.dedent_interpd
def inset_axes(parent_axes, width, height, loc='upper right', bbox_to_anchor=None, bbox_transform=None, axes_class=None, axes_kwargs=None, borderpad=0.5):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create an inset axes with a given width and height.\n\n    Both sizes used can be specified either in inches or percentage.\n    For example,::\n\n        inset_axes(parent_axes, width=\'40%%\', height=\'30%%\', loc=\'lower left\')\n\n    creates in inset axes in the lower left corner of *parent_axes* which spans\n    over 30%% in height and 40%% in width of the *parent_axes*. Since the usage\n    of `.inset_axes` may become slightly tricky when exceeding such standard\n    cases, it is recommended to read :doc:`the examples\n    </gallery/axes_grid1/inset_locator_demo>`.\n\n    Notes\n    -----\n    The meaning of *bbox_to_anchor* and *bbox_to_transform* is interpreted\n    differently from that of legend. The value of bbox_to_anchor\n    (or the return value of its get_points method; the default is\n    *parent_axes.bbox*) is transformed by the bbox_transform (the default\n    is Identity transform) and then interpreted as points in the pixel\n    coordinate (which is dpi dependent).\n\n    Thus, following three calls are identical and creates an inset axes\n    with respect to the *parent_axes*::\n\n       axins = inset_axes(parent_axes, "30%%", "40%%")\n       axins = inset_axes(parent_axes, "30%%", "40%%",\n                          bbox_to_anchor=parent_axes.bbox)\n       axins = inset_axes(parent_axes, "30%%", "40%%",\n                          bbox_to_anchor=(0, 0, 1, 1),\n                          bbox_transform=parent_axes.transAxes)\n\n    Parameters\n    ----------\n    parent_axes : `matplotlib.axes.Axes`\n        Axes to place the inset axes.\n\n    width, height : float or str\n        Size of the inset axes to create. If a float is provided, it is\n        the size in inches, e.g. *width=1.3*. If a string is provided, it is\n        the size in relative units, e.g. *width=\'40%%\'*. By default, i.e. if\n        neither *bbox_to_anchor* nor *bbox_transform* are specified, those\n        are relative to the parent_axes. Otherwise, they are to be understood\n        relative to the bounding box provided via *bbox_to_anchor*.\n\n    loc : str, default: \'upper right\'\n        Location to place the inset axes.  Valid locations are\n        \'upper left\', \'upper center\', \'upper right\',\n        \'center left\', \'center\', \'center right\',\n        \'lower left\', \'lower center\', \'lower right\'.\n        For backward compatibility, numeric values are accepted as well.\n        See the parameter *loc* of `.Legend` for details.\n\n    bbox_to_anchor : tuple or `~matplotlib.transforms.BboxBase`, optional\n        Bbox that the inset axes will be anchored to. If None,\n        a tuple of (0, 0, 1, 1) is used if *bbox_transform* is set\n        to *parent_axes.transAxes* or *parent_axes.figure.transFigure*.\n        Otherwise, *parent_axes.bbox* is used. If a tuple, can be either\n        [left, bottom, width, height], or [left, bottom].\n        If the kwargs *width* and/or *height* are specified in relative units,\n        the 2-tuple [left, bottom] cannot be used. Note that,\n        unless *bbox_transform* is set, the units of the bounding box\n        are interpreted in the pixel coordinate. When using *bbox_to_anchor*\n        with tuple, it almost always makes sense to also specify\n        a *bbox_transform*. This might often be the axes transform\n        *parent_axes.transAxes*.\n\n    bbox_transform : `~matplotlib.transforms.Transform`, optional\n        Transformation for the bbox that contains the inset axes.\n        If None, a `.transforms.IdentityTransform` is used. The value\n        of *bbox_to_anchor* (or the return value of its get_points method)\n        is transformed by the *bbox_transform* and then interpreted\n        as points in the pixel coordinate (which is dpi dependent).\n        You may provide *bbox_to_anchor* in some normalized coordinate,\n        and give an appropriate transform (e.g., *parent_axes.transAxes*).\n\n    axes_class : `~matplotlib.axes.Axes` type, default: `.HostAxes`\n        The type of the newly created inset axes.\n\n    axes_kwargs : dict, optional\n        Keyword arguments to pass to the constructor of the inset axes.\n        Valid arguments include:\n\n        %(Axes:kwdoc)s\n\n    borderpad : float, default: 0.5\n        Padding between inset axes and the bbox_to_anchor.\n        The units are axes font size, i.e. for a default font size of 10 points\n        *borderpad = 0.5* is equivalent to a padding of 5 points.\n\n    Returns\n    -------\n    inset_axes : *axes_class*\n        Inset axes object created.\n    '
    if bbox_transform in [parent_axes.transAxes, parent_axes.figure.transFigure] and bbox_to_anchor is None:
        _api.warn_external('Using the axes or figure transform requires a bounding box in the respective coordinates. Using bbox_to_anchor=(0, 0, 1, 1) now.')
        bbox_to_anchor = (0, 0, 1, 1)
    if bbox_to_anchor is None:
        bbox_to_anchor = parent_axes.bbox
    if isinstance(bbox_to_anchor, tuple) and (isinstance(width, str) or isinstance(height, str)):
        if len(bbox_to_anchor) != 4:
            raise ValueError('Using relative units for width or height requires to provide a 4-tuple or a `Bbox` instance to `bbox_to_anchor.')
    return _add_inset_axes(parent_axes, axes_class, axes_kwargs, AnchoredSizeLocator(bbox_to_anchor, width, height, loc=loc, bbox_transform=bbox_transform, borderpad=borderpad))

@_docstring.dedent_interpd
def zoomed_inset_axes(parent_axes, zoom, loc='upper right', bbox_to_anchor=None, bbox_transform=None, axes_class=None, axes_kwargs=None, borderpad=0.5):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create an anchored inset axes by scaling a parent axes. For usage, also see\n    :doc:`the examples </gallery/axes_grid1/inset_locator_demo2>`.\n\n    Parameters\n    ----------\n    parent_axes : `~matplotlib.axes.Axes`\n        Axes to place the inset axes.\n\n    zoom : float\n        Scaling factor of the data axes. *zoom* > 1 will enlarge the\n        coordinates (i.e., "zoomed in"), while *zoom* < 1 will shrink the\n        coordinates (i.e., "zoomed out").\n\n    loc : str, default: \'upper right\'\n        Location to place the inset axes.  Valid locations are\n        \'upper left\', \'upper center\', \'upper right\',\n        \'center left\', \'center\', \'center right\',\n        \'lower left\', \'lower center\', \'lower right\'.\n        For backward compatibility, numeric values are accepted as well.\n        See the parameter *loc* of `.Legend` for details.\n\n    bbox_to_anchor : tuple or `~matplotlib.transforms.BboxBase`, optional\n        Bbox that the inset axes will be anchored to. If None,\n        *parent_axes.bbox* is used. If a tuple, can be either\n        [left, bottom, width, height], or [left, bottom].\n        If the kwargs *width* and/or *height* are specified in relative units,\n        the 2-tuple [left, bottom] cannot be used. Note that\n        the units of the bounding box are determined through the transform\n        in use. When using *bbox_to_anchor* it almost always makes sense to\n        also specify a *bbox_transform*. This might often be the axes transform\n        *parent_axes.transAxes*.\n\n    bbox_transform : `~matplotlib.transforms.Transform`, optional\n        Transformation for the bbox that contains the inset axes.\n        If None, a `.transforms.IdentityTransform` is used (i.e. pixel\n        coordinates). This is useful when not providing any argument to\n        *bbox_to_anchor*. When using *bbox_to_anchor* it almost always makes\n        sense to also specify a *bbox_transform*. This might often be the\n        axes transform *parent_axes.transAxes*. Inversely, when specifying\n        the axes- or figure-transform here, be aware that not specifying\n        *bbox_to_anchor* will use *parent_axes.bbox*, the units of which are\n        in display (pixel) coordinates.\n\n    axes_class : `~matplotlib.axes.Axes` type, default: `.HostAxes`\n        The type of the newly created inset axes.\n\n    axes_kwargs : dict, optional\n        Keyword arguments to pass to the constructor of the inset axes.\n        Valid arguments include:\n\n        %(Axes:kwdoc)s\n\n    borderpad : float, default: 0.5\n        Padding between inset axes and the bbox_to_anchor.\n        The units are axes font size, i.e. for a default font size of 10 points\n        *borderpad = 0.5* is equivalent to a padding of 5 points.\n\n    Returns\n    -------\n    inset_axes : *axes_class*\n        Inset axes object created.\n    '
    return _add_inset_axes(parent_axes, axes_class, axes_kwargs, AnchoredZoomLocator(parent_axes, zoom=zoom, loc=loc, bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform, borderpad=borderpad))

class _TransformedBboxWithCallback(TransformedBbox):
    """
    Variant of `.TransformBbox` which calls *callback* before returning points.

    Used by `.mark_inset` to unstale the parent axes' viewlim as needed.
    """

    def __init__(self, *args, callback, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._callback = callback

    def get_points(self):
        if False:
            i = 10
            return i + 15
        self._callback()
        return super().get_points()

@_docstring.dedent_interpd
def mark_inset(parent_axes, inset_axes, loc1, loc2, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Draw a box to mark the location of an area represented by an inset axes.\n\n    This function draws a box in *parent_axes* at the bounding box of\n    *inset_axes*, and shows a connection with the inset axes by drawing lines\n    at the corners, giving a "zoomed in" effect.\n\n    Parameters\n    ----------\n    parent_axes : `~matplotlib.axes.Axes`\n        Axes which contains the area of the inset axes.\n\n    inset_axes : `~matplotlib.axes.Axes`\n        The inset axes.\n\n    loc1, loc2 : {1, 2, 3, 4}\n        Corners to use for connecting the inset axes and the area in the\n        parent axes.\n\n    **kwargs\n        Patch properties for the lines and box drawn:\n\n        %(Patch:kwdoc)s\n\n    Returns\n    -------\n    pp : `~matplotlib.patches.Patch`\n        The patch drawn to represent the area of the inset axes.\n\n    p1, p2 : `~matplotlib.patches.Patch`\n        The patches connecting two corners of the inset axes and its area.\n    '
    rect = _TransformedBboxWithCallback(inset_axes.viewLim, parent_axes.transData, callback=parent_axes._unstale_viewLim)
    kwargs.setdefault('fill', bool({'fc', 'facecolor', 'color'}.intersection(kwargs)))
    pp = BboxPatch(rect, **kwargs)
    parent_axes.add_patch(pp)
    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)
    return (pp, p1, p2)