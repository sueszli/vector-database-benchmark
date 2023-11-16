"""
Container classes for `.Artist`\\s.

`OffsetBox`
    The base of all container artists defined in this module.

`AnchoredOffsetbox`, `AnchoredText`
    Anchor and align an arbitrary `.Artist` or a text relative to the parent
    axes or a specific anchor point.

`DrawingArea`
    A container with fixed width and height. Children have a fixed position
    inside the container and may be clipped.

`HPacker`, `VPacker`
    Containers for layouting their children vertically or horizontally.

`PaddedBox`
    A container to add a padding around an `.Artist`.

`TextArea`
    Contains a single `.Text` instance.
"""
import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, bbox_artist as mbbox_artist
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
DEBUG = False

def _compat_get_offset(meth):
    if False:
        print('Hello World!')
    '\n    Decorator for the get_offset method of OffsetBox and subclasses, that\n    allows supporting both the new signature (self, bbox, renderer) and the old\n    signature (self, width, height, xdescent, ydescent, renderer).\n    '
    sigs = [lambda self, width, height, xdescent, ydescent, renderer: locals(), lambda self, bbox, renderer: locals()]

    @functools.wraps(meth)
    def get_offset(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        params = _api.select_matching_signature(sigs, self, *args, **kwargs)
        bbox = params['bbox'] if 'bbox' in params else Bbox.from_bounds(-params['xdescent'], -params['ydescent'], params['width'], params['height'])
        return meth(params['self'], bbox, params['renderer'])
    return get_offset

def _bbox_artist(*args, **kwargs):
    if False:
        print('Hello World!')
    if DEBUG:
        mbbox_artist(*args, **kwargs)

def _get_packed_offsets(widths, total, sep, mode='fixed'):
    if False:
        while True:
            i = 10
    "\n    Pack boxes specified by their *widths*.\n\n    For simplicity of the description, the terminology used here assumes a\n    horizontal layout, but the function works equally for a vertical layout.\n\n    There are three packing *mode*\\s:\n\n    - 'fixed': The elements are packed tight to the left with a spacing of\n      *sep* in between. If *total* is *None* the returned total will be the\n      right edge of the last box. A non-*None* total will be passed unchecked\n      to the output. In particular this means that right edge of the last\n      box may be further to the right than the returned total.\n\n    - 'expand': Distribute the boxes with equal spacing so that the left edge\n      of the first box is at 0, and the right edge of the last box is at\n      *total*. The parameter *sep* is ignored in this mode. A total of *None*\n      is accepted and considered equal to 1. The total is returned unchanged\n      (except for the conversion *None* to 1). If the total is smaller than\n      the sum of the widths, the laid out boxes will overlap.\n\n    - 'equal': If *total* is given, the total space is divided in N equal\n      ranges and each box is left-aligned within its subspace.\n      Otherwise (*total* is *None*), *sep* must be provided and each box is\n      left-aligned in its subspace of width ``(max(widths) + sep)``. The\n      total width is then calculated to be ``N * (max(widths) + sep)``.\n\n    Parameters\n    ----------\n    widths : list of float\n        Widths of boxes to be packed.\n    total : float or None\n        Intended total length. *None* if not used.\n    sep : float or None\n        Spacing between boxes.\n    mode : {'fixed', 'expand', 'equal'}\n        The packing mode.\n\n    Returns\n    -------\n    total : float\n        The total width needed to accommodate the laid out boxes.\n    offsets : array of float\n        The left offsets of the boxes.\n    "
    _api.check_in_list(['fixed', 'expand', 'equal'], mode=mode)
    if mode == 'fixed':
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        if total is None:
            total = offsets_[-1] - sep
        return (total, offsets)
    elif mode == 'expand':
        if total is None:
            total = 1
        if len(widths) > 1:
            sep = (total - sum(widths)) / (len(widths) - 1)
        else:
            sep = 0
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        return (total, offsets)
    elif mode == 'equal':
        maxh = max(widths)
        if total is None:
            if sep is None:
                raise ValueError("total and sep cannot both be None when using layout mode 'equal'")
            total = (maxh + sep) * len(widths)
        else:
            sep = total / len(widths) - maxh
        offsets = (maxh + sep) * np.arange(len(widths))
        return (total, offsets)

def _get_aligned_offsets(yspans, height, align='baseline'):
    if False:
        print('Hello World!')
    '\n    Align boxes each specified by their ``(y0, y1)`` spans.\n\n    For simplicity of the description, the terminology used here assumes a\n    horizontal layout (i.e., vertical alignment), but the function works\n    equally for a vertical layout.\n\n    Parameters\n    ----------\n    yspans\n        List of (y0, y1) spans of boxes to be aligned.\n    height : float or None\n        Intended total height. If None, the maximum of the heights\n        (``y1 - y0``) in *yspans* is used.\n    align : {\'baseline\', \'left\', \'top\', \'right\', \'bottom\', \'center\'}\n        The alignment anchor of the boxes.\n\n    Returns\n    -------\n    (y0, y1)\n        y range spanned by the packing.  If a *height* was originally passed\n        in, then for all alignments other than "baseline", a span of ``(0,\n        height)`` is used without checking that it is actually large enough).\n    descent\n        The descent of the packing.\n    offsets\n        The bottom offsets of the boxes.\n    '
    _api.check_in_list(['baseline', 'left', 'top', 'right', 'bottom', 'center'], align=align)
    if height is None:
        height = max((y1 - y0 for (y0, y1) in yspans))
    if align == 'baseline':
        yspan = (min((y0 for (y0, y1) in yspans)), max((y1 for (y0, y1) in yspans)))
        offsets = [0] * len(yspans)
    elif align in ['left', 'bottom']:
        yspan = (0, height)
        offsets = [-y0 for (y0, y1) in yspans]
    elif align in ['right', 'top']:
        yspan = (0, height)
        offsets = [height - y1 for (y0, y1) in yspans]
    elif align == 'center':
        yspan = (0, height)
        offsets = [(height - (y1 - y0)) * 0.5 - y0 for (y0, y1) in yspans]
    return (yspan, offsets)

class OffsetBox(martist.Artist):
    """
    The OffsetBox is a simple container artist.

    The child artists are meant to be drawn at a relative position to its
    parent.

    Being an artist itself, all parameters are passed on to `.Artist`.
    """

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args)
        self._internal_update(kwargs)
        self.set_clip_on(False)
        self._children = []
        self._offset = (0, 0)

    def set_figure(self, fig):
        if False:
            print('Hello World!')
        '\n        Set the `.Figure` for the `.OffsetBox` and all its children.\n\n        Parameters\n        ----------\n        fig : `~matplotlib.figure.Figure`\n        '
        super().set_figure(fig)
        for c in self.get_children():
            c.set_figure(fig)

    @martist.Artist.axes.setter
    def axes(self, ax):
        if False:
            for i in range(10):
                print('nop')
        martist.Artist.axes.fset(self, ax)
        for c in self.get_children():
            if c is not None:
                c.axes = ax

    def contains(self, mouseevent):
        if False:
            print('Hello World!')
        '\n        Delegate the mouse event contains-check to the children.\n\n        As a container, the `.OffsetBox` does not respond itself to\n        mouseevents.\n\n        Parameters\n        ----------\n        mouseevent : `~matplotlib.backend_bases.MouseEvent`\n\n        Returns\n        -------\n        contains : bool\n            Whether any values are within the radius.\n        details : dict\n            An artist-specific dictionary of details of the event context,\n            such as which points are contained in the pick radius. See the\n            individual Artist subclasses for details.\n\n        See Also\n        --------\n        .Artist.contains\n        '
        if self._different_canvas(mouseevent):
            return (False, {})
        for c in self.get_children():
            (a, b) = c.contains(mouseevent)
            if a:
                return (a, b)
        return (False, {})

    def set_offset(self, xy):
        if False:
            while True:
                i = 10
        '\n        Set the offset.\n\n        Parameters\n        ----------\n        xy : (float, float) or callable\n            The (x, y) coordinates of the offset in display units. These can\n            either be given explicitly as a tuple (x, y), or by providing a\n            function that converts the extent into the offset. This function\n            must have the signature::\n\n                def offset(width, height, xdescent, ydescent, renderer) -> (float, float)\n        '
        self._offset = xy
        self.stale = True

    @_compat_get_offset
    def get_offset(self, bbox, renderer):
        if False:
            while True:
                i = 10
        '\n        Return the offset as a tuple (x, y).\n\n        The extent parameters have to be provided to handle the case where the\n        offset is dynamically determined by a callable (see\n        `~.OffsetBox.set_offset`).\n\n        Parameters\n        ----------\n        bbox : `.Bbox`\n        renderer : `.RendererBase` subclass\n        '
        return self._offset(bbox.width, bbox.height, -bbox.x0, -bbox.y0, renderer) if callable(self._offset) else self._offset

    def set_width(self, width):
        if False:
            i = 10
            return i + 15
        '\n        Set the width of the box.\n\n        Parameters\n        ----------\n        width : float\n        '
        self.width = width
        self.stale = True

    def set_height(self, height):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the height of the box.\n\n        Parameters\n        ----------\n        height : float\n        '
        self.height = height
        self.stale = True

    def get_visible_children(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of the visible child `.Artist`\\s.'
        return [c for c in self._children if c.get_visible()]

    def get_children(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of the child `.Artist`\\s.'
        return self._children

    def _get_bbox_and_child_offsets(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the bbox of the offsetbox and the child offsets.\n\n        The bbox should satisfy ``x0 <= x1 and y0 <= y1``.\n\n        Parameters\n        ----------\n        renderer : `.RendererBase` subclass\n\n        Returns\n        -------\n        bbox\n        list of (xoffset, yoffset) pairs\n        '
        raise NotImplementedError('get_bbox_and_offsets must be overridden in derived classes')

    def get_bbox(self, renderer):
        if False:
            print('Hello World!')
        'Return the bbox of the offsetbox, ignoring parent offsets.'
        (bbox, offsets) = self._get_bbox_and_child_offsets(renderer)
        return bbox

    def get_window_extent(self, renderer=None):
        if False:
            return 10
        if renderer is None:
            renderer = self.figure._get_renderer()
        bbox = self.get_bbox(renderer)
        try:
            (px, py) = self.get_offset(bbox, renderer)
        except TypeError:
            (px, py) = self.get_offset()
        return bbox.translated(px, py)

    def draw(self, renderer):
        if False:
            print('Hello World!')
        '\n        Update the location of children if necessary and draw them\n        to the given *renderer*.\n        '
        (bbox, offsets) = self._get_bbox_and_child_offsets(renderer)
        (px, py) = self.get_offset(bbox, renderer)
        for (c, (ox, oy)) in zip(self.get_visible_children(), offsets):
            c.set_offset((px + ox, py + oy))
            c.draw(renderer)
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.0))
        self.stale = False

class PackerBase(OffsetBox):

    def __init__(self, pad=0.0, sep=0.0, width=None, height=None, align='baseline', mode='fixed', children=None):
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        pad : float, default: 0.0\n            The boundary padding in points.\n\n        sep : float, default: 0.0\n            The spacing between items in points.\n\n        width, height : float, optional\n            Width and height of the container box in pixels, calculated if\n            *None*.\n\n        align : {'top', 'bottom', 'left', 'right', 'center', 'baseline'}, default: 'baseline'\n            Alignment of boxes.\n\n        mode : {'fixed', 'expand', 'equal'}, default: 'fixed'\n            The packing mode.\n\n            - 'fixed' packs the given `.Artist`\\s tight with *sep* spacing.\n            - 'expand' uses the maximal available space to distribute the\n              artists with equal spacing in between.\n            - 'equal': Each artist an equal fraction of the available space\n              and is left-aligned (or top-aligned) therein.\n\n        children : list of `.Artist`\n            The artists to pack.\n\n        Notes\n        -----\n        *pad* and *sep* are in points and will be scaled with the renderer\n        dpi, while *width* and *height* are in pixels.\n        "
        super().__init__()
        self.height = height
        self.width = width
        self.sep = sep
        self.pad = pad
        self.mode = mode
        self.align = align
        self._children = children

class VPacker(PackerBase):
    """
    VPacker packs its children vertically, automatically adjusting their
    relative positions at draw time.
    """

    def _get_bbox_and_child_offsets(self, renderer):
        if False:
            print('Hello World!')
        dpicor = renderer.points_to_pixels(1.0)
        pad = self.pad * dpicor
        sep = self.sep * dpicor
        if self.width is not None:
            for c in self.get_visible_children():
                if isinstance(c, PackerBase) and c.mode == 'expand':
                    c.set_width(self.width)
        bboxes = [c.get_bbox(renderer) for c in self.get_visible_children()]
        ((x0, x1), xoffsets) = _get_aligned_offsets([bbox.intervalx for bbox in bboxes], self.width, self.align)
        (height, yoffsets) = _get_packed_offsets([bbox.height for bbox in bboxes], self.height, sep, self.mode)
        yoffsets = height - (yoffsets + [bbox.y1 for bbox in bboxes])
        ydescent = yoffsets[0]
        yoffsets = yoffsets - ydescent
        return (Bbox.from_bounds(x0, -ydescent, x1 - x0, height).padded(pad), [*zip(xoffsets, yoffsets)])

class HPacker(PackerBase):
    """
    HPacker packs its children horizontally, automatically adjusting their
    relative positions at draw time.
    """

    def _get_bbox_and_child_offsets(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        dpicor = renderer.points_to_pixels(1.0)
        pad = self.pad * dpicor
        sep = self.sep * dpicor
        bboxes = [c.get_bbox(renderer) for c in self.get_visible_children()]
        if not bboxes:
            return (Bbox.from_bounds(0, 0, 0, 0).padded(pad), [])
        ((y0, y1), yoffsets) = _get_aligned_offsets([bbox.intervaly for bbox in bboxes], self.height, self.align)
        (width, xoffsets) = _get_packed_offsets([bbox.width for bbox in bboxes], self.width, sep, self.mode)
        x0 = bboxes[0].x0
        xoffsets -= [bbox.x0 for bbox in bboxes] - x0
        return (Bbox.from_bounds(x0, y0, width, y1 - y0).padded(pad), [*zip(xoffsets, yoffsets)])

class PaddedBox(OffsetBox):
    """
    A container to add a padding around an `.Artist`.

    The `.PaddedBox` contains a `.FancyBboxPatch` that is used to visualize
    it when rendering.
    """

    def __init__(self, child, pad=0.0, *, draw_frame=False, patch_attrs=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        child : `~matplotlib.artist.Artist`\n            The contained `.Artist`.\n        pad : float, default: 0.0\n            The padding in points. This will be scaled with the renderer dpi.\n            In contrast, *width* and *height* are in *pixels* and thus not\n            scaled.\n        draw_frame : bool\n            Whether to draw the contained `.FancyBboxPatch`.\n        patch_attrs : dict or None\n            Additional parameters passed to the contained `.FancyBboxPatch`.\n        '
        super().__init__()
        self.pad = pad
        self._children = [child]
        self.patch = FancyBboxPatch(xy=(0.0, 0.0), width=1.0, height=1.0, facecolor='w', edgecolor='k', mutation_scale=1, snap=True, visible=draw_frame, boxstyle='square,pad=0')
        if patch_attrs is not None:
            self.patch.update(patch_attrs)

    def _get_bbox_and_child_offsets(self, renderer):
        if False:
            print('Hello World!')
        pad = self.pad * renderer.points_to_pixels(1.0)
        return (self._children[0].get_bbox(renderer).padded(pad), [(0, 0)])

    def draw(self, renderer):
        if False:
            print('Hello World!')
        (bbox, offsets) = self._get_bbox_and_child_offsets(renderer)
        (px, py) = self.get_offset(bbox, renderer)
        for (c, (ox, oy)) in zip(self.get_visible_children(), offsets):
            c.set_offset((px + ox, py + oy))
        self.draw_frame(renderer)
        for c in self.get_visible_children():
            c.draw(renderer)
        self.stale = False

    def update_frame(self, bbox, fontsize=None):
        if False:
            print('Hello World!')
        self.patch.set_bounds(bbox.bounds)
        if fontsize:
            self.patch.set_mutation_scale(fontsize)
        self.stale = True

    def draw_frame(self, renderer):
        if False:
            i = 10
            return i + 15
        self.update_frame(self.get_window_extent(renderer))
        self.patch.draw(renderer)

class DrawingArea(OffsetBox):
    """
    The DrawingArea can contain any Artist as a child. The DrawingArea
    has a fixed width and height. The position of children relative to
    the parent is fixed. The children can be clipped at the
    boundaries of the parent.
    """

    def __init__(self, width, height, xdescent=0.0, ydescent=0.0, clip=False):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        width, height : float\n            Width and height of the container box.\n        xdescent, ydescent : float\n            Descent of the box in x- and y-direction.\n        clip : bool\n            Whether to clip the children to the box.\n        '
        super().__init__()
        self.width = width
        self.height = height
        self.xdescent = xdescent
        self.ydescent = ydescent
        self._clip_children = clip
        self.offset_transform = mtransforms.Affine2D()
        self.dpi_transform = mtransforms.Affine2D()

    @property
    def clip_children(self):
        if False:
            print('Hello World!')
        '\n        If the children of this DrawingArea should be clipped\n        by DrawingArea bounding box.\n        '
        return self._clip_children

    @clip_children.setter
    def clip_children(self, val):
        if False:
            for i in range(10):
                print('nop')
        self._clip_children = bool(val)
        self.stale = True

    def get_transform(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the `~matplotlib.transforms.Transform` applied to the children.\n        '
        return self.dpi_transform + self.offset_transform

    def set_transform(self, t):
        if False:
            while True:
                i = 10
        '\n        set_transform is ignored.\n        '

    def set_offset(self, xy):
        if False:
            print('Hello World!')
        '\n        Set the offset of the container.\n\n        Parameters\n        ----------\n        xy : (float, float)\n            The (x, y) coordinates of the offset in display units.\n        '
        self._offset = xy
        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])
        self.stale = True

    def get_offset(self):
        if False:
            i = 10
            return i + 15
        'Return offset of the container.'
        return self._offset

    def get_bbox(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        dpi_cor = renderer.points_to_pixels(1.0)
        return Bbox.from_bounds(-self.xdescent * dpi_cor, -self.ydescent * dpi_cor, self.width * dpi_cor, self.height * dpi_cor)

    def add_artist(self, a):
        if False:
            return 10
        'Add an `.Artist` to the container box.'
        self._children.append(a)
        if not a.is_transform_set():
            a.set_transform(self.get_transform())
        if self.axes is not None:
            a.axes = self.axes
        fig = self.figure
        if fig is not None:
            a.set_figure(fig)

    def draw(self, renderer):
        if False:
            while True:
                i = 10
        dpi_cor = renderer.points_to_pixels(1.0)
        self.dpi_transform.clear()
        self.dpi_transform.scale(dpi_cor)
        tpath = mtransforms.TransformedPath(mpath.Path([[0, 0], [0, self.height], [self.width, self.height], [self.width, 0]]), self.get_transform())
        for c in self._children:
            if self._clip_children and (not (c.clipbox or c._clippath)):
                c.set_clip_path(tpath)
            c.draw(renderer)
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.0))
        self.stale = False

class TextArea(OffsetBox):
    """
    The TextArea is a container artist for a single Text instance.

    The text is placed at (0, 0) with baseline+left alignment, by default. The
    width and height of the TextArea instance is the width and height of its
    child text.
    """

    def __init__(self, s, *, textprops=None, multilinebaseline=False):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        s : str\n            The text to be displayed.\n        textprops : dict, default: {}\n            Dictionary of keyword parameters to be passed to the `.Text`\n            instance in the TextArea.\n        multilinebaseline : bool, default: False\n            Whether the baseline for multiline text is adjusted so that it\n            is (approximately) center-aligned with single-line text.\n        '
        if textprops is None:
            textprops = {}
        self._text = mtext.Text(0, 0, s, **textprops)
        super().__init__()
        self._children = [self._text]
        self.offset_transform = mtransforms.Affine2D()
        self._baseline_transform = mtransforms.Affine2D()
        self._text.set_transform(self.offset_transform + self._baseline_transform)
        self._multilinebaseline = multilinebaseline

    def set_text(self, s):
        if False:
            i = 10
            return i + 15
        'Set the text of this area as a string.'
        self._text.set_text(s)
        self.stale = True

    def get_text(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the string representation of this area's text."
        return self._text.get_text()

    def set_multilinebaseline(self, t):
        if False:
            return 10
        '\n        Set multilinebaseline.\n\n        If True, the baseline for multiline text is adjusted so that it is\n        (approximately) center-aligned with single-line text.  This is used\n        e.g. by the legend implementation so that single-line labels are\n        baseline-aligned, but multiline labels are "center"-aligned with them.\n        '
        self._multilinebaseline = t
        self.stale = True

    def get_multilinebaseline(self):
        if False:
            i = 10
            return i + 15
        '\n        Get multilinebaseline.\n        '
        return self._multilinebaseline

    def set_transform(self, t):
        if False:
            print('Hello World!')
        '\n        set_transform is ignored.\n        '

    def set_offset(self, xy):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the offset of the container.\n\n        Parameters\n        ----------\n        xy : (float, float)\n            The (x, y) coordinates of the offset in display units.\n        '
        self._offset = xy
        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])
        self.stale = True

    def get_offset(self):
        if False:
            for i in range(10):
                print('nop')
        'Return offset of the container.'
        return self._offset

    def get_bbox(self, renderer):
        if False:
            return 10
        (_, h_, d_) = renderer.get_text_width_height_descent('lp', self._text._fontproperties, ismath='TeX' if self._text.get_usetex() else False)
        (bbox, info, yd) = self._text._get_layout(renderer)
        (w, h) = bbox.size
        self._baseline_transform.clear()
        if len(info) > 1 and self._multilinebaseline:
            yd_new = 0.5 * h - 0.5 * (h_ - d_)
            self._baseline_transform.translate(0, yd - yd_new)
            yd = yd_new
        else:
            h_d = max(h_ - d_, h - yd)
            h = h_d + yd
        ha = self._text.get_horizontalalignment()
        x0 = {'left': 0, 'center': -w / 2, 'right': -w}[ha]
        return Bbox.from_bounds(x0, -yd, w, h)

    def draw(self, renderer):
        if False:
            return 10
        self._text.draw(renderer)
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.0))
        self.stale = False

class AuxTransformBox(OffsetBox):
    """
    Offset Box with the aux_transform. Its children will be
    transformed with the aux_transform first then will be
    offsetted. The absolute coordinate of the aux_transform is meaning
    as it will be automatically adjust so that the left-lower corner
    of the bounding box of children will be set to (0, 0) before the
    offset transform.

    It is similar to drawing area, except that the extent of the box
    is not predetermined but calculated from the window extent of its
    children. Furthermore, the extent of the children will be
    calculated in the transformed coordinate.
    """

    def __init__(self, aux_transform):
        if False:
            while True:
                i = 10
        self.aux_transform = aux_transform
        super().__init__()
        self.offset_transform = mtransforms.Affine2D()
        self.ref_offset_transform = mtransforms.Affine2D()

    def add_artist(self, a):
        if False:
            return 10
        'Add an `.Artist` to the container box.'
        self._children.append(a)
        a.set_transform(self.get_transform())
        self.stale = True

    def get_transform(self):
        if False:
            return 10
        '\n        Return the :class:`~matplotlib.transforms.Transform` applied\n        to the children\n        '
        return self.aux_transform + self.ref_offset_transform + self.offset_transform

    def set_transform(self, t):
        if False:
            while True:
                i = 10
        '\n        set_transform is ignored.\n        '

    def set_offset(self, xy):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the offset of the container.\n\n        Parameters\n        ----------\n        xy : (float, float)\n            The (x, y) coordinates of the offset in display units.\n        '
        self._offset = xy
        self.offset_transform.clear()
        self.offset_transform.translate(xy[0], xy[1])
        self.stale = True

    def get_offset(self):
        if False:
            for i in range(10):
                print('nop')
        'Return offset of the container.'
        return self._offset

    def get_bbox(self, renderer):
        if False:
            print('Hello World!')
        _off = self.offset_transform.get_matrix()
        self.ref_offset_transform.clear()
        self.offset_transform.clear()
        bboxes = [c.get_window_extent(renderer) for c in self._children]
        ub = Bbox.union(bboxes)
        self.ref_offset_transform.translate(-ub.x0, -ub.y0)
        self.offset_transform.set_matrix(_off)
        return Bbox.from_bounds(0, 0, ub.width, ub.height)

    def draw(self, renderer):
        if False:
            i = 10
            return i + 15
        for c in self._children:
            c.draw(renderer)
        _bbox_artist(self, renderer, fill=False, props=dict(pad=0.0))
        self.stale = False

class AnchoredOffsetbox(OffsetBox):
    """
    An offset box placed according to location *loc*.

    AnchoredOffsetbox has a single child.  When multiple children are needed,
    use an extra OffsetBox to enclose them.  By default, the offset box is
    anchored against its parent axes. You may explicitly specify the
    *bbox_to_anchor*.
    """
    zorder = 5
    codes = {'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4, 'right': 5, 'center left': 6, 'center right': 7, 'lower center': 8, 'upper center': 9, 'center': 10}

    def __init__(self, loc, *, pad=0.4, borderpad=0.5, child=None, prop=None, frameon=True, bbox_to_anchor=None, bbox_transform=None, **kwargs):
        if False:
            return 10
        "\n        Parameters\n        ----------\n        loc : str\n            The box location.  Valid locations are\n            'upper left', 'upper center', 'upper right',\n            'center left', 'center', 'center right',\n            'lower left', 'lower center', 'lower right'.\n            For backward compatibility, numeric values are accepted as well.\n            See the parameter *loc* of `.Legend` for details.\n        pad : float, default: 0.4\n            Padding around the child as fraction of the fontsize.\n        borderpad : float, default: 0.5\n            Padding between the offsetbox frame and the *bbox_to_anchor*.\n        child : `.OffsetBox`\n            The box that will be anchored.\n        prop : `.FontProperties`\n            This is only used as a reference for paddings. If not given,\n            :rc:`legend.fontsize` is used.\n        frameon : bool\n            Whether to draw a frame around the box.\n        bbox_to_anchor : `.BboxBase`, 2-tuple, or 4-tuple of floats\n            Box that is used to position the legend in conjunction with *loc*.\n        bbox_transform : None or :class:`matplotlib.transforms.Transform`\n            The transform for the bounding box (*bbox_to_anchor*).\n        **kwargs\n            All other parameters are passed on to `.OffsetBox`.\n\n        Notes\n        -----\n        See `.Legend` for a detailed description of the anchoring mechanism.\n        "
        super().__init__(**kwargs)
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)
        self.set_child(child)
        if isinstance(loc, str):
            loc = _api.check_getitem(self.codes, loc=loc)
        self.loc = loc
        self.borderpad = borderpad
        self.pad = pad
        if prop is None:
            self.prop = FontProperties(size=mpl.rcParams['legend.fontsize'])
        else:
            self.prop = FontProperties._from_any(prop)
            if isinstance(prop, dict) and 'size' not in prop:
                self.prop.set_size(mpl.rcParams['legend.fontsize'])
        self.patch = FancyBboxPatch(xy=(0.0, 0.0), width=1.0, height=1.0, facecolor='w', edgecolor='k', mutation_scale=self.prop.get_size_in_points(), snap=True, visible=frameon, boxstyle='square,pad=0')

    def set_child(self, child):
        if False:
            i = 10
            return i + 15
        'Set the child to be anchored.'
        self._child = child
        if child is not None:
            child.axes = self.axes
        self.stale = True

    def get_child(self):
        if False:
            while True:
                i = 10
        'Return the child.'
        return self._child

    def get_children(self):
        if False:
            return 10
        'Return the list of children.'
        return [self._child]

    def get_bbox(self, renderer):
        if False:
            i = 10
            return i + 15
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        pad = self.pad * fontsize
        return self.get_child().get_bbox(renderer).padded(pad)

    def get_bbox_to_anchor(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the bbox that the box is anchored to.'
        if self._bbox_to_anchor is None:
            return self.axes.bbox
        else:
            transform = self._bbox_to_anchor_transform
            if transform is None:
                return self._bbox_to_anchor
            else:
                return TransformedBbox(self._bbox_to_anchor, transform)

    def set_bbox_to_anchor(self, bbox, transform=None):
        if False:
            while True:
                i = 10
        '\n        Set the bbox that the box is anchored to.\n\n        *bbox* can be a Bbox instance, a list of [left, bottom, width,\n        height], or a list of [left, bottom] where the width and\n        height will be assumed to be zero. The bbox will be\n        transformed to display coordinate by the given transform.\n        '
        if bbox is None or isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            try:
                l = len(bbox)
            except TypeError as err:
                raise ValueError(f'Invalid bbox: {bbox}') from err
            if l == 2:
                bbox = [bbox[0], bbox[1], 0, 0]
            self._bbox_to_anchor = Bbox.from_bounds(*bbox)
        self._bbox_to_anchor_transform = transform
        self.stale = True

    @_compat_get_offset
    def get_offset(self, bbox, renderer):
        if False:
            print('Hello World!')
        pad = self.borderpad * renderer.points_to_pixels(self.prop.get_size_in_points())
        bbox_to_anchor = self.get_bbox_to_anchor()
        (x0, y0) = _get_anchored_bbox(self.loc, Bbox.from_bounds(0, 0, bbox.width, bbox.height), bbox_to_anchor, pad)
        return (x0 - bbox.x0, y0 - bbox.y0)

    def update_frame(self, bbox, fontsize=None):
        if False:
            i = 10
            return i + 15
        self.patch.set_bounds(bbox.bounds)
        if fontsize:
            self.patch.set_mutation_scale(fontsize)

    def draw(self, renderer):
        if False:
            i = 10
            return i + 15
        if not self.get_visible():
            return
        bbox = self.get_window_extent(renderer)
        fontsize = renderer.points_to_pixels(self.prop.get_size_in_points())
        self.update_frame(bbox, fontsize)
        self.patch.draw(renderer)
        (px, py) = self.get_offset(self.get_bbox(renderer), renderer)
        self.get_child().set_offset((px, py))
        self.get_child().draw(renderer)
        self.stale = False

def _get_anchored_bbox(loc, bbox, parentbbox, borderpad):
    if False:
        print('Hello World!')
    '\n    Return the (x, y) position of the *bbox* anchored at the *parentbbox* with\n    the *loc* code with the *borderpad*.\n    '
    c = [None, 'NE', 'NW', 'SW', 'SE', 'E', 'W', 'E', 'S', 'N', 'C'][loc]
    container = parentbbox.padded(-borderpad)
    return bbox.anchored(c, container=container).p0

class AnchoredText(AnchoredOffsetbox):
    """
    AnchoredOffsetbox with Text.
    """

    def __init__(self, s, loc, *, pad=0.4, borderpad=0.5, prop=None, **kwargs):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        s : str\n            Text.\n\n        loc : str\n            Location code. See `AnchoredOffsetbox`.\n\n        pad : float, default: 0.4\n            Padding around the text as fraction of the fontsize.\n\n        borderpad : float, default: 0.5\n            Spacing between the offsetbox frame and the *bbox_to_anchor*.\n\n        prop : dict, optional\n            Dictionary of keyword parameters to be passed to the\n            `~matplotlib.text.Text` instance contained inside AnchoredText.\n\n        **kwargs\n            All other parameters are passed to `AnchoredOffsetbox`.\n        '
        if prop is None:
            prop = {}
        badkwargs = {'va', 'verticalalignment'}
        if badkwargs & set(prop):
            raise ValueError('Mixing verticalalignment with AnchoredText is not supported.')
        self.txt = TextArea(s, textprops=prop)
        fp = self.txt._text.get_fontproperties()
        super().__init__(loc, pad=pad, borderpad=borderpad, child=self.txt, prop=fp, **kwargs)

class OffsetImage(OffsetBox):

    def __init__(self, arr, *, zoom=1, cmap=None, norm=None, interpolation=None, origin=None, filternorm=True, filterrad=4.0, resample=False, dpi_cor=True, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._dpi_cor = dpi_cor
        self.image = BboxImage(bbox=self.get_window_extent, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin, filternorm=filternorm, filterrad=filterrad, resample=resample, **kwargs)
        self._children = [self.image]
        self.set_zoom(zoom)
        self.set_data(arr)

    def set_data(self, arr):
        if False:
            i = 10
            return i + 15
        self._data = np.asarray(arr)
        self.image.set_data(self._data)
        self.stale = True

    def get_data(self):
        if False:
            i = 10
            return i + 15
        return self._data

    def set_zoom(self, zoom):
        if False:
            for i in range(10):
                print('nop')
        self._zoom = zoom
        self.stale = True

    def get_zoom(self):
        if False:
            return 10
        return self._zoom

    def get_offset(self):
        if False:
            for i in range(10):
                print('nop')
        'Return offset of the container.'
        return self._offset

    def get_children(self):
        if False:
            i = 10
            return i + 15
        return [self.image]

    def get_bbox(self, renderer):
        if False:
            i = 10
            return i + 15
        dpi_cor = renderer.points_to_pixels(1.0) if self._dpi_cor else 1.0
        zoom = self.get_zoom()
        data = self.get_data()
        (ny, nx) = data.shape[:2]
        (w, h) = (dpi_cor * nx * zoom, dpi_cor * ny * zoom)
        return Bbox.from_bounds(0, 0, w, h)

    def draw(self, renderer):
        if False:
            while True:
                i = 10
        self.image.draw(renderer)
        self.stale = False

class AnnotationBbox(martist.Artist, mtext._AnnotationBase):
    """
    Container for an `OffsetBox` referring to a specific position *xy*.

    Optionally an arrow pointing from the offsetbox to *xy* can be drawn.

    This is like `.Annotation`, but with `OffsetBox` instead of `.Text`.
    """
    zorder = 3

    def __str__(self):
        if False:
            return 10
        return f'AnnotationBbox({self.xy[0]:g},{self.xy[1]:g})'

    @_docstring.dedent_interpd
    def __init__(self, offsetbox, xy, xybox=None, xycoords='data', boxcoords=None, *, frameon=True, pad=0.4, annotation_clip=None, box_alignment=(0.5, 0.5), bboxprops=None, arrowprops=None, fontsize=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Parameters\n        ----------\n        offsetbox : `OffsetBox`\n\n        xy : (float, float)\n            The point *(x, y)* to annotate. The coordinate system is determined\n            by *xycoords*.\n\n        xybox : (float, float), default: *xy*\n            The position *(x, y)* to place the text at. The coordinate system\n            is determined by *boxcoords*.\n\n        xycoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: 'data'\n            The coordinate system that *xy* is given in. See the parameter\n            *xycoords* in `.Annotation` for a detailed description.\n\n        boxcoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: value of *xycoords*\n            The coordinate system that *xybox* is given in. See the parameter\n            *textcoords* in `.Annotation` for a detailed description.\n\n        frameon : bool, default: True\n            By default, the text is surrounded by a white `.FancyBboxPatch`\n            (accessible as the ``patch`` attribute of the `.AnnotationBbox`).\n            If *frameon* is set to False, this patch is made invisible.\n\n        annotation_clip: bool or None, default: None\n            Whether to clip (i.e. not draw) the annotation when the annotation\n            point *xy* is outside the axes area.\n\n            - If *True*, the annotation will be clipped when *xy* is outside\n              the axes.\n            - If *False*, the annotation will always be drawn.\n            - If *None*, the annotation will be clipped when *xy* is outside\n              the axes and *xycoords* is 'data'.\n\n        pad : float, default: 0.4\n            Padding around the offsetbox.\n\n        box_alignment : (float, float)\n            A tuple of two floats for a vertical and horizontal alignment of\n            the offset box w.r.t. the *boxcoords*.\n            The lower-left corner is (0, 0) and upper-right corner is (1, 1).\n\n        bboxprops : dict, optional\n            A dictionary of properties to set for the annotation bounding box,\n            for example *boxstyle* and *alpha*.  See `.FancyBboxPatch` for\n            details.\n\n        arrowprops: dict, optional\n            Arrow properties, see `.Annotation` for description.\n\n        fontsize: float or str, optional\n            Translated to points and passed as *mutation_scale* into\n            `.FancyBboxPatch` to scale attributes of the box style (e.g. pad\n            or rounding_size).  The name is chosen in analogy to `.Text` where\n            *fontsize* defines the mutation scale as well.  If not given,\n            :rc:`legend.fontsize` is used.  See `.Text.set_fontsize` for valid\n            values.\n\n        **kwargs\n            Other `AnnotationBbox` properties.  See `.AnnotationBbox.set` for\n            a list.\n        "
        martist.Artist.__init__(self)
        mtext._AnnotationBase.__init__(self, xy, xycoords=xycoords, annotation_clip=annotation_clip)
        self.offsetbox = offsetbox
        self.arrowprops = arrowprops.copy() if arrowprops is not None else None
        self.set_fontsize(fontsize)
        self.xybox = xybox if xybox is not None else xy
        self.boxcoords = boxcoords if boxcoords is not None else xycoords
        self._box_alignment = box_alignment
        if arrowprops is not None:
            self._arrow_relpos = self.arrowprops.pop('relpos', (0.5, 0.5))
            self.arrow_patch = FancyArrowPatch((0, 0), (1, 1), **self.arrowprops)
        else:
            self._arrow_relpos = None
            self.arrow_patch = None
        self.patch = FancyBboxPatch(xy=(0.0, 0.0), width=1.0, height=1.0, facecolor='w', edgecolor='k', mutation_scale=self.prop.get_size_in_points(), snap=True, visible=frameon)
        self.patch.set_boxstyle('square', pad=pad)
        if bboxprops:
            self.patch.set(**bboxprops)
        self._internal_update(kwargs)

    @property
    def xyann(self):
        if False:
            while True:
                i = 10
        return self.xybox

    @xyann.setter
    def xyann(self, xyann):
        if False:
            i = 10
            return i + 15
        self.xybox = xyann
        self.stale = True

    @property
    def anncoords(self):
        if False:
            i = 10
            return i + 15
        return self.boxcoords

    @anncoords.setter
    def anncoords(self, coords):
        if False:
            print('Hello World!')
        self.boxcoords = coords
        self.stale = True

    def contains(self, mouseevent):
        if False:
            for i in range(10):
                print('nop')
        if self._different_canvas(mouseevent):
            return (False, {})
        if not self._check_xy(None):
            return (False, {})
        return self.offsetbox.contains(mouseevent)

    def get_children(self):
        if False:
            for i in range(10):
                print('nop')
        children = [self.offsetbox, self.patch]
        if self.arrow_patch:
            children.append(self.arrow_patch)
        return children

    def set_figure(self, fig):
        if False:
            print('Hello World!')
        if self.arrow_patch is not None:
            self.arrow_patch.set_figure(fig)
        self.offsetbox.set_figure(fig)
        martist.Artist.set_figure(self, fig)

    def set_fontsize(self, s=None):
        if False:
            return 10
        '\n        Set the fontsize in points.\n\n        If *s* is not given, reset to :rc:`legend.fontsize`.\n        '
        if s is None:
            s = mpl.rcParams['legend.fontsize']
        self.prop = FontProperties(size=s)
        self.stale = True

    def get_fontsize(self):
        if False:
            i = 10
            return i + 15
        'Return the fontsize in points.'
        return self.prop.get_size_in_points()

    def get_window_extent(self, renderer=None):
        if False:
            print('Hello World!')
        if renderer is None:
            renderer = self.figure._get_renderer()
        self.update_positions(renderer)
        return Bbox.union([child.get_window_extent(renderer) for child in self.get_children()])

    def get_tightbbox(self, renderer=None):
        if False:
            i = 10
            return i + 15
        if renderer is None:
            renderer = self.figure._get_renderer()
        self.update_positions(renderer)
        return Bbox.union([child.get_tightbbox(renderer) for child in self.get_children()])

    def update_positions(self, renderer):
        if False:
            return 10
        'Update pixel positions for the annotated point, the text, and the arrow.'
        (ox0, oy0) = self._get_xy(renderer, self.xybox, self.boxcoords)
        bbox = self.offsetbox.get_bbox(renderer)
        (fw, fh) = self._box_alignment
        self.offsetbox.set_offset((ox0 - fw * bbox.width - bbox.x0, oy0 - fh * bbox.height - bbox.y0))
        bbox = self.offsetbox.get_window_extent(renderer)
        self.patch.set_bounds(bbox.bounds)
        mutation_scale = renderer.points_to_pixels(self.get_fontsize())
        self.patch.set_mutation_scale(mutation_scale)
        if self.arrowprops:
            arrow_begin = bbox.p0 + bbox.size * self._arrow_relpos
            arrow_end = self._get_position_xy(renderer)
            self.arrow_patch.set_positions(arrow_begin, arrow_end)
            if 'mutation_scale' in self.arrowprops:
                mutation_scale = renderer.points_to_pixels(self.arrowprops['mutation_scale'])
            self.arrow_patch.set_mutation_scale(mutation_scale)
            patchA = self.arrowprops.get('patchA', self.patch)
            self.arrow_patch.set_patchA(patchA)

    def draw(self, renderer):
        if False:
            i = 10
            return i + 15
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible() or not self._check_xy(renderer):
            return
        renderer.open_group(self.__class__.__name__, gid=self.get_gid())
        self.update_positions(renderer)
        if self.arrow_patch is not None:
            if self.arrow_patch.figure is None and self.figure is not None:
                self.arrow_patch.figure = self.figure
            self.arrow_patch.draw(renderer)
        self.patch.draw(renderer)
        self.offsetbox.draw(renderer)
        renderer.close_group(self.__class__.__name__)
        self.stale = False

class DraggableBase:
    """
    Helper base class for a draggable artist (legend, offsetbox).

    Derived classes must override the following methods::

        def save_offset(self):
            '''
            Called when the object is picked for dragging; should save the
            reference position of the artist.
            '''

        def update_offset(self, dx, dy):
            '''
            Called during the dragging; (*dx*, *dy*) is the pixel offset from
            the point where the mouse drag started.
            '''

    Optionally, you may override the following method::

        def finalize_offset(self):
            '''Called when the mouse is released.'''

    In the current implementation of `.DraggableLegend` and
    `DraggableAnnotation`, `update_offset` places the artists in display
    coordinates, and `finalize_offset` recalculates their position in axes
    coordinate and set a relevant attribute.
    """

    def __init__(self, ref_artist, use_blit=False):
        if False:
            print('Hello World!')
        self.ref_artist = ref_artist
        if not ref_artist.pickable():
            ref_artist.set_picker(True)
        self.got_artist = False
        self._use_blit = use_blit and self.canvas.supports_blit
        callbacks = ref_artist.figure._canvas_callbacks
        self._disconnectors = [functools.partial(callbacks.disconnect, callbacks._connect_picklable(name, func)) for (name, func) in [('pick_event', self.on_pick), ('button_release_event', self.on_release), ('motion_notify_event', self.on_motion)]]
    canvas = property(lambda self: self.ref_artist.figure.canvas)
    cids = property(lambda self: [disconnect.args[0] for disconnect in self._disconnectors[:2]])

    def on_motion(self, evt):
        if False:
            print('Hello World!')
        if self._check_still_parented() and self.got_artist:
            dx = evt.x - self.mouse_x
            dy = evt.y - self.mouse_y
            self.update_offset(dx, dy)
            if self._use_blit:
                self.canvas.restore_region(self.background)
                self.ref_artist.draw(self.ref_artist.figure._get_renderer())
                self.canvas.blit()
            else:
                self.canvas.draw()

    def on_pick(self, evt):
        if False:
            i = 10
            return i + 15
        if self._check_still_parented() and evt.artist == self.ref_artist:
            self.mouse_x = evt.mouseevent.x
            self.mouse_y = evt.mouseevent.y
            self.got_artist = True
            if self._use_blit:
                self.ref_artist.set_animated(True)
                self.canvas.draw()
                self.background = self.canvas.copy_from_bbox(self.ref_artist.figure.bbox)
                self.ref_artist.draw(self.ref_artist.figure._get_renderer())
                self.canvas.blit()
            self.save_offset()

    def on_release(self, event):
        if False:
            i = 10
            return i + 15
        if self._check_still_parented() and self.got_artist:
            self.finalize_offset()
            self.got_artist = False
            if self._use_blit:
                self.ref_artist.set_animated(False)

    def _check_still_parented(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ref_artist.figure is None:
            self.disconnect()
            return False
        else:
            return True

    def disconnect(self):
        if False:
            for i in range(10):
                print('nop')
        'Disconnect the callbacks.'
        for disconnector in self._disconnectors:
            disconnector()

    def save_offset(self):
        if False:
            while True:
                i = 10
        pass

    def update_offset(self, dx, dy):
        if False:
            for i in range(10):
                print('nop')
        pass

    def finalize_offset(self):
        if False:
            return 10
        pass

class DraggableOffsetBox(DraggableBase):

    def __init__(self, ref_artist, offsetbox, use_blit=False):
        if False:
            i = 10
            return i + 15
        super().__init__(ref_artist, use_blit=use_blit)
        self.offsetbox = offsetbox

    def save_offset(self):
        if False:
            return 10
        offsetbox = self.offsetbox
        renderer = offsetbox.figure._get_renderer()
        offset = offsetbox.get_offset(offsetbox.get_bbox(renderer), renderer)
        (self.offsetbox_x, self.offsetbox_y) = offset
        self.offsetbox.set_offset(offset)

    def update_offset(self, dx, dy):
        if False:
            i = 10
            return i + 15
        loc_in_canvas = (self.offsetbox_x + dx, self.offsetbox_y + dy)
        self.offsetbox.set_offset(loc_in_canvas)

    def get_loc_in_canvas(self):
        if False:
            while True:
                i = 10
        offsetbox = self.offsetbox
        renderer = offsetbox.figure._get_renderer()
        bbox = offsetbox.get_bbox(renderer)
        (ox, oy) = offsetbox._offset
        loc_in_canvas = (ox + bbox.x0, oy + bbox.y0)
        return loc_in_canvas

class DraggableAnnotation(DraggableBase):

    def __init__(self, annotation, use_blit=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(annotation, use_blit=use_blit)
        self.annotation = annotation

    def save_offset(self):
        if False:
            return 10
        ann = self.annotation
        (self.ox, self.oy) = ann.get_transform().transform(ann.xyann)

    def update_offset(self, dx, dy):
        if False:
            i = 10
            return i + 15
        ann = self.annotation
        ann.xyann = ann.get_transform().inverted().transform((self.ox + dx, self.oy + dy))