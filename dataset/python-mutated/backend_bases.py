"""
Abstract base classes define the primitives that renderers and
graphics contexts must implement to serve as a Matplotlib backend.

`RendererBase`
    An abstract base class to handle drawing/rendering operations.

`FigureCanvasBase`
    The abstraction layer that separates the `.Figure` from the backend
    specific details like a user interface drawing area.

`GraphicsContextBase`
    An abstract base class that provides color, line styles, etc.

`Event`
    The base class for all of the Matplotlib event handling.  Derived classes
    such as `KeyEvent` and `MouseEvent` store the meta data like keys and
    buttons pressed, x and y locations in pixel and `~.axes.Axes` coordinates.

`ShowBase`
    The base class for the ``Show`` class of each interactive backend; the
    'show' callable is then set to ``Show.__call__``.

`ToolContainerBase`
    The base class for the Toolbar class of each interactive backend.
"""
from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import signal
import socket
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import _api, backend_tools as tools, cbook, colors, _docstring, text, _tight_bbox, transforms, widgets, is_interactive, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
_log = logging.getLogger(__name__)
_default_filetypes = {'eps': 'Encapsulated Postscript', 'jpg': 'Joint Photographic Experts Group', 'jpeg': 'Joint Photographic Experts Group', 'pdf': 'Portable Document Format', 'pgf': 'PGF code for LaTeX', 'png': 'Portable Network Graphics', 'ps': 'Postscript', 'raw': 'Raw RGBA bitmap', 'rgba': 'Raw RGBA bitmap', 'svg': 'Scalable Vector Graphics', 'svgz': 'Scalable Vector Graphics', 'tif': 'Tagged Image File Format', 'tiff': 'Tagged Image File Format', 'webp': 'WebP Image Format'}
_default_backends = {'eps': 'matplotlib.backends.backend_ps', 'jpg': 'matplotlib.backends.backend_agg', 'jpeg': 'matplotlib.backends.backend_agg', 'pdf': 'matplotlib.backends.backend_pdf', 'pgf': 'matplotlib.backends.backend_pgf', 'png': 'matplotlib.backends.backend_agg', 'ps': 'matplotlib.backends.backend_ps', 'raw': 'matplotlib.backends.backend_agg', 'rgba': 'matplotlib.backends.backend_agg', 'svg': 'matplotlib.backends.backend_svg', 'svgz': 'matplotlib.backends.backend_svg', 'tif': 'matplotlib.backends.backend_agg', 'tiff': 'matplotlib.backends.backend_agg', 'webp': 'matplotlib.backends.backend_agg'}

def _safe_pyplot_import():
    if False:
        print('Hello World!')
    '\n    Import and return ``pyplot``, correctly setting the backend if one is\n    already forced.\n    '
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        current_framework = cbook._get_running_interactive_framework()
        if current_framework is None:
            raise
        backend_mapping = {'qt': 'qtagg', 'gtk3': 'gtk3agg', 'gtk4': 'gtk4agg', 'wx': 'wxagg', 'tk': 'tkagg', 'macosx': 'macosx', 'headless': 'agg'}
        backend = backend_mapping[current_framework]
        rcParams['backend'] = mpl.rcParamsOrig['backend'] = backend
        import matplotlib.pyplot as plt
    return plt

def register_backend(format, backend, description=None):
    if False:
        while True:
            i = 10
    '\n    Register a backend for saving to a given file format.\n\n    Parameters\n    ----------\n    format : str\n        File extension\n    backend : module string or canvas class\n        Backend for handling file output\n    description : str, default: ""\n        Description of the file type.\n    '
    if description is None:
        description = ''
    _default_backends[format] = backend
    _default_filetypes[format] = description

def get_registered_canvas_class(format):
    if False:
        return 10
    '\n    Return the registered default canvas for given file format.\n    Handles deferred import of required backend.\n    '
    if format not in _default_backends:
        return None
    backend_class = _default_backends[format]
    if isinstance(backend_class, str):
        backend_class = importlib.import_module(backend_class).FigureCanvas
        _default_backends[format] = backend_class
    return backend_class

class RendererBase:
    """
    An abstract base class to handle drawing/rendering operations.

    The following methods must be implemented in the backend for full
    functionality (though just implementing `draw_path` alone would give a
    highly capable backend):

    * `draw_path`
    * `draw_image`
    * `draw_gouraud_triangles`

    The following methods *should* be implemented in the backend for
    optimization reasons:

    * `draw_text`
    * `draw_markers`
    * `draw_path_collection`
    * `draw_quad_mesh`
    """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self._texmanager = None
        self._text2path = text.TextToPath()
        self._raster_depth = 0
        self._rasterizing = False

    def open_group(self, s, gid=None):
        if False:
            print('Hello World!')
        '\n        Open a grouping element with label *s* and *gid* (if set) as id.\n\n        Only used by the SVG renderer.\n        '

    def close_group(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close a grouping element with label *s*.\n\n        Only used by the SVG renderer.\n        '

    def draw_path(self, gc, path, transform, rgbFace=None):
        if False:
            for i in range(10):
                print('nop')
        'Draw a `~.path.Path` instance using the given affine transform.'
        raise NotImplementedError

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        if False:
            while True:
                i = 10
        "\n        Draw a marker at each of *path*'s vertices (excluding control points).\n\n        The base (fallback) implementation makes multiple calls to `draw_path`.\n        Backends may want to override this method in order to draw the marker\n        only once and reuse it multiple times.\n\n        Parameters\n        ----------\n        gc : `.GraphicsContextBase`\n            The graphics context.\n        marker_path : `~matplotlib.path.Path`\n            The path for the marker.\n        marker_trans : `~matplotlib.transforms.Transform`\n            An affine transform applied to the marker.\n        path : `~matplotlib.path.Path`\n            The locations to draw the markers.\n        trans : `~matplotlib.transforms.Transform`\n            An affine transform applied to the path.\n        rgbFace : color, optional\n        "
        for (vertices, codes) in path.iter_segments(trans, simplify=False):
            if len(vertices):
                (x, y) = vertices[-2:]
                self.draw_path(gc, marker_path, marker_trans + transforms.Affine2D().translate(x, y), rgbFace)

    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
        if False:
            print('Hello World!')
        '\n        Draw a collection of *paths*.\n\n        Each path is first transformed by the corresponding entry\n        in *all_transforms* (a list of (3, 3) matrices) and then by\n        *master_transform*.  They are then translated by the corresponding\n        entry in *offsets*, which has been first transformed by *offset_trans*.\n\n        *facecolors*, *edgecolors*, *linewidths*, *linestyles*, and\n        *antialiased* are lists that set the corresponding properties.\n\n        *offset_position* is unused now, but the argument is kept for\n        backwards compatibility.\n\n        The base (fallback) implementation makes multiple calls to `draw_path`.\n        Backends may want to override this in order to render each set of\n        path data only once, and then reference that path multiple times with\n        the different offsets, colors, styles etc.  The generator methods\n        `_iter_collection_raw_paths` and `_iter_collection` are provided to\n        help with (and standardize) the implementation across backends.  It\n        is highly recommended to use those generators, so that changes to the\n        behavior of `draw_path_collection` can be made globally.\n        '
        path_ids = self._iter_collection_raw_paths(master_transform, paths, all_transforms)
        for (xo, yo, path_id, gc0, rgbFace) in self._iter_collection(gc, list(path_ids), offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
            (path, transform) = path_id
            if xo != 0 or yo != 0:
                transform = transform.frozen()
                transform.translate(xo, yo)
            self.draw_path(gc0, path, transform, rgbFace)

    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight, coordinates, offsets, offsetTrans, facecolors, antialiased, edgecolors):
        if False:
            i = 10
            return i + 15
        '\n        Draw a quadmesh.\n\n        The base (fallback) implementation converts the quadmesh to paths and\n        then calls `draw_path_collection`.\n        '
        from matplotlib.collections import QuadMesh
        paths = QuadMesh._convert_mesh_to_paths(coordinates)
        if edgecolors is None:
            edgecolors = facecolors
        linewidths = np.array([gc.get_linewidth()], float)
        return self.draw_path_collection(gc, master_transform, paths, [], offsets, offsetTrans, facecolors, edgecolors, linewidths, [], [antialiased], [None], 'screen')

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform):
        if False:
            print('Hello World!')
        '\n        Draw a series of Gouraud triangles.\n\n        Parameters\n        ----------\n        gc : `.GraphicsContextBase`\n            The graphics context.\n        triangles_array : (N, 3, 2) array-like\n            Array of *N* (x, y) points for the triangles.\n        colors_array : (N, 3, 4) array-like\n            Array of *N* RGBA colors for each point of the triangles.\n        transform : `~matplotlib.transforms.Transform`\n            An affine transform to apply to the points.\n        '
        raise NotImplementedError

    def _iter_collection_raw_paths(self, master_transform, paths, all_transforms):
        if False:
            print('Hello World!')
        '\n        Helper method (along with `_iter_collection`) to implement\n        `draw_path_collection` in a memory-efficient manner.\n\n        This method yields all of the base path/transform combinations, given a\n        master transform, a list of paths and list of transforms.\n\n        The arguments should be exactly what is passed in to\n        `draw_path_collection`.\n\n        The backend should take each yielded path and transform and create an\n        object that can be referenced (reused) later.\n        '
        Npaths = len(paths)
        Ntransforms = len(all_transforms)
        N = max(Npaths, Ntransforms)
        if Npaths == 0:
            return
        transform = transforms.IdentityTransform()
        for i in range(N):
            path = paths[i % Npaths]
            if Ntransforms:
                transform = Affine2D(all_transforms[i % Ntransforms])
            yield (path, transform + master_transform)

    def _iter_collection_uses_per_path(self, paths, all_transforms, offsets, facecolors, edgecolors):
        if False:
            while True:
                i = 10
        '\n        Compute how many times each raw path object returned by\n        `_iter_collection_raw_paths` would be used when calling\n        `_iter_collection`. This is intended for the backend to decide\n        on the tradeoff between using the paths in-line and storing\n        them once and reusing. Rounds up in case the number of uses\n        is not the same for every path.\n        '
        Npaths = len(paths)
        if Npaths == 0 or len(facecolors) == len(edgecolors) == 0:
            return 0
        Npath_ids = max(Npaths, len(all_transforms))
        N = max(Npath_ids, len(offsets))
        return (N + Npath_ids - 1) // Npath_ids

    def _iter_collection(self, gc, path_ids, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
        if False:
            while True:
                i = 10
        '\n        Helper method (along with `_iter_collection_raw_paths`) to implement\n        `draw_path_collection` in a memory-efficient manner.\n\n        This method yields all of the path, offset and graphics context\n        combinations to draw the path collection.  The caller should already\n        have looped over the results of `_iter_collection_raw_paths` to draw\n        this collection.\n\n        The arguments should be the same as that passed into\n        `draw_path_collection`, with the exception of *path_ids*, which is a\n        list of arbitrary objects that the backend will use to reference one of\n        the paths created in the `_iter_collection_raw_paths` stage.\n\n        Each yielded result is of the form::\n\n           xo, yo, path_id, gc, rgbFace\n\n        where *xo*, *yo* is an offset; *path_id* is one of the elements of\n        *path_ids*; *gc* is a graphics context and *rgbFace* is a color to\n        use for filling the path.\n        '
        Npaths = len(path_ids)
        Noffsets = len(offsets)
        N = max(Npaths, Noffsets)
        Nfacecolors = len(facecolors)
        Nedgecolors = len(edgecolors)
        Nlinewidths = len(linewidths)
        Nlinestyles = len(linestyles)
        Nurls = len(urls)
        if Nfacecolors == 0 and Nedgecolors == 0 or Npaths == 0:
            return
        gc0 = self.new_gc()
        gc0.copy_properties(gc)

        def cycle_or_default(seq, default=None):
            if False:
                print('Hello World!')
            return itertools.cycle(seq) if len(seq) else itertools.repeat(default)
        pathids = cycle_or_default(path_ids)
        toffsets = cycle_or_default(offset_trans.transform(offsets), (0, 0))
        fcs = cycle_or_default(facecolors)
        ecs = cycle_or_default(edgecolors)
        lws = cycle_or_default(linewidths)
        lss = cycle_or_default(linestyles)
        aas = cycle_or_default(antialiaseds)
        urls = cycle_or_default(urls)
        if Nedgecolors == 0:
            gc0.set_linewidth(0.0)
        for (pathid, (xo, yo), fc, ec, lw, ls, aa, url) in itertools.islice(zip(pathids, toffsets, fcs, ecs, lws, lss, aas, urls), N):
            if not (np.isfinite(xo) and np.isfinite(yo)):
                continue
            if Nedgecolors:
                if Nlinewidths:
                    gc0.set_linewidth(lw)
                if Nlinestyles:
                    gc0.set_dashes(*ls)
                if len(ec) == 4 and ec[3] == 0.0:
                    gc0.set_linewidth(0)
                else:
                    gc0.set_foreground(ec)
            if fc is not None and len(fc) == 4 and (fc[3] == 0):
                fc = None
            gc0.set_antialiased(aa)
            if Nurls:
                gc0.set_url(url)
            yield (xo, yo, pathid, gc0, fc)
        gc0.restore()

    def get_image_magnification(self):
        if False:
            print('Hello World!')
        '\n        Get the factor by which to magnify images passed to `draw_image`.\n        Allows a backend to have images at a different resolution to other\n        artists.\n        '
        return 1.0

    def draw_image(self, gc, x, y, im, transform=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Draw an RGBA image.\n\n        Parameters\n        ----------\n        gc : `.GraphicsContextBase`\n            A graphics context with clipping information.\n\n        x : scalar\n            The distance in physical units (i.e., dots or pixels) from the left\n            hand side of the canvas.\n\n        y : scalar\n            The distance in physical units (i.e., dots or pixels) from the\n            bottom side of the canvas.\n\n        im : (N, M, 4) array of `numpy.uint8`\n            An array of RGBA pixels.\n\n        transform : `~matplotlib.transforms.Affine2DBase`\n            If and only if the concrete backend is written such that\n            `option_scale_image` returns ``True``, an affine transformation\n            (i.e., an `.Affine2DBase`) *may* be passed to `draw_image`.  The\n            translation vector of the transformation is given in physical units\n            (i.e., dots or pixels). Note that the transformation does not\n            override *x* and *y*, and has to be applied *before* translating\n            the result by *x* and *y* (this can be accomplished by adding *x*\n            and *y* to the translation vector defined by *transform*).\n        '
        raise NotImplementedError

    def option_image_nocomposite(self):
        if False:
            return 10
        '\n        Return whether image composition by Matplotlib should be skipped.\n\n        Raster backends should usually return False (letting the C-level\n        rasterizer take care of image composition); vector backends should\n        usually return ``not rcParams["image.composite_image"]``.\n        '
        return False

    def option_scale_image(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return whether arbitrary affine transformations in `draw_image` are\n        supported (True for most vector backends).\n        '
        return False

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        if False:
            return 10
        '\n        Draw a TeX instance.\n\n        Parameters\n        ----------\n        gc : `.GraphicsContextBase`\n            The graphics context.\n        x : float\n            The x location of the text in display coords.\n        y : float\n            The y location of the text baseline in display coords.\n        s : str\n            The TeX text string.\n        prop : `~matplotlib.font_manager.FontProperties`\n            The font properties.\n        angle : float\n            The rotation angle in degrees anti-clockwise.\n        mtext : `~matplotlib.text.Text`\n            The original text object to be rendered.\n        '
        self._draw_text_as_path(gc, x, y, s, prop, angle, ismath='TeX')

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if False:
            return 10
        '\n        Draw a text instance.\n\n        Parameters\n        ----------\n        gc : `.GraphicsContextBase`\n            The graphics context.\n        x : float\n            The x location of the text in display coords.\n        y : float\n            The y location of the text baseline in display coords.\n        s : str\n            The text string.\n        prop : `~matplotlib.font_manager.FontProperties`\n            The font properties.\n        angle : float\n            The rotation angle in degrees anti-clockwise.\n        ismath : bool or "TeX"\n            If True, use mathtext parser. If "TeX", use tex for rendering.\n        mtext : `~matplotlib.text.Text`\n            The original text object to be rendered.\n\n        Notes\n        -----\n        **Note for backend implementers:**\n\n        When you are trying to determine if you have gotten your bounding box\n        right (which is what enables the text layout/alignment to work\n        properly), it helps to change the line in text.py::\n\n            if 0: bbox_artist(self, renderer)\n\n        to if 1, and then the actual bounding box will be plotted along with\n        your text.\n        '
        self._draw_text_as_path(gc, x, y, s, prop, angle, ismath)

    def _get_text_path_transform(self, x, y, s, prop, angle, ismath):
        if False:
            while True:
                i = 10
        '\n        Return the text path and transform.\n\n        Parameters\n        ----------\n        x : float\n            The x location of the text in display coords.\n        y : float\n            The y location of the text baseline in display coords.\n        s : str\n            The text to be converted.\n        prop : `~matplotlib.font_manager.FontProperties`\n            The font property.\n        angle : float\n            Angle in degrees to render the text at.\n        ismath : bool or "TeX"\n            If True, use mathtext parser. If "TeX", use tex for rendering.\n        '
        text2path = self._text2path
        fontsize = self.points_to_pixels(prop.get_size_in_points())
        (verts, codes) = text2path.get_text_path(prop, s, ismath=ismath)
        path = Path(verts, codes)
        angle = np.deg2rad(angle)
        if self.flipy():
            (width, height) = self.get_canvas_width_height()
            transform = Affine2D().scale(fontsize / text2path.FONT_SCALE).rotate(angle).translate(x, height - y)
        else:
            transform = Affine2D().scale(fontsize / text2path.FONT_SCALE).rotate(angle).translate(x, y)
        return (path, transform)

    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath):
        if False:
            return 10
        '\n        Draw the text by converting them to paths using `.TextToPath`.\n\n        Parameters\n        ----------\n        gc : `.GraphicsContextBase`\n            The graphics context.\n        x : float\n            The x location of the text in display coords.\n        y : float\n            The y location of the text baseline in display coords.\n        s : str\n            The text to be converted.\n        prop : `~matplotlib.font_manager.FontProperties`\n            The font property.\n        angle : float\n            Angle in degrees to render the text at.\n        ismath : bool or "TeX"\n            If True, use mathtext parser. If "TeX", use tex for rendering.\n        '
        (path, transform) = self._get_text_path_transform(x, y, s, prop, angle, ismath)
        color = gc.get_rgb()
        gc.set_linewidth(0.0)
        self.draw_path(gc, path, transform, rgbFace=color)

    def get_text_width_height_descent(self, s, prop, ismath):
        if False:
            print('Hello World!')
        '\n        Get the width, height, and descent (offset from the bottom to the baseline), in\n        display coords, of the string *s* with `.FontProperties` *prop*.\n\n        Whitespace at the start and the end of *s* is included in the reported width.\n        '
        fontsize = prop.get_size_in_points()
        if ismath == 'TeX':
            return self.get_texmanager().get_text_width_height_descent(s, fontsize, renderer=self)
        dpi = self.points_to_pixels(72)
        if ismath:
            dims = self._text2path.mathtext_parser.parse(s, dpi, prop)
            return dims[0:3]
        flags = self._text2path._get_hinting_flag()
        font = self._text2path._get_font(prop)
        font.set_size(fontsize, dpi)
        font.set_text(s, 0.0, flags=flags)
        (w, h) = font.get_width_height()
        d = font.get_descent()
        w /= 64.0
        h /= 64.0
        d /= 64.0
        return (w, h, d)

    def flipy(self):
        if False:
            return 10
        '\n        Return whether y values increase from top to bottom.\n\n        Note that this only affects drawing of texts.\n        '
        return True

    def get_canvas_width_height(self):
        if False:
            while True:
                i = 10
        'Return the canvas width and height in display coords.'
        return (1, 1)

    def get_texmanager(self):
        if False:
            return 10
        'Return the `.TexManager` instance.'
        if self._texmanager is None:
            self._texmanager = TexManager()
        return self._texmanager

    def new_gc(self):
        if False:
            while True:
                i = 10
        'Return an instance of a `.GraphicsContextBase`.'
        return GraphicsContextBase()

    def points_to_pixels(self, points):
        if False:
            while True:
                i = 10
        "\n        Convert points to display units.\n\n        You need to override this function (unless your backend\n        doesn't have a dpi, e.g., postscript or svg).  Some imaging\n        systems assume some value for pixels per inch::\n\n            points to pixels = points * pixels_per_inch/72 * dpi/72\n\n        Parameters\n        ----------\n        points : float or array-like\n\n        Returns\n        -------\n        Points converted to pixels\n        "
        return points

    def start_rasterizing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Switch to the raster renderer.\n\n        Used by `.MixedModeRenderer`.\n        '

    def stop_rasterizing(self):
        if False:
            print('Hello World!')
        '\n        Switch back to the vector renderer and draw the contents of the raster\n        renderer as an image on the vector renderer.\n\n        Used by `.MixedModeRenderer`.\n        '

    def start_filter(self):
        if False:
            i = 10
            return i + 15
        '\n        Switch to a temporary renderer for image filtering effects.\n\n        Currently only supported by the agg renderer.\n        '

    def stop_filter(self, filter_func):
        if False:
            return 10
        '\n        Switch back to the original renderer.  The contents of the temporary\n        renderer is processed with the *filter_func* and is drawn on the\n        original renderer as an image.\n\n        Currently only supported by the agg renderer.\n        '

    def _draw_disabled(self):
        if False:
            print('Hello World!')
        '\n        Context manager to temporary disable drawing.\n\n        This is used for getting the drawn size of Artists.  This lets us\n        run the draw process to update any Python state but does not pay the\n        cost of the draw_XYZ calls on the canvas.\n        '
        no_ops = {meth_name: lambda *args, **kwargs: None for meth_name in dir(RendererBase) if meth_name.startswith('draw_') or meth_name in ['open_group', 'close_group']}
        return _setattr_cm(self, **no_ops)

class GraphicsContextBase:
    """An abstract base class that provides color, line styles, etc."""

    def __init__(self):
        if False:
            return 10
        self._alpha = 1.0
        self._forced_alpha = False
        self._antialiased = 1
        self._capstyle = CapStyle('butt')
        self._cliprect = None
        self._clippath = None
        self._dashes = (0, None)
        self._joinstyle = JoinStyle('round')
        self._linestyle = 'solid'
        self._linewidth = 1
        self._rgb = (0.0, 0.0, 0.0, 1.0)
        self._hatch = None
        self._hatch_color = colors.to_rgba(rcParams['hatch.color'])
        self._hatch_linewidth = rcParams['hatch.linewidth']
        self._url = None
        self._gid = None
        self._snap = None
        self._sketch = None

    def copy_properties(self, gc):
        if False:
            while True:
                i = 10
        'Copy properties from *gc* to self.'
        self._alpha = gc._alpha
        self._forced_alpha = gc._forced_alpha
        self._antialiased = gc._antialiased
        self._capstyle = gc._capstyle
        self._cliprect = gc._cliprect
        self._clippath = gc._clippath
        self._dashes = gc._dashes
        self._joinstyle = gc._joinstyle
        self._linestyle = gc._linestyle
        self._linewidth = gc._linewidth
        self._rgb = gc._rgb
        self._hatch = gc._hatch
        self._hatch_color = gc._hatch_color
        self._hatch_linewidth = gc._hatch_linewidth
        self._url = gc._url
        self._gid = gc._gid
        self._snap = gc._snap
        self._sketch = gc._sketch

    def restore(self):
        if False:
            return 10
        '\n        Restore the graphics context from the stack - needed only\n        for backends that save graphics contexts on a stack.\n        '

    def get_alpha(self):
        if False:
            print('Hello World!')
        '\n        Return the alpha value used for blending - not supported on all\n        backends.\n        '
        return self._alpha

    def get_antialiased(self):
        if False:
            return 10
        'Return whether the object should try to do antialiased rendering.'
        return self._antialiased

    def get_capstyle(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the `.CapStyle`.'
        return self._capstyle.name

    def get_clip_rectangle(self):
        if False:
            print('Hello World!')
        '\n        Return the clip rectangle as a `~matplotlib.transforms.Bbox` instance.\n        '
        return self._cliprect

    def get_clip_path(self):
        if False:
            return 10
        '\n        Return the clip path in the form (path, transform), where path\n        is a `~.path.Path` instance, and transform is\n        an affine transform to apply to the path before clipping.\n        '
        if self._clippath is not None:
            (tpath, tr) = self._clippath.get_transformed_path_and_affine()
            if np.all(np.isfinite(tpath.vertices)):
                return (tpath, tr)
            else:
                _log.warning('Ill-defined clip_path detected. Returning None.')
                return (None, None)
        return (None, None)

    def get_dashes(self):
        if False:
            return 10
        '\n        Return the dash style as an (offset, dash-list) pair.\n\n        See `.set_dashes` for details.\n\n        Default value is (None, None).\n        '
        return self._dashes

    def get_forced_alpha(self):
        if False:
            while True:
                i = 10
        '\n        Return whether the value given by get_alpha() should be used to\n        override any other alpha-channel values.\n        '
        return self._forced_alpha

    def get_joinstyle(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the `.JoinStyle`.'
        return self._joinstyle.name

    def get_linewidth(self):
        if False:
            i = 10
            return i + 15
        'Return the line width in points.'
        return self._linewidth

    def get_rgb(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a tuple of three or four floats from 0-1.'
        return self._rgb

    def get_url(self):
        if False:
            i = 10
            return i + 15
        'Return a url if one is set, None otherwise.'
        return self._url

    def get_gid(self):
        if False:
            while True:
                i = 10
        'Return the object identifier if one is set, None otherwise.'
        return self._gid

    def get_snap(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the snap setting, which can be:\n\n        * True: snap vertices to the nearest pixel center\n        * False: leave vertices as-is\n        * None: (auto) If the path contains only rectilinear line segments,\n          round to the nearest pixel center\n        '
        return self._snap

    def set_alpha(self, alpha):
        if False:
            print('Hello World!')
        '\n        Set the alpha value used for blending - not supported on all backends.\n\n        If ``alpha=None`` (the default), the alpha components of the\n        foreground and fill colors will be used to set their respective\n        transparencies (where applicable); otherwise, ``alpha`` will override\n        them.\n        '
        if alpha is not None:
            self._alpha = alpha
            self._forced_alpha = True
        else:
            self._alpha = 1.0
            self._forced_alpha = False
        self.set_foreground(self._rgb, isRGBA=True)

    def set_antialiased(self, b):
        if False:
            while True:
                i = 10
        'Set whether object should be drawn with antialiased rendering.'
        self._antialiased = int(bool(b))

    @_docstring.interpd
    def set_capstyle(self, cs):
        if False:
            while True:
                i = 10
        '\n        Set how to draw endpoints of lines.\n\n        Parameters\n        ----------\n        cs : `.CapStyle` or %(CapStyle)s\n        '
        self._capstyle = CapStyle(cs)

    def set_clip_rectangle(self, rectangle):
        if False:
            while True:
                i = 10
        'Set the clip rectangle to a `.Bbox` or None.'
        self._cliprect = rectangle

    def set_clip_path(self, path):
        if False:
            i = 10
            return i + 15
        'Set the clip path to a `.TransformedPath` or None.'
        _api.check_isinstance((transforms.TransformedPath, None), path=path)
        self._clippath = path

    def set_dashes(self, dash_offset, dash_list):
        if False:
            while True:
                i = 10
        '\n        Set the dash style for the gc.\n\n        Parameters\n        ----------\n        dash_offset : float\n            Distance, in points, into the dash pattern at which to\n            start the pattern. It is usually set to 0.\n        dash_list : array-like or None\n            The on-off sequence as points.  None specifies a solid line. All\n            values must otherwise be non-negative (:math:`\\ge 0`).\n\n        Notes\n        -----\n        See p. 666 of the PostScript\n        `Language Reference\n        <https://www.adobe.com/jp/print/postscript/pdfs/PLRM.pdf>`_\n        for more info.\n        '
        if dash_list is not None:
            dl = np.asarray(dash_list)
            if np.any(dl < 0.0):
                raise ValueError('All values in the dash list must be non-negative')
            if dl.size and (not np.any(dl > 0.0)):
                raise ValueError('At least one value in the dash list must be positive')
        self._dashes = (dash_offset, dash_list)

    def set_foreground(self, fg, isRGBA=False):
        if False:
            i = 10
            return i + 15
        '\n        Set the foreground color.\n\n        Parameters\n        ----------\n        fg : color\n        isRGBA : bool\n            If *fg* is known to be an ``(r, g, b, a)`` tuple, *isRGBA* can be\n            set to True to improve performance.\n        '
        if self._forced_alpha and isRGBA:
            self._rgb = fg[:3] + (self._alpha,)
        elif self._forced_alpha:
            self._rgb = colors.to_rgba(fg, self._alpha)
        elif isRGBA:
            self._rgb = fg
        else:
            self._rgb = colors.to_rgba(fg)

    @_docstring.interpd
    def set_joinstyle(self, js):
        if False:
            print('Hello World!')
        '\n        Set how to draw connections between line segments.\n\n        Parameters\n        ----------\n        js : `.JoinStyle` or %(JoinStyle)s\n        '
        self._joinstyle = JoinStyle(js)

    def set_linewidth(self, w):
        if False:
            print('Hello World!')
        'Set the linewidth in points.'
        self._linewidth = float(w)

    def set_url(self, url):
        if False:
            i = 10
            return i + 15
        'Set the url for links in compatible backends.'
        self._url = url

    def set_gid(self, id):
        if False:
            for i in range(10):
                print('nop')
        'Set the id.'
        self._gid = id

    def set_snap(self, snap):
        if False:
            while True:
                i = 10
        '\n        Set the snap setting which may be:\n\n        * True: snap vertices to the nearest pixel center\n        * False: leave vertices as-is\n        * None: (auto) If the path contains only rectilinear line segments,\n          round to the nearest pixel center\n        '
        self._snap = snap

    def set_hatch(self, hatch):
        if False:
            while True:
                i = 10
        'Set the hatch style (for fills).'
        self._hatch = hatch

    def get_hatch(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the current hatch style.'
        return self._hatch

    def get_hatch_path(self, density=6.0):
        if False:
            for i in range(10):
                print('nop')
        'Return a `.Path` for the current hatch.'
        hatch = self.get_hatch()
        if hatch is None:
            return None
        return Path.hatch(hatch, density)

    def get_hatch_color(self):
        if False:
            print('Hello World!')
        'Get the hatch color.'
        return self._hatch_color

    def set_hatch_color(self, hatch_color):
        if False:
            return 10
        'Set the hatch color.'
        self._hatch_color = hatch_color

    def get_hatch_linewidth(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the hatch linewidth.'
        return self._hatch_linewidth

    def get_sketch_params(self):
        if False:
            print('Hello World!')
        '\n        Return the sketch parameters for the artist.\n\n        Returns\n        -------\n        tuple or `None`\n\n            A 3-tuple with the following elements:\n\n            * ``scale``: The amplitude of the wiggle perpendicular to the\n              source line.\n            * ``length``: The length of the wiggle along the line.\n            * ``randomness``: The scale factor by which the length is\n              shrunken or expanded.\n\n            May return `None` if no sketch parameters were set.\n        '
        return self._sketch

    def set_sketch_params(self, scale=None, length=None, randomness=None):
        if False:
            while True:
                i = 10
        '\n        Set the sketch parameters.\n\n        Parameters\n        ----------\n        scale : float, optional\n            The amplitude of the wiggle perpendicular to the source line, in\n            pixels.  If scale is `None`, or not provided, no sketch filter will\n            be provided.\n        length : float, default: 128\n            The length of the wiggle along the line, in pixels.\n        randomness : float, default: 16\n            The scale factor by which the length is shrunken or expanded.\n        '
        self._sketch = None if scale is None else (scale, length or 128.0, randomness or 16.0)

class TimerBase:
    """
    A base class for providing timer events, useful for things animations.
    Backends need to implement a few specific methods in order to use their
    own timing mechanisms so that the timer events are integrated into their
    event loops.

    Subclasses must override the following methods:

    - ``_timer_start``: Backend-specific code for starting the timer.
    - ``_timer_stop``: Backend-specific code for stopping the timer.

    Subclasses may additionally override the following methods:

    - ``_timer_set_single_shot``: Code for setting the timer to single shot
      operating mode, if supported by the timer object.  If not, the `Timer`
      class itself will store the flag and the ``_on_timer`` method should be
      overridden to support such behavior.

    - ``_timer_set_interval``: Code for setting the interval on the timer, if
      there is a method for doing so on the timer object.

    - ``_on_timer``: The internal function that any timer object should call,
      which will handle the task of running all callbacks that have been set.
    """

    def __init__(self, interval=None, callbacks=None):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        interval : int, default: 1000ms\n            The time between timer events in milliseconds.  Will be stored as\n            ``timer.interval``.\n        callbacks : list[tuple[callable, tuple, dict]]\n            List of (func, args, kwargs) tuples that will be called upon timer\n            events.  This list is accessible as ``timer.callbacks`` and can be\n            manipulated directly, or the functions `~.TimerBase.add_callback`\n            and `~.TimerBase.remove_callback` can be used.\n        '
        self.callbacks = [] if callbacks is None else callbacks.copy()
        self.interval = 1000 if interval is None else interval
        self.single_shot = False

    def __del__(self):
        if False:
            while True:
                i = 10
        'Need to stop timer and possibly disconnect timer.'
        self._timer_stop()

    @_api.delete_parameter('3.9', 'interval', alternative='timer.interval')
    def start(self, interval=None):
        if False:
            i = 10
            return i + 15
        '\n        Start the timer object.\n\n        Parameters\n        ----------\n        interval : int, optional\n            Timer interval in milliseconds; overrides a previously set interval\n            if provided.\n        '
        if interval is not None:
            self.interval = interval
        self._timer_start()

    def stop(self):
        if False:
            print('Hello World!')
        'Stop the timer.'
        self._timer_stop()

    def _timer_start(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _timer_stop(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def interval(self):
        if False:
            while True:
                i = 10
        'The time between timer events, in milliseconds.'
        return self._interval

    @interval.setter
    def interval(self, interval):
        if False:
            while True:
                i = 10
        interval = max(int(interval), 1)
        self._interval = interval
        self._timer_set_interval()

    @property
    def single_shot(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether this timer should stop after a single run.'
        return self._single

    @single_shot.setter
    def single_shot(self, ss):
        if False:
            print('Hello World!')
        self._single = ss
        self._timer_set_single_shot()

    def add_callback(self, func, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Register *func* to be called by timer when the event fires. Any\n        additional arguments provided will be passed to *func*.\n\n        This function returns *func*, which makes it possible to use it as a\n        decorator.\n        '
        self.callbacks.append((func, args, kwargs))
        return func

    def remove_callback(self, func, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Remove *func* from list of callbacks.\n\n        *args* and *kwargs* are optional and used to distinguish between copies\n        of the same function registered to be called with different arguments.\n        This behavior is deprecated.  In the future, ``*args, **kwargs`` won't\n        be considered anymore; to keep a specific callback removable by itself,\n        pass it to `add_callback` as a `functools.partial` object.\n        "
        if args or kwargs:
            _api.warn_deprecated('3.1', message='In a future version, Timer.remove_callback will not take *args, **kwargs anymore, but remove all callbacks where the callable matches; to keep a specific callback removable by itself, pass it to add_callback as a functools.partial object.')
            self.callbacks.remove((func, args, kwargs))
        else:
            funcs = [c[0] for c in self.callbacks]
            if func in funcs:
                self.callbacks.pop(funcs.index(func))

    def _timer_set_interval(self):
        if False:
            print('Hello World!')
        'Used to set interval on underlying timer object.'

    def _timer_set_single_shot(self):
        if False:
            i = 10
            return i + 15
        'Used to set single shot on underlying timer object.'

    def _on_timer(self):
        if False:
            while True:
                i = 10
        '\n        Runs all function that have been registered as callbacks. Functions\n        can return False (or 0) if they should not be called any more. If there\n        are no callbacks, the timer is automatically stopped.\n        '
        for (func, args, kwargs) in self.callbacks:
            ret = func(*args, **kwargs)
            if ret == 0:
                self.callbacks.remove((func, args, kwargs))
        if len(self.callbacks) == 0:
            self.stop()

class Event:
    """
    A Matplotlib event.

    The following attributes are defined and shown with their default values.
    Subclasses may define additional attributes.

    Attributes
    ----------
    name : str
        The event name.
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance generating the event.
    guiEvent
        The GUI event that triggered the Matplotlib event.
    """

    def __init__(self, name, canvas, guiEvent=None):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.canvas = canvas
        self._guiEvent = guiEvent
        self._guiEvent_deleted = False

    def _process(self):
        if False:
            for i in range(10):
                print('nop')
        'Process this event on ``self.canvas``, then unset ``guiEvent``.'
        self.canvas.callbacks.process(self.name, self)
        self._guiEvent_deleted = True

    @property
    def guiEvent(self):
        if False:
            while True:
                i = 10
        if self._guiEvent_deleted:
            _api.warn_deprecated('3.8', message='Accessing guiEvent outside of the original GUI event handler is unsafe and deprecated since %(since)s; in the future, the attribute will be set to None after quitting the event handler.  You may separately record the value of the guiEvent attribute at your own risk.')
        return self._guiEvent

class DrawEvent(Event):
    """
    An event triggered by a draw operation on the canvas.

    In most backends, callbacks subscribed to this event will be fired after
    the rendering is complete but before the screen is updated. Any extra
    artists drawn to the canvas's renderer will be reflected without an
    explicit call to ``blit``.

    .. warning::

       Calling ``canvas.draw`` and ``canvas.blit`` in these callbacks may
       not be safe with all backends and may cause infinite recursion.

    A DrawEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    renderer : `RendererBase`
        The renderer for the draw event.
    """

    def __init__(self, name, canvas, renderer):
        if False:
            return 10
        super().__init__(name, canvas)
        self.renderer = renderer

class ResizeEvent(Event):
    """
    An event triggered by a canvas resize.

    A ResizeEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    width : int
        Width of the canvas in pixels.
    height : int
        Height of the canvas in pixels.
    """

    def __init__(self, name, canvas):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name, canvas)
        (self.width, self.height) = canvas.get_width_height()

class CloseEvent(Event):
    """An event triggered by a figure being closed."""

class LocationEvent(Event):
    """
    An event that has a screen location.

    A LocationEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    x, y : int or None
        Event location in pixels from bottom left of canvas.
    inaxes : `~matplotlib.axes.Axes` or None
        The `~.axes.Axes` instance over which the mouse is, if any.
    xdata, ydata : float or None
        Data coordinates of the mouse within *inaxes*, or *None* if the mouse
        is not over an Axes.
    modifiers : frozenset
        The keyboard modifiers currently being pressed (except for KeyEvent).
    """
    _lastevent = None
    lastevent = _api.deprecated('3.8')(_api.classproperty(lambda cls: cls._lastevent))
    _last_axes_ref = None

    def __init__(self, name, canvas, x, y, guiEvent=None, *, modifiers=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name, canvas, guiEvent=guiEvent)
        self.x = int(x) if x is not None else x
        self.y = int(y) if y is not None else y
        self.inaxes = None
        self.xdata = None
        self.ydata = None
        self.modifiers = frozenset(modifiers if modifiers is not None else [])
        if x is None or y is None:
            return
        self._set_inaxes(self.canvas.inaxes((x, y)) if self.canvas.mouse_grabber is None else self.canvas.mouse_grabber, (x, y))

    def _set_inaxes(self, inaxes, xy=None):
        if False:
            for i in range(10):
                print('nop')
        self.inaxes = inaxes
        if inaxes is not None:
            try:
                (self.xdata, self.ydata) = inaxes.transData.inverted().transform(xy if xy is not None else (self.x, self.y))
            except ValueError:
                pass

class MouseButton(IntEnum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    BACK = 8
    FORWARD = 9

class MouseEvent(LocationEvent):
    """
    A mouse event ('button_press_event', 'button_release_event', 'scroll_event', 'motion_notify_event').

    A MouseEvent has a number of special attributes in addition to those
    defined by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    button : None or `MouseButton` or {'up', 'down'}
        The button pressed. 'up' and 'down' are used for scroll events.

        Note that LEFT and RIGHT actually refer to the "primary" and
        "secondary" buttons, i.e. if the user inverts their left and right
        buttons ("left-handed setting") then the LEFT button will be the one
        physically on the right.

        If this is unset, *name* is "scroll_event", and *step* is nonzero, then
        this will be set to "up" or "down" depending on the sign of *step*.

    key : None or str
        The key pressed when the mouse event triggered, e.g. 'shift'.
        See `KeyEvent`.

        .. warning::
           This key is currently obtained from the last 'key_press_event' or
           'key_release_event' that occurred within the canvas.  Thus, if the
           last change of keyboard state occurred while the canvas did not have
           focus, this attribute will be wrong.  On the other hand, the
           ``modifiers`` attribute should always be correct, but it can only
           report on modifier keys.

    step : float
        The number of scroll steps (positive for 'up', negative for 'down').
        This applies only to 'scroll_event' and defaults to 0 otherwise.

    dblclick : bool
        Whether the event is a double-click. This applies only to
        'button_press_event' and is False otherwise. In particular, it's
        not used in 'button_release_event'.

    Examples
    --------
    ::

        def on_press(event):
            print('you pressed', event.button, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect('button_press_event', on_press)
    """

    def __init__(self, name, canvas, x, y, button=None, key=None, step=0, dblclick=False, guiEvent=None, *, modifiers=None):
        if False:
            i = 10
            return i + 15
        super().__init__(name, canvas, x, y, guiEvent=guiEvent, modifiers=modifiers)
        if button in MouseButton.__members__.values():
            button = MouseButton(button)
        if name == 'scroll_event' and button is None:
            if step > 0:
                button = 'up'
            elif step < 0:
                button = 'down'
        self.button = button
        self.key = key
        self.step = step
        self.dblclick = dblclick

    def __str__(self):
        if False:
            print('Hello World!')
        return f'{self.name}: xy=({self.x}, {self.y}) xydata=({self.xdata}, {self.ydata}) button={self.button} dblclick={self.dblclick} inaxes={self.inaxes}'

class PickEvent(Event):
    """
    A pick event.

    This event is fired when the user picks a location on the canvas
    sufficiently close to an artist that has been made pickable with
    `.Artist.set_picker`.

    A PickEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    mouseevent : `MouseEvent`
        The mouse event that generated the pick.
    artist : `~matplotlib.artist.Artist`
        The picked artist.  Note that artists are not pickable by default
        (see `.Artist.set_picker`).
    other
        Additional attributes may be present depending on the type of the
        picked object; e.g., a `.Line2D` pick may define different extra
        attributes than a `.PatchCollection` pick.

    Examples
    --------
    Bind a function ``on_pick()`` to pick events, that prints the coordinates
    of the picked data point::

        ax.plot(np.rand(100), 'o', picker=5)  # 5 points tolerance

        def on_pick(event):
            line = event.artist
            xdata, ydata = line.get_data()
            ind = event.ind
            print(f'on pick line: {xdata[ind]:.3f}, {ydata[ind]:.3f}')

        cid = fig.canvas.mpl_connect('pick_event', on_pick)
    """

    def __init__(self, name, canvas, mouseevent, artist, guiEvent=None, **kwargs):
        if False:
            return 10
        if guiEvent is None:
            guiEvent = mouseevent.guiEvent
        super().__init__(name, canvas, guiEvent)
        self.mouseevent = mouseevent
        self.artist = artist
        self.__dict__.update(kwargs)

class KeyEvent(LocationEvent):
    """
    A key event (key press, key release).

    A KeyEvent has a number of special attributes in addition to those defined
    by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    key : None or str
        The key(s) pressed. Could be *None*, a single case sensitive Unicode
        character ("g", "G", "#", etc.), a special key ("control", "shift",
        "f1", "up", etc.) or a combination of the above (e.g., "ctrl+alt+g",
        "ctrl+alt+G").

    Notes
    -----
    Modifier keys will be prefixed to the pressed key and will be in the order
    "ctrl", "alt", "super". The exception to this rule is when the pressed key
    is itself a modifier key, therefore "ctrl+alt" and "alt+control" can both
    be valid key values.

    Examples
    --------
    ::

        def on_key(event):
            print('you pressed', event.key, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
    """

    def __init__(self, name, canvas, key, x=0, y=0, guiEvent=None):
        if False:
            while True:
                i = 10
        super().__init__(name, canvas, x, y, guiEvent=guiEvent)
        self.key = key

def _key_handler(event):
    if False:
        return 10
    if event.name == 'key_press_event':
        event.canvas._key = event.key
    elif event.name == 'key_release_event':
        event.canvas._key = None

def _mouse_handler(event):
    if False:
        for i in range(10):
            print('nop')
    if event.name == 'button_press_event':
        event.canvas._button = event.button
    elif event.name == 'button_release_event':
        event.canvas._button = None
    elif event.name == 'motion_notify_event' and event.button is None:
        event.button = event.canvas._button
    if event.key is None:
        event.key = event.canvas._key
    if event.name == 'motion_notify_event':
        last_ref = LocationEvent._last_axes_ref
        last_axes = last_ref() if last_ref else None
        if last_axes != event.inaxes:
            if last_axes is not None:
                try:
                    leave_event = LocationEvent('axes_leave_event', last_axes.figure.canvas, event.x, event.y, event.guiEvent, modifiers=event.modifiers)
                    leave_event._set_inaxes(last_axes)
                    last_axes.figure.canvas.callbacks.process('axes_leave_event', leave_event)
                except Exception:
                    pass
            if event.inaxes is not None:
                event.canvas.callbacks.process('axes_enter_event', event)
        LocationEvent._last_axes_ref = weakref.ref(event.inaxes) if event.inaxes else None
        LocationEvent._lastevent = None if event.name == 'figure_leave_event' else event

def _get_renderer(figure, print_method=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the renderer that would be used to save a `.Figure`.\n\n    If you need a renderer without any active draw methods use\n    renderer._draw_disabled to temporary patch them out at your call site.\n    '

    class Done(Exception):
        pass

    def _draw(renderer):
        if False:
            for i in range(10):
                print('nop')
        raise Done(renderer)
    with cbook._setattr_cm(figure, draw=_draw), ExitStack() as stack:
        if print_method is None:
            fmt = figure.canvas.get_default_filetype()
            print_method = stack.enter_context(figure.canvas._switch_canvas_and_return_print_method(fmt))
        try:
            print_method(io.BytesIO())
        except Done as exc:
            (renderer,) = exc.args
            return renderer
        else:
            raise RuntimeError(f'{print_method} did not call Figure.draw, so no renderer is available')

def _no_output_draw(figure):
    if False:
        while True:
            i = 10
    figure.draw_without_rendering()

def _is_non_interactive_terminal_ipython(ip):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return whether we are in a terminal IPython, but non interactive.\n\n    When in _terminal_ IPython, ip.parent will have and `interact` attribute,\n    if this attribute is False we do not setup eventloop integration as the\n    user will _not_ interact with IPython. In all other case (ZMQKernel, or is\n    interactive), we do.\n    '
    return hasattr(ip, 'parent') and ip.parent is not None and (getattr(ip.parent, 'interact', None) is False)

@contextmanager
def _allow_interrupt(prepare_notifier, handle_sigint):
    if False:
        for i in range(10):
            print('nop')
    "\n    A context manager that allows terminating a plot by sending a SIGINT.  It\n    is necessary because the running backend prevents the Python interpreter\n    from running and processing signals (i.e., to raise a KeyboardInterrupt).\n    To solve this, one needs to somehow wake up the interpreter and make it\n    close the plot window.  We do this by using the signal.set_wakeup_fd()\n    function which organizes a write of the signal number into a socketpair.\n    A backend-specific function, *prepare_notifier*, arranges to listen to\n    the pair's read socket while the event loop is running.  (If it returns a\n    notifier object, that object is kept alive while the context manager runs.)\n\n    If SIGINT was indeed caught, after exiting the on_signal() function the\n    interpreter reacts to the signal according to the handler function which\n    had been set up by a signal.signal() call; here, we arrange to call the\n    backend-specific *handle_sigint* function.  Finally, we call the old SIGINT\n    handler with the same arguments that were given to our custom handler.\n\n    We do this only if the old handler for SIGINT was not None, which means\n    that a non-python handler was installed, i.e. in Julia, and not SIG_IGN\n    which means we should ignore the interrupts.\n\n    Parameters\n    ----------\n    prepare_notifier : Callable[[socket.socket], object]\n    handle_sigint : Callable[[], object]\n    "
    old_sigint_handler = signal.getsignal(signal.SIGINT)
    if old_sigint_handler in (None, signal.SIG_IGN, signal.SIG_DFL):
        yield
        return
    handler_args = None
    (wsock, rsock) = socket.socketpair()
    wsock.setblocking(False)
    rsock.setblocking(False)
    old_wakeup_fd = signal.set_wakeup_fd(wsock.fileno())
    notifier = prepare_notifier(rsock)

    def save_args_and_handle_sigint(*args):
        if False:
            print('Hello World!')
        nonlocal handler_args
        handler_args = args
        handle_sigint()
    signal.signal(signal.SIGINT, save_args_and_handle_sigint)
    try:
        yield
    finally:
        wsock.close()
        rsock.close()
        signal.set_wakeup_fd(old_wakeup_fd)
        signal.signal(signal.SIGINT, old_sigint_handler)
        if handler_args is not None:
            old_sigint_handler(*handler_args)

class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level figure instance.
    """
    required_interactive_framework = None
    manager_class = _api.classproperty(lambda cls: FigureManagerBase)
    events = ['resize_event', 'draw_event', 'key_press_event', 'key_release_event', 'button_press_event', 'button_release_event', 'scroll_event', 'motion_notify_event', 'pick_event', 'figure_enter_event', 'figure_leave_event', 'axes_enter_event', 'axes_leave_event', 'close_event']
    fixed_dpi = None
    filetypes = _default_filetypes

    @_api.classproperty
    def supports_blit(cls):
        if False:
            return 10
        'If this Canvas sub-class supports blitting.'
        return hasattr(cls, 'copy_from_bbox') and hasattr(cls, 'restore_region')

    def __init__(self, figure=None):
        if False:
            return 10
        from matplotlib.figure import Figure
        self._fix_ipython_backend2gui()
        self._is_idle_drawing = True
        self._is_saving = False
        if figure is None:
            figure = Figure()
        figure.set_canvas(self)
        self.figure = figure
        self.manager = None
        self.widgetlock = widgets.LockDraw()
        self._button = None
        self._key = None
        self.mouse_grabber = None
        self.toolbar = None
        self._is_idle_drawing = False
        figure._original_dpi = figure.dpi
        self._device_pixel_ratio = 1
        super().__init__()
    callbacks = property(lambda self: self.figure._canvas_callbacks)
    button_pick_id = property(lambda self: self.figure._button_pick_id)
    scroll_pick_id = property(lambda self: self.figure._scroll_pick_id)

    @classmethod
    @functools.cache
    def _fix_ipython_backend2gui(cls):
        if False:
            i = 10
            return i + 15
        if sys.modules.get('IPython') is None:
            return
        import IPython
        ip = IPython.get_ipython()
        if not ip:
            return
        from IPython.core import pylabtools as pt
        if not hasattr(pt, 'backend2gui') or not hasattr(ip, 'enable_matplotlib'):
            return
        backend2gui_rif = {'qt': 'qt', 'gtk3': 'gtk3', 'gtk4': 'gtk4', 'wx': 'wx', 'macosx': 'osx'}.get(cls.required_interactive_framework)
        if backend2gui_rif:
            if _is_non_interactive_terminal_ipython(ip):
                ip.enable_gui(backend2gui_rif)

    @classmethod
    def new_manager(cls, figure, num):
        if False:
            i = 10
            return i + 15
        '\n        Create a new figure manager for *figure*, using this canvas class.\n\n        Notes\n        -----\n        This method should not be reimplemented in subclasses.  If\n        custom manager creation logic is needed, please reimplement\n        ``FigureManager.create_with_canvas``.\n        '
        return cls.manager_class.create_with_canvas(cls, figure, num)

    @contextmanager
    def _idle_draw_cntx(self):
        if False:
            while True:
                i = 10
        self._is_idle_drawing = True
        try:
            yield
        finally:
            self._is_idle_drawing = False

    def is_saving(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return whether the renderer is in the process of saving\n        to a file, rather than rendering for an on-screen buffer.\n        '
        return self._is_saving

    def blit(self, bbox=None):
        if False:
            return 10
        'Blit the canvas in bbox (default entire canvas).'

    def inaxes(self, xy):
        if False:
            print('Hello World!')
        '\n        Return the topmost visible `~.axes.Axes` containing the point *xy*.\n\n        Parameters\n        ----------\n        xy : (float, float)\n            (x, y) pixel positions from left/bottom of the canvas.\n\n        Returns\n        -------\n        `~matplotlib.axes.Axes` or None\n            The topmost visible Axes containing the point, or None if there\n            is no Axes at the point.\n        '
        axes_list = [a for a in self.figure.get_axes() if a.patch.contains_point(xy) and a.get_visible()]
        if axes_list:
            axes = cbook._topmost_artist(axes_list)
        else:
            axes = None
        return axes

    def grab_mouse(self, ax):
        if False:
            print('Hello World!')
        '\n        Set the child `~.axes.Axes` which is grabbing the mouse events.\n\n        Usually called by the widgets themselves. It is an error to call this\n        if the mouse is already grabbed by another Axes.\n        '
        if self.mouse_grabber not in (None, ax):
            raise RuntimeError('Another Axes already grabs mouse input')
        self.mouse_grabber = ax

    def release_mouse(self, ax):
        if False:
            while True:
                i = 10
        "\n        Release the mouse grab held by the `~.axes.Axes` *ax*.\n\n        Usually called by the widgets. It is ok to call this even if *ax*\n        doesn't have the mouse grab currently.\n        "
        if self.mouse_grabber is ax:
            self.mouse_grabber = None

    def set_cursor(self, cursor):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the current cursor.\n\n        This may have no effect if the backend does not display anything.\n\n        If required by the backend, this method should trigger an update in\n        the backend event loop after the cursor is set, as this method may be\n        called e.g. before a long-running task during which the GUI is not\n        updated.\n\n        Parameters\n        ----------\n        cursor : `.Cursors`\n            The cursor to display over the canvas. Note: some backends may\n            change the cursor for the entire window.\n        '

    def draw(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Render the `.Figure`.\n\n        This method must walk the artist tree, even if no output is produced,\n        because it triggers deferred work that users may want to access\n        before saving output to disk. For example computing limits,\n        auto-limits, and tick values.\n        '

    def draw_idle(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Request a widget redraw once control returns to the GUI event loop.\n\n        Even if multiple calls to `draw_idle` occur before control returns\n        to the GUI event loop, the figure will only be rendered once.\n\n        Notes\n        -----\n        Backends may choose to override the method and implement their own\n        strategy to prevent multiple renderings.\n\n        '
        if not self._is_idle_drawing:
            with self._idle_draw_cntx():
                self.draw(*args, **kwargs)

    @property
    def device_pixel_ratio(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The ratio of physical to logical pixels used for the canvas on screen.\n\n        By default, this is 1, meaning physical and logical pixels are the same\n        size. Subclasses that support High DPI screens may set this property to\n        indicate that said ratio is different. All Matplotlib interaction,\n        unless working directly with the canvas, remains in logical pixels.\n\n        '
        return self._device_pixel_ratio

    def _set_device_pixel_ratio(self, ratio):
        if False:
            i = 10
            return i + 15
        '\n        Set the ratio of physical to logical pixels used for the canvas.\n\n        Subclasses that support High DPI screens can set this property to\n        indicate that said ratio is different. The canvas itself will be\n        created at the physical size, while the client side will use the\n        logical size. Thus the DPI of the Figure will change to be scaled by\n        this ratio. Implementations that support High DPI screens should use\n        physical pixels for events so that transforms back to Axes space are\n        correct.\n\n        By default, this is 1, meaning physical and logical pixels are the same\n        size.\n\n        Parameters\n        ----------\n        ratio : float\n            The ratio of logical to physical pixels used for the canvas.\n\n        Returns\n        -------\n        bool\n            Whether the ratio has changed. Backends may interpret this as a\n            signal to resize the window, repaint the canvas, or change any\n            other relevant properties.\n        '
        if self._device_pixel_ratio == ratio:
            return False
        dpi = ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)
        self._device_pixel_ratio = ratio
        return True

    def get_width_height(self, *, physical=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the figure width and height in integral points or pixels.\n\n        When the figure is used on High DPI screens (and the backend supports\n        it), the truncation to integers occurs after scaling by the device\n        pixel ratio.\n\n        Parameters\n        ----------\n        physical : bool, default: False\n            Whether to return true physical pixels or logical pixels. Physical\n            pixels may be used by backends that support HiDPI, but still\n            configure the canvas using its actual size.\n\n        Returns\n        -------\n        width, height : int\n            The size of the figure, in points or pixels, depending on the\n            backend.\n        '
        return tuple((int(size / (1 if physical else self.device_pixel_ratio)) for size in self.figure.bbox.max))

    @classmethod
    def get_supported_filetypes(cls):
        if False:
            return 10
        'Return dict of savefig file formats supported by this backend.'
        return cls.filetypes

    @classmethod
    def get_supported_filetypes_grouped(cls):
        if False:
            return 10
        "\n        Return a dict of savefig file formats supported by this backend,\n        where the keys are a file type name, such as 'Joint Photographic\n        Experts Group', and the values are a list of filename extensions used\n        for that filetype, such as ['jpg', 'jpeg'].\n        "
        groupings = {}
        for (ext, name) in cls.filetypes.items():
            groupings.setdefault(name, []).append(ext)
            groupings[name].sort()
        return groupings

    @contextmanager
    def _switch_canvas_and_return_print_method(self, fmt, backend=None):
        if False:
            i = 10
            return i + 15
        "\n        Context manager temporarily setting the canvas for saving the figure::\n\n            with canvas._switch_canvas_and_return_print_method(fmt, backend) \\\n                    as print_method:\n                # ``print_method`` is a suitable ``print_{fmt}`` method, and\n                # the figure's canvas is temporarily switched to the method's\n                # canvas within the with... block.  ``print_method`` is also\n                # wrapped to suppress extra kwargs passed by ``print_figure``.\n\n        Parameters\n        ----------\n        fmt : str\n            If *backend* is None, then determine a suitable canvas class for\n            saving to format *fmt* -- either the current canvas class, if it\n            supports *fmt*, or whatever `get_registered_canvas_class` returns;\n            switch the figure canvas to that canvas class.\n        backend : str or None, default: None\n            If not None, switch the figure canvas to the ``FigureCanvas`` class\n            of the given backend.\n        "
        canvas = None
        if backend is not None:
            canvas_class = importlib.import_module(cbook._backend_module_name(backend)).FigureCanvas
            if not hasattr(canvas_class, f'print_{fmt}'):
                raise ValueError(f'The {backend!r} backend does not support {fmt} output')
            canvas = canvas_class(self.figure)
        elif hasattr(self, f'print_{fmt}'):
            canvas = self
        else:
            canvas_class = get_registered_canvas_class(fmt)
            if canvas_class is None:
                raise ValueError('Format {!r} is not supported (supported formats: {})'.format(fmt, ', '.join(sorted(self.get_supported_filetypes()))))
            canvas = canvas_class(self.figure)
        canvas._is_saving = self._is_saving
        meth = getattr(canvas, f'print_{fmt}')
        mod = meth.func.__module__ if hasattr(meth, 'func') else meth.__module__
        if mod.startswith(('matplotlib.', 'mpl_toolkits.')):
            optional_kws = {'dpi', 'facecolor', 'edgecolor', 'orientation', 'bbox_inches_restore'}
            skip = optional_kws - {*inspect.signature(meth).parameters}
            print_method = functools.wraps(meth)(lambda *args, **kwargs: meth(*args, **{k: v for (k, v) in kwargs.items() if k not in skip}))
        else:
            print_method = meth
        try:
            yield print_method
        finally:
            self.figure.canvas = self

    def print_figure(self, filename, dpi=None, facecolor=None, edgecolor=None, orientation='portrait', format=None, *, bbox_inches=None, pad_inches=None, bbox_extra_artists=None, backend=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Render the figure to hardcopy. Set the figure patch face and edge\n        colors.  This is useful because some of the GUIs have a gray figure\n        face color background and you\'ll probably want to override this on\n        hardcopy.\n\n        Parameters\n        ----------\n        filename : str or path-like or file-like\n            The file where the figure is saved.\n\n        dpi : float, default: :rc:`savefig.dpi`\n            The dots per inch to save the figure in.\n\n        facecolor : color or \'auto\', default: :rc:`savefig.facecolor`\n            The facecolor of the figure.  If \'auto\', use the current figure\n            facecolor.\n\n        edgecolor : color or \'auto\', default: :rc:`savefig.edgecolor`\n            The edgecolor of the figure.  If \'auto\', use the current figure\n            edgecolor.\n\n        orientation : {\'landscape\', \'portrait\'}, default: \'portrait\'\n            Only currently applies to PostScript printing.\n\n        format : str, optional\n            Force a specific file format. If not given, the format is inferred\n            from the *filename* extension, and if that fails from\n            :rc:`savefig.format`.\n\n        bbox_inches : \'tight\' or `.Bbox`, default: :rc:`savefig.bbox`\n            Bounding box in inches: only the given portion of the figure is\n            saved.  If \'tight\', try to figure out the tight bbox of the figure.\n\n        pad_inches : float or \'layout\', default: :rc:`savefig.pad_inches`\n            Amount of padding in inches around the figure when bbox_inches is\n            \'tight\'. If \'layout\' use the padding from the constrained or\n            compressed layout engine; ignored if one of those engines is not in\n            use.\n\n        bbox_extra_artists : list of `~matplotlib.artist.Artist`, optional\n            A list of extra artists that will be considered when the\n            tight bbox is calculated.\n\n        backend : str, optional\n            Use a non-default backend to render the file, e.g. to render a\n            png file with the "cairo" backend rather than the default "agg",\n            or a pdf file with the "pgf" backend rather than the default\n            "pdf".  Note that the default backend is normally sufficient.  See\n            :ref:`the-builtin-backends` for a list of valid backends for each\n            file format.  Custom backends can be referenced as "module://...".\n        '
        if format is None:
            if isinstance(filename, os.PathLike):
                filename = os.fspath(filename)
            if isinstance(filename, str):
                format = os.path.splitext(filename)[1][1:]
            if format is None or format == '':
                format = self.get_default_filetype()
                if isinstance(filename, str):
                    filename = filename.rstrip('.') + '.' + format
        format = format.lower()
        if dpi is None:
            dpi = rcParams['savefig.dpi']
        if dpi == 'figure':
            dpi = getattr(self.figure, '_original_dpi', self.figure.dpi)
        if kwargs.get('papertype') == 'auto':
            _api.warn_deprecated('3.8', name="papertype='auto'", addendum="Pass an explicit paper type, 'figure', or omit the *papertype* argument entirely.")
        with cbook._setattr_cm(self, manager=None), self._switch_canvas_and_return_print_method(format, backend) as print_method, cbook._setattr_cm(self.figure, dpi=dpi), cbook._setattr_cm(self.figure.canvas, _device_pixel_ratio=1), cbook._setattr_cm(self.figure.canvas, _is_saving=True), ExitStack() as stack:
            for prop in ['facecolor', 'edgecolor']:
                color = locals()[prop]
                if color is None:
                    color = rcParams[f'savefig.{prop}']
                if not cbook._str_equal(color, 'auto'):
                    stack.enter_context(self.figure._cm_set(**{prop: color}))
            if bbox_inches is None:
                bbox_inches = rcParams['savefig.bbox']
            layout_engine = self.figure.get_layout_engine()
            if layout_engine is not None or bbox_inches == 'tight':
                renderer = _get_renderer(self.figure, functools.partial(print_method, orientation=orientation))
                with getattr(renderer, '_draw_disabled', nullcontext)():
                    self.figure.draw(renderer)
            if bbox_inches:
                if bbox_inches == 'tight':
                    bbox_inches = self.figure.get_tightbbox(renderer, bbox_extra_artists=bbox_extra_artists)
                    if isinstance(layout_engine, ConstrainedLayoutEngine) and pad_inches == 'layout':
                        h_pad = layout_engine.get()['h_pad']
                        w_pad = layout_engine.get()['w_pad']
                    else:
                        if pad_inches in [None, 'layout']:
                            pad_inches = rcParams['savefig.pad_inches']
                        h_pad = w_pad = pad_inches
                    bbox_inches = bbox_inches.padded(w_pad, h_pad)
                restore_bbox = _tight_bbox.adjust_bbox(self.figure, bbox_inches, self.figure.canvas.fixed_dpi)
                _bbox_inches_restore = (bbox_inches, restore_bbox)
            else:
                _bbox_inches_restore = None
            stack.enter_context(self.figure._cm_set(layout_engine='none'))
            try:
                with cbook._setattr_cm(self.figure, dpi=dpi):
                    result = print_method(filename, facecolor=facecolor, edgecolor=edgecolor, orientation=orientation, bbox_inches_restore=_bbox_inches_restore, **kwargs)
            finally:
                if bbox_inches and restore_bbox:
                    restore_bbox()
            return result

    @classmethod
    def get_default_filetype(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the default savefig file format as specified in\n        :rc:`savefig.format`.\n\n        The returned string does not include a period. This method is\n        overridden in backends that only support a single file type.\n        '
        return rcParams['savefig.format']

    def get_default_filename(self):
        if False:
            while True:
                i = 10
        '\n        Return a string, which includes extension, suitable for use as\n        a default filename.\n        '
        basename = self.manager.get_window_title() if self.manager is not None else ''
        basename = (basename or 'image').replace(' ', '_')
        filetype = self.get_default_filetype()
        filename = basename + '.' + filetype
        return filename

    @_api.deprecated('3.8')
    def switch_backends(self, FigureCanvasClass):
        if False:
            return 10
        '\n        Instantiate an instance of FigureCanvasClass\n\n        This is used for backend switching, e.g., to instantiate a\n        FigureCanvasPS from a FigureCanvasGTK.  Note, deep copying is\n        not done, so any changes to one of the instances (e.g., setting\n        figure size or line props), will be reflected in the other\n        '
        newCanvas = FigureCanvasClass(self.figure)
        newCanvas._is_saving = self._is_saving
        return newCanvas

    def mpl_connect(self, s, func):
        if False:
            for i in range(10):
                print('nop')
        "\n        Bind function *func* to event *s*.\n\n        Parameters\n        ----------\n        s : str\n            One of the following events ids:\n\n            - 'button_press_event'\n            - 'button_release_event'\n            - 'draw_event'\n            - 'key_press_event'\n            - 'key_release_event'\n            - 'motion_notify_event'\n            - 'pick_event'\n            - 'resize_event'\n            - 'scroll_event'\n            - 'figure_enter_event',\n            - 'figure_leave_event',\n            - 'axes_enter_event',\n            - 'axes_leave_event'\n            - 'close_event'.\n\n        func : callable\n            The callback function to be executed, which must have the\n            signature::\n\n                def func(event: Event) -> Any\n\n            For the location events (button and key press/release), if the\n            mouse is over the Axes, the ``inaxes`` attribute of the event will\n            be set to the `~matplotlib.axes.Axes` the event occurs is over, and\n            additionally, the variables ``xdata`` and ``ydata`` attributes will\n            be set to the mouse location in data coordinates.  See `.KeyEvent`\n            and `.MouseEvent` for more info.\n\n            .. note::\n\n                If func is a method, this only stores a weak reference to the\n                method. Thus, the figure does not influence the lifetime of\n                the associated object. Usually, you want to make sure that the\n                object is kept alive throughout the lifetime of the figure by\n                holding a reference to it.\n\n        Returns\n        -------\n        cid\n            A connection id that can be used with\n            `.FigureCanvasBase.mpl_disconnect`.\n\n        Examples\n        --------\n        ::\n\n            def on_press(event):\n                print('you pressed', event.button, event.xdata, event.ydata)\n\n            cid = canvas.mpl_connect('button_press_event', on_press)\n        "
        return self.callbacks.connect(s, func)

    def mpl_disconnect(self, cid):
        if False:
            while True:
                i = 10
        "\n        Disconnect the callback with id *cid*.\n\n        Examples\n        --------\n        ::\n\n            cid = canvas.mpl_connect('button_press_event', on_press)\n            # ... later\n            canvas.mpl_disconnect(cid)\n        "
        self.callbacks.disconnect(cid)
    _timer_cls = TimerBase

    def new_timer(self, interval=None, callbacks=None):
        if False:
            i = 10
            return i + 15
        "\n        Create a new backend-specific subclass of `.Timer`.\n\n        This is useful for getting periodic events through the backend's native\n        event loop.  Implemented only for backends with GUIs.\n\n        Parameters\n        ----------\n        interval : int\n            Timer interval in milliseconds.\n\n        callbacks : list[tuple[callable, tuple, dict]]\n            Sequence of (func, args, kwargs) where ``func(*args, **kwargs)``\n            will be executed by the timer every *interval*.\n\n            Callbacks which return ``False`` or ``0`` will be removed from the\n            timer.\n\n        Examples\n        --------\n        >>> timer = fig.canvas.new_timer(callbacks=[(f1, (1,), {'a': 3})])\n        "
        return self._timer_cls(interval=interval, callbacks=callbacks)

    def flush_events(self):
        if False:
            print('Hello World!')
        '\n        Flush the GUI events for the figure.\n\n        Interactive backends need to reimplement this method.\n        '

    def start_event_loop(self, timeout=0):
        if False:
            return 10
        '\n        Start a blocking event loop.\n\n        Such an event loop is used by interactive functions, such as\n        `~.Figure.ginput` and `~.Figure.waitforbuttonpress`, to wait for\n        events.\n\n        The event loop blocks until a callback function triggers\n        `stop_event_loop`, or *timeout* is reached.\n\n        If *timeout* is 0 or negative, never timeout.\n\n        Only interactive backends need to reimplement this method and it relies\n        on `flush_events` being properly implemented.\n\n        Interactive backends should implement this in a more native way.\n        '
        if timeout <= 0:
            timeout = np.inf
        timestep = 0.01
        counter = 0
        self._looping = True
        while self._looping and counter * timestep < timeout:
            self.flush_events()
            time.sleep(timestep)
            counter += 1

    def stop_event_loop(self):
        if False:
            return 10
        '\n        Stop the current blocking event loop.\n\n        Interactive backends need to reimplement this to match\n        `start_event_loop`\n        '
        self._looping = False

def key_press_handler(event, canvas=None, toolbar=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Implement the default Matplotlib key bindings for the canvas and toolbar\n    described at :ref:`key-event-handling`.\n\n    Parameters\n    ----------\n    event : `KeyEvent`\n        A key press/release event.\n    canvas : `FigureCanvasBase`, default: ``event.canvas``\n        The backend-specific canvas instance.  This parameter is kept for\n        back-compatibility, but, if set, should always be equal to\n        ``event.canvas``.\n    toolbar : `NavigationToolbar2`, default: ``event.canvas.toolbar``\n        The navigation cursor toolbar.  This parameter is kept for\n        back-compatibility, but, if set, should always be equal to\n        ``event.canvas.toolbar``.\n    '
    if event.key is None:
        return
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar
    if event.key in rcParams['keymap.fullscreen']:
        try:
            canvas.manager.full_screen_toggle()
        except AttributeError:
            pass
    if event.key in rcParams['keymap.quit']:
        Gcf.destroy_fig(canvas.figure)
    if event.key in rcParams['keymap.quit_all']:
        Gcf.destroy_all()
    if toolbar is not None:
        if event.key in rcParams['keymap.home']:
            toolbar.home()
        elif event.key in rcParams['keymap.back']:
            toolbar.back()
        elif event.key in rcParams['keymap.forward']:
            toolbar.forward()
        elif event.key in rcParams['keymap.pan']:
            toolbar.pan()
            toolbar._update_cursor(event)
        elif event.key in rcParams['keymap.zoom']:
            toolbar.zoom()
            toolbar._update_cursor(event)
        elif event.key in rcParams['keymap.save']:
            toolbar.save_figure()
    if event.inaxes is None:
        return

    def _get_uniform_gridstate(ticks):
        if False:
            print('Hello World!')
        return True if all((tick.gridline.get_visible() for tick in ticks)) else False if not any((tick.gridline.get_visible() for tick in ticks)) else None
    ax = event.inaxes
    if event.key in rcParams['keymap.grid'] and None not in [_get_uniform_gridstate(ax.xaxis.minorTicks), _get_uniform_gridstate(ax.yaxis.minorTicks)]:
        x_state = _get_uniform_gridstate(ax.xaxis.majorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.majorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            (x_state, y_state) = cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)]
        except ValueError:
            pass
        else:
            ax.grid(x_state, which='major' if x_state else 'both', axis='x')
            ax.grid(y_state, which='major' if y_state else 'both', axis='y')
            canvas.draw_idle()
    if event.key in rcParams['keymap.grid_minor'] and None not in [_get_uniform_gridstate(ax.xaxis.majorTicks), _get_uniform_gridstate(ax.yaxis.majorTicks)]:
        x_state = _get_uniform_gridstate(ax.xaxis.minorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.minorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            (x_state, y_state) = cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)]
        except ValueError:
            pass
        else:
            ax.grid(x_state, which='both', axis='x')
            ax.grid(y_state, which='both', axis='y')
            canvas.draw_idle()
    elif event.key in rcParams['keymap.yscale']:
        scale = ax.get_yscale()
        if scale == 'log':
            ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
        elif scale == 'linear':
            try:
                ax.set_yscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
    elif event.key in rcParams['keymap.xscale']:
        scalex = ax.get_xscale()
        if scalex == 'log':
            ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()
        elif scalex == 'linear':
            try:
                ax.set_xscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()

def button_press_handler(event, canvas=None, toolbar=None):
    if False:
        i = 10
        return i + 15
    '\n    The default Matplotlib button actions for extra mouse buttons.\n\n    Parameters are as for `key_press_handler`, except that *event* is a\n    `MouseEvent`.\n    '
    if canvas is None:
        canvas = event.canvas
    if toolbar is None:
        toolbar = canvas.toolbar
    if toolbar is not None:
        button_name = str(MouseButton(event.button))
        if button_name in rcParams['keymap.back']:
            toolbar.back()
        elif button_name in rcParams['keymap.forward']:
            toolbar.forward()

class NonGuiException(Exception):
    """Raised when trying show a figure in a non-GUI backend."""
    pass

class FigureManagerBase:
    """
    A backend-independent abstraction of a figure container and controller.

    The figure manager is used by pyplot to interact with the window in a
    backend-independent way. It's an adapter for the real (GUI) framework that
    represents the visual figure on screen.

    GUI backends define from this class to translate common operations such
    as *show* or *resize* to the GUI-specific code. Non-GUI backends do not
    support these operations an can just use the base class.

    This following basic operations are accessible:

    **Window operations**

    - `~.FigureManagerBase.show`
    - `~.FigureManagerBase.destroy`
    - `~.FigureManagerBase.full_screen_toggle`
    - `~.FigureManagerBase.resize`
    - `~.FigureManagerBase.get_window_title`
    - `~.FigureManagerBase.set_window_title`

    **Key and mouse button press handling**

    The figure manager sets up default key and mouse button press handling by
    hooking up the `.key_press_handler` to the matplotlib event system. This
    ensures the same shortcuts and mouse actions across backends.

    **Other operations**

    Subclasses will have additional attributes and functions to access
    additional functionality. This is of course backend-specific. For example,
    most GUI backends have ``window`` and ``toolbar`` attributes that give
    access to the native GUI widgets of the respective framework.

    Attributes
    ----------
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance.

    num : int or str
        The figure number.

    key_press_handler_id : int
        The default key handler cid, when using the toolmanager.
        To disable the default key press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.key_press_handler_id)

    button_press_handler_id : int
        The default mouse button handler cid, when using the toolmanager.
        To disable the default button press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.button_press_handler_id)
    """
    _toolbar2_class = None
    _toolmanager_toolbar_class = None

    def __init__(self, canvas, num):
        if False:
            print('Hello World!')
        self.canvas = canvas
        canvas.manager = self
        self.num = num
        self.set_window_title(f'Figure {num:d}')
        self.key_press_handler_id = None
        self.button_press_handler_id = None
        if rcParams['toolbar'] != 'toolmanager':
            self.key_press_handler_id = self.canvas.mpl_connect('key_press_event', key_press_handler)
            self.button_press_handler_id = self.canvas.mpl_connect('button_press_event', button_press_handler)
        self.toolmanager = ToolManager(canvas.figure) if mpl.rcParams['toolbar'] == 'toolmanager' else None
        if mpl.rcParams['toolbar'] == 'toolbar2' and self._toolbar2_class:
            self.toolbar = self._toolbar2_class(self.canvas)
        elif mpl.rcParams['toolbar'] == 'toolmanager' and self._toolmanager_toolbar_class:
            self.toolbar = self._toolmanager_toolbar_class(self.toolmanager)
        else:
            self.toolbar = None
        if self.toolmanager:
            tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                tools.add_tools_to_container(self.toolbar)

        @self.canvas.figure.add_axobserver
        def notify_axes_change(fig):
            if False:
                print('Hello World!')
            if self.toolmanager is None and self.toolbar is not None:
                self.toolbar.update()

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        if False:
            while True:
                i = 10
        '\n        Create a manager for a given *figure* using a specific *canvas_class*.\n\n        Backends should override this method if they have specific needs for\n        setting up the canvas or the manager.\n        '
        return cls(canvas_class(figure), num)

    @classmethod
    def start_main_loop(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start the main event loop.\n\n        This method is called by `.FigureManagerBase.pyplot_show`, which is the\n        implementation of `.pyplot.show`.  To customize the behavior of\n        `.pyplot.show`, interactive backends should usually override\n        `~.FigureManagerBase.start_main_loop`; if more customized logic is\n        necessary, `~.FigureManagerBase.pyplot_show` can also be overridden.\n        '

    @classmethod
    def pyplot_show(cls, *, block=None):
        if False:
            while True:
                i = 10
        "\n        Show all figures.  This method is the implementation of `.pyplot.show`.\n\n        To customize the behavior of `.pyplot.show`, interactive backends\n        should usually override `~.FigureManagerBase.start_main_loop`; if more\n        customized logic is necessary, `~.FigureManagerBase.pyplot_show` can\n        also be overridden.\n\n        Parameters\n        ----------\n        block : bool, optional\n            Whether to block by calling ``start_main_loop``.  The default,\n            None, means to block if we are neither in IPython's ``%pylab`` mode\n            nor in ``interactive`` mode.\n        "
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return
        for manager in managers:
            try:
                manager.show()
            except NonGuiException as exc:
                _api.warn_external(str(exc))
        if block is None:
            pyplot_show = getattr(sys.modules.get('matplotlib.pyplot'), 'show', None)
            ipython_pylab = hasattr(pyplot_show, '_needmain')
            block = not ipython_pylab and (not is_interactive())
        if block:
            cls.start_main_loop()

    def show(self):
        if False:
            return 10
        '\n        For GUI backends, show the figure window and redraw.\n        For non-GUI backends, raise an exception, unless running headless (i.e.\n        on Linux with an unset DISPLAY); this exception is converted to a\n        warning in `.Figure.show`.\n        '
        if sys.platform == 'linux' and (not os.environ.get('DISPLAY')):
            return
        raise NonGuiException(f'{type(self.canvas).__name__} is non-interactive, and thus cannot be shown')

    def destroy(self):
        if False:
            while True:
                i = 10
        pass

    def full_screen_toggle(self):
        if False:
            while True:
                i = 10
        pass

    def resize(self, w, h):
        if False:
            while True:
                i = 10
        'For GUI backends, resize the window (in physical pixels).'

    def get_window_title(self):
        if False:
            while True:
                i = 10
        '\n        Return the title text of the window containing the figure, or None\n        if there is no window (e.g., a PS backend).\n        '
        return 'image'

    def set_window_title(self, title):
        if False:
            while True:
                i = 10
        '\n        Set the title text of the window containing the figure.\n\n        This has no effect for non-GUI (e.g., PS) backends.\n        '
cursors = tools.cursors

class _Mode(str, Enum):
    NONE = ''
    PAN = 'pan/zoom'
    ZOOM = 'zoom rect'

    def __str__(self):
        if False:
            print('Hello World!')
        return self.value

    @property
    def _navigate_mode(self):
        if False:
            print('Hello World!')
        return self.name if self is not _Mode.NONE else None

class NavigationToolbar2:
    """
    Base class for the navigation cursor, version 2.

    Backends must implement a canvas that handles connections for
    'button_press_event' and 'button_release_event'.  See
    :meth:`FigureCanvasBase.mpl_connect` for more information.

    They must also define

    :meth:`save_figure`
        Save the current figure.

    :meth:`draw_rubberband` (optional)
        Draw the zoom to rect "rubberband" rectangle.

    :meth:`set_message` (optional)
        Display message.

    :meth:`set_history_buttons` (optional)
        You can change the history back / forward buttons to indicate disabled / enabled
        state.

    and override ``__init__`` to set up the toolbar -- without forgetting to
    call the base-class init.  Typically, ``__init__`` needs to set up toolbar
    buttons connected to the `home`, `back`, `forward`, `pan`, `zoom`, and
    `save_figure` methods and using standard icons in the "images" subdirectory
    of the data path.

    That's it, we'll do the rest!
    """
    toolitems = (('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous view', 'back', 'back'), ('Forward', 'Forward to next view', 'forward', 'forward'), (None, None, None, None), ('Pan', 'Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect', 'move', 'pan'), ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'), ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'), (None, None, None, None), ('Save', 'Save the figure', 'filesave', 'save_figure'))

    def __init__(self, canvas):
        if False:
            return 10
        self.canvas = canvas
        canvas.toolbar = self
        self._nav_stack = cbook._Stack()
        self._last_cursor = tools.Cursors.POINTER
        self._id_press = self.canvas.mpl_connect('button_press_event', self._zoom_pan_handler)
        self._id_release = self.canvas.mpl_connect('button_release_event', self._zoom_pan_handler)
        self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self._pan_info = None
        self._zoom_info = None
        self.mode = _Mode.NONE
        self.set_history_buttons()

    def set_message(self, s):
        if False:
            return 10
        'Display a message on toolbar or in status bar.'

    def draw_rubberband(self, event, x0, y0, x1, y1):
        if False:
            return 10
        '\n        Draw a rectangle rubberband to indicate zoom limits.\n\n        Note that it is not guaranteed that ``x0 <= x1`` and ``y0 <= y1``.\n        '

    def remove_rubberband(self):
        if False:
            print('Hello World!')
        'Remove the rubberband.'

    def home(self, *args):
        if False:
            while True:
                i = 10
        '\n        Restore the original view.\n\n        For convenience of being directly connected as a GUI callback, which\n        often get passed additional parameters, this method accepts arbitrary\n        parameters, but does not use them.\n        '
        self._nav_stack.home()
        self.set_history_buttons()
        self._update_view()

    def back(self, *args):
        if False:
            print('Hello World!')
        '\n        Move back up the view lim stack.\n\n        For convenience of being directly connected as a GUI callback, which\n        often get passed additional parameters, this method accepts arbitrary\n        parameters, but does not use them.\n        '
        self._nav_stack.back()
        self.set_history_buttons()
        self._update_view()

    def forward(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        Move forward in the view lim stack.\n\n        For convenience of being directly connected as a GUI callback, which\n        often get passed additional parameters, this method accepts arbitrary\n        parameters, but does not use them.\n        '
        self._nav_stack.forward()
        self.set_history_buttons()
        self._update_view()

    def _update_cursor(self, event):
        if False:
            print('Hello World!')
        '\n        Update the cursor after a mouse move event or a tool (de)activation.\n        '
        if self.mode and event.inaxes and event.inaxes.get_navigate():
            if self.mode == _Mode.ZOOM and self._last_cursor != tools.Cursors.SELECT_REGION:
                self.canvas.set_cursor(tools.Cursors.SELECT_REGION)
                self._last_cursor = tools.Cursors.SELECT_REGION
            elif self.mode == _Mode.PAN and self._last_cursor != tools.Cursors.MOVE:
                self.canvas.set_cursor(tools.Cursors.MOVE)
                self._last_cursor = tools.Cursors.MOVE
        elif self._last_cursor != tools.Cursors.POINTER:
            self.canvas.set_cursor(tools.Cursors.POINTER)
            self._last_cursor = tools.Cursors.POINTER

    @contextmanager
    def _wait_cursor_for_draw_cm(self):
        if False:
            while True:
                i = 10
        "\n        Set the cursor to a wait cursor when drawing the canvas.\n\n        In order to avoid constantly changing the cursor when the canvas\n        changes frequently, do nothing if this context was triggered during the\n        last second.  (Optimally we'd prefer only setting the wait cursor if\n        the *current* draw takes too long, but the current draw blocks the GUI\n        thread).\n        "
        (self._draw_time, last_draw_time) = (time.time(), getattr(self, '_draw_time', -np.inf))
        if self._draw_time - last_draw_time > 1:
            try:
                self.canvas.set_cursor(tools.Cursors.WAIT)
                yield
            finally:
                self.canvas.set_cursor(self._last_cursor)
        else:
            yield

    @staticmethod
    def _mouse_event_to_message(event):
        if False:
            return 10
        if event.inaxes and event.inaxes.get_navigate():
            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:
                s = s.rstrip()
                artists = [a for a in event.inaxes._mouseover_set if a.contains(event)[0] and a.get_visible()]
                if artists:
                    a = cbook._topmost_artist(artists)
                    if a is not event.inaxes.patch:
                        data = a.get_cursor_data(event)
                        if data is not None:
                            data_str = a.format_cursor_data(data).rstrip()
                            if data_str:
                                s = s + '\n' + data_str
                return s
        return ''

    def mouse_move(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._update_cursor(event)
        self.set_message(self._mouse_event_to_message(event))

    def _zoom_pan_handler(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self.mode == _Mode.PAN:
            if event.name == 'button_press_event':
                self.press_pan(event)
            elif event.name == 'button_release_event':
                self.release_pan(event)
        if self.mode == _Mode.ZOOM:
            if event.name == 'button_press_event':
                self.press_zoom(event)
            elif event.name == 'button_release_event':
                self.release_zoom(event)

    def pan(self, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Toggle the pan/zoom tool.\n\n        Pan with left button, zoom with right.\n        '
        if not self.canvas.widgetlock.available(self):
            self.set_message('pan unavailable')
            return
        if self.mode == _Mode.PAN:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.PAN
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)
    _PanInfo = namedtuple('_PanInfo', 'button axes cid')

    def press_pan(self, event):
        if False:
            while True:
                i = 10
        'Callback for mouse button press in pan/zoom mode.'
        if event.button not in [MouseButton.LEFT, MouseButton.RIGHT] or event.x is None or event.y is None:
            return
        axes = [a for a in self.canvas.figure.get_axes() if a.in_axes(event) and a.get_navigate() and a.can_pan()]
        if not axes:
            return
        if self._nav_stack() is None:
            self.push_current()
        for ax in axes:
            ax.start_pan(event.x, event.y, event.button)
        self.canvas.mpl_disconnect(self._id_drag)
        id_drag = self.canvas.mpl_connect('motion_notify_event', self.drag_pan)
        self._pan_info = self._PanInfo(button=event.button, axes=axes, cid=id_drag)

    def drag_pan(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Callback for dragging in pan/zoom mode.'
        for ax in self._pan_info.axes:
            ax.drag_pan(self._pan_info.button, event.key, event.x, event.y)
        self.canvas.draw_idle()

    def release_pan(self, event):
        if False:
            while True:
                i = 10
        'Callback for mouse button release in pan/zoom mode.'
        if self._pan_info is None:
            return
        self.canvas.mpl_disconnect(self._pan_info.cid)
        self._id_drag = self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        for ax in self._pan_info.axes:
            ax.end_pan()
        self.canvas.draw_idle()
        self._pan_info = None
        self.push_current()

    def zoom(self, *args):
        if False:
            for i in range(10):
                print('nop')
        if not self.canvas.widgetlock.available(self):
            self.set_message('zoom unavailable')
            return
        'Toggle zoom to rect mode.'
        if self.mode == _Mode.ZOOM:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.ZOOM
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)
    _ZoomInfo = namedtuple('_ZoomInfo', 'direction start_xy axes cid cbar')

    def press_zoom(self, event):
        if False:
            return 10
        'Callback for mouse button press in zoom to rect mode.'
        if event.button not in [MouseButton.LEFT, MouseButton.RIGHT] or event.x is None or event.y is None:
            return
        axes = [a for a in self.canvas.figure.get_axes() if a.in_axes(event) and a.get_navigate() and a.can_zoom()]
        if not axes:
            return
        if self._nav_stack() is None:
            self.push_current()
        id_zoom = self.canvas.mpl_connect('motion_notify_event', self.drag_zoom)
        if hasattr(axes[0], '_colorbar'):
            cbar = axes[0]._colorbar.orientation
        else:
            cbar = None
        self._zoom_info = self._ZoomInfo(direction='in' if event.button == 1 else 'out', start_xy=(event.x, event.y), axes=axes, cid=id_zoom, cbar=cbar)

    def drag_zoom(self, event):
        if False:
            print('Hello World!')
        'Callback for dragging in zoom mode.'
        start_xy = self._zoom_info.start_xy
        ax = self._zoom_info.axes[0]
        ((x1, y1), (x2, y2)) = np.clip([start_xy, [event.x, event.y]], ax.bbox.min, ax.bbox.max)
        key = event.key
        if self._zoom_info.cbar == 'horizontal':
            key = 'x'
        elif self._zoom_info.cbar == 'vertical':
            key = 'y'
        if key == 'x':
            (y1, y2) = ax.bbox.intervaly
        elif key == 'y':
            (x1, x2) = ax.bbox.intervalx
        self.draw_rubberband(event, x1, y1, x2, y2)

    def release_zoom(self, event):
        if False:
            print('Hello World!')
        'Callback for mouse button release in zoom to rect mode.'
        if self._zoom_info is None:
            return
        self.canvas.mpl_disconnect(self._zoom_info.cid)
        self.remove_rubberband()
        (start_x, start_y) = self._zoom_info.start_xy
        key = event.key
        if self._zoom_info.cbar == 'horizontal':
            key = 'x'
        elif self._zoom_info.cbar == 'vertical':
            key = 'y'
        if abs(event.x - start_x) < 5 and key != 'y' or (abs(event.y - start_y) < 5 and key != 'x'):
            self.canvas.draw_idle()
            self._zoom_info = None
            return
        for (i, ax) in enumerate(self._zoom_info.axes):
            twinx = any((ax.get_shared_x_axes().joined(ax, prev) for prev in self._zoom_info.axes[:i]))
            twiny = any((ax.get_shared_y_axes().joined(ax, prev) for prev in self._zoom_info.axes[:i]))
            ax._set_view_from_bbox((start_x, start_y, event.x, event.y), self._zoom_info.direction, key, twinx, twiny)
        self.canvas.draw_idle()
        self._zoom_info = None
        self.push_current()

    def push_current(self):
        if False:
            i = 10
            return i + 15
        'Push the current view limits and position onto the stack.'
        self._nav_stack.push(WeakKeyDictionary({ax: (ax._get_view(), (ax.get_position(True).frozen(), ax.get_position().frozen())) for ax in self.canvas.figure.axes}))
        self.set_history_buttons()

    def _update_view(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the viewlim and position from the view and position stack for\n        each Axes.\n        '
        nav_info = self._nav_stack()
        if nav_info is None:
            return
        items = list(nav_info.items())
        for (ax, (view, (pos_orig, pos_active))) in items:
            ax._set_view(view)
            ax._set_position(pos_orig, 'original')
            ax._set_position(pos_active, 'active')
        self.canvas.draw_idle()

    def configure_subplots(self, *args):
        if False:
            i = 10
            return i + 15
        if hasattr(self, 'subplot_tool'):
            self.subplot_tool.figure.canvas.manager.show()
            return
        from matplotlib.figure import Figure
        with mpl.rc_context({'toolbar': 'none'}):
            manager = type(self.canvas).new_manager(Figure(figsize=(6, 3)), -1)
        manager.set_window_title('Subplot configuration tool')
        tool_fig = manager.canvas.figure
        tool_fig.subplots_adjust(top=0.9)
        self.subplot_tool = widgets.SubplotTool(self.canvas.figure, tool_fig)
        cid = self.canvas.mpl_connect('close_event', lambda e: manager.destroy())

        def on_tool_fig_close(e):
            if False:
                return 10
            self.canvas.mpl_disconnect(cid)
            del self.subplot_tool
        tool_fig.canvas.mpl_connect('close_event', on_tool_fig_close)
        manager.show()
        return self.subplot_tool

    def save_figure(self, *args):
        if False:
            while True:
                i = 10
        'Save the current figure.'
        raise NotImplementedError

    def update(self):
        if False:
            i = 10
            return i + 15
        'Reset the Axes stack.'
        self._nav_stack.clear()
        self.set_history_buttons()

    def set_history_buttons(self):
        if False:
            while True:
                i = 10
        'Enable or disable the back/forward button.'

class ToolContainerBase:
    """
    Base class for all tool containers, e.g. toolbars.

    Attributes
    ----------
    toolmanager : `.ToolManager`
        The tools with which this `ToolContainer` wants to communicate.
    """
    _icon_extension = '.png'
    '\n    Toolcontainer button icon image format extension\n\n    **String**: Image extension\n    '

    def __init__(self, toolmanager):
        if False:
            while True:
                i = 10
        self.toolmanager = toolmanager
        toolmanager.toolmanager_connect('tool_message_event', lambda event: self.set_message(event.message))
        toolmanager.toolmanager_connect('tool_removed_event', lambda event: self.remove_toolitem(event.tool.name))

    def _tool_toggled_cbk(self, event):
        if False:
            i = 10
            return i + 15
        "\n        Capture the 'tool_trigger_[name]'\n\n        This only gets used for toggled tools.\n        "
        self.toggle_toolitem(event.tool.name, event.tool.toggled)

    def add_tool(self, tool, group, position=-1):
        if False:
            while True:
                i = 10
        '\n        Add a tool to this container.\n\n        Parameters\n        ----------\n        tool : tool_like\n            The tool to add, see `.ToolManager.get_tool`.\n        group : str\n            The name of the group to add this tool to.\n        position : int, default: -1\n            The position within the group to place this tool.\n        '
        tool = self.toolmanager.get_tool(tool)
        image = self._get_image_filename(tool.image)
        toggle = getattr(tool, 'toggled', None) is not None
        self.add_toolitem(tool.name, group, position, image, tool.description, toggle)
        if toggle:
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name, self._tool_toggled_cbk)
            if tool.toggled:
                self.toggle_toolitem(tool.name, True)

    def _get_image_filename(self, image):
        if False:
            i = 10
            return i + 15
        'Find the image based on its name.'
        if not image:
            return None
        basedir = cbook._get_data_path('images')
        for fname in [image, image + self._icon_extension, str(basedir / image), str(basedir / (image + self._icon_extension))]:
            if os.path.isfile(fname):
                return fname

    def trigger_tool(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Trigger the tool.\n\n        Parameters\n        ----------\n        name : str\n            Name (id) of the tool triggered from within the container.\n        '
        self.toolmanager.trigger_tool(name, sender=self)

    def add_toolitem(self, name, group, position, image, description, toggle):
        if False:
            for i in range(10):
                print('nop')
        "\n        Add a toolitem to the container.\n\n        This method must be implemented per backend.\n\n        The callback associated with the button click event,\n        must be *exactly* ``self.trigger_tool(name)``.\n\n        Parameters\n        ----------\n        name : str\n            Name of the tool to add, this gets used as the tool's ID and as the\n            default label of the buttons.\n        group : str\n            Name of the group that this tool belongs to.\n        position : int\n            Position of the tool within its group, if -1 it goes at the end.\n        image : str\n            Filename of the image for the button or `None`.\n        description : str\n            Description of the tool, used for the tooltips.\n        toggle : bool\n            * `True` : The button is a toggle (change the pressed/unpressed\n              state between consecutive clicks).\n            * `False` : The button is a normal button (returns to unpressed\n              state after release).\n        "
        raise NotImplementedError

    def toggle_toolitem(self, name, toggled):
        if False:
            print('Hello World!')
        '\n        Toggle the toolitem without firing event.\n\n        Parameters\n        ----------\n        name : str\n            Id of the tool to toggle.\n        toggled : bool\n            Whether to set this tool as toggled or not.\n        '
        raise NotImplementedError

    def remove_toolitem(self, name):
        if False:
            return 10
        '\n        Remove a toolitem from the `ToolContainer`.\n\n        This method must get implemented per backend.\n\n        Called when `.ToolManager` emits a `tool_removed_event`.\n\n        Parameters\n        ----------\n        name : str\n            Name of the tool to remove.\n        '
        raise NotImplementedError

    def set_message(self, s):
        if False:
            for i in range(10):
                print('nop')
        '\n        Display a message on the toolbar.\n\n        Parameters\n        ----------\n        s : str\n            Message text.\n        '
        raise NotImplementedError

class _Backend:
    backend_version = 'unknown'
    FigureCanvas = None
    FigureManager = FigureManagerBase
    mainloop = None

    @classmethod
    def new_figure_manager(cls, num, *args, **kwargs):
        if False:
            print('Hello World!')
        'Create a new figure manager instance.'
        from matplotlib.figure import Figure
        fig_cls = kwargs.pop('FigureClass', Figure)
        fig = fig_cls(*args, **kwargs)
        return cls.new_figure_manager_given_figure(num, fig)

    @classmethod
    def new_figure_manager_given_figure(cls, num, figure):
        if False:
            i = 10
            return i + 15
        'Create a new figure manager instance for the given figure.'
        return cls.FigureCanvas.new_manager(figure, num)

    @classmethod
    def draw_if_interactive(cls):
        if False:
            return 10
        manager_class = cls.FigureCanvas.manager_class
        backend_is_interactive = manager_class.start_main_loop != FigureManagerBase.start_main_loop or manager_class.pyplot_show != FigureManagerBase.pyplot_show
        if backend_is_interactive and is_interactive():
            manager = Gcf.get_active()
            if manager:
                manager.canvas.draw_idle()

    @classmethod
    def show(cls, *, block=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Show all figures.\n\n        `show` blocks by calling `mainloop` if *block* is ``True``, or if it is\n        ``None`` and we are not in `interactive` mode and if IPython's\n        ``%matplotlib`` integration has not been activated.\n        "
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return
        for manager in managers:
            try:
                manager.show()
            except NonGuiException as exc:
                _api.warn_external(str(exc))
        if cls.mainloop is None:
            return
        if block is None:
            pyplot_show = getattr(sys.modules.get('matplotlib.pyplot'), 'show', None)
            ipython_pylab = hasattr(pyplot_show, '_needmain')
            block = not ipython_pylab and (not is_interactive())
        if block:
            cls.mainloop()

    @staticmethod
    def export(cls):
        if False:
            while True:
                i = 10
        for name in ['backend_version', 'FigureCanvas', 'FigureManager', 'new_figure_manager', 'new_figure_manager_given_figure', 'draw_if_interactive', 'show']:
            setattr(sys.modules[cls.__module__], name, getattr(cls, name))

        class Show(ShowBase):

            def mainloop(self):
                if False:
                    i = 10
                    return i + 15
                return cls.mainloop()
        setattr(sys.modules[cls.__module__], 'Show', Show)
        return cls

class ShowBase(_Backend):
    """
    Simple base class to generate a ``show()`` function in backends.

    Subclass must override ``mainloop()`` method.
    """

    def __call__(self, block=None):
        if False:
            return 10
        return self.show(block=block)