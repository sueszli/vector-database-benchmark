"""
Module containing 3D artist code and functions to convert 2D
artists into 3D versions which can be added to an Axes3D.
"""
import math
import numpy as np
from contextlib import contextmanager
from matplotlib import artist, cbook, colors as mcolors, lines, text as mtext, path as mpath
from matplotlib.collections import Collection, LineCollection, PolyCollection, PatchCollection, PathCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d

def _norm_angle(a):
    if False:
        for i in range(10):
            print('nop')
    'Return the given angle normalized to -180 < *a* <= 180 degrees.'
    a = (a + 360) % 360
    if a > 180:
        a = a - 360
    return a

def _norm_text_angle(a):
    if False:
        print('Hello World!')
    'Return the given angle normalized to -90 < *a* <= 90 degrees.'
    a = (a + 180) % 180
    if a > 90:
        a = a - 180
    return a

def get_dir_vector(zdir):
    if False:
        i = 10
        return i + 15
    "\n    Return a direction vector.\n\n    Parameters\n    ----------\n    zdir : {'x', 'y', 'z', None, 3-tuple}\n        The direction. Possible values are:\n\n        - 'x': equivalent to (1, 0, 0)\n        - 'y': equivalent to (0, 1, 0)\n        - 'z': equivalent to (0, 0, 1)\n        - *None*: equivalent to (0, 0, 0)\n        - an iterable (x, y, z) is converted to an array\n\n    Returns\n    -------\n    x, y, z : array\n        The direction vector.\n    "
    if zdir == 'x':
        return np.array((1, 0, 0))
    elif zdir == 'y':
        return np.array((0, 1, 0))
    elif zdir == 'z':
        return np.array((0, 0, 1))
    elif zdir is None:
        return np.array((0, 0, 0))
    elif np.iterable(zdir) and len(zdir) == 3:
        return np.array(zdir)
    else:
        raise ValueError("'x', 'y', 'z', None or vector of length 3 expected")

class Text3D(mtext.Text):
    """
    Text object with 3D position and direction.

    Parameters
    ----------
    x, y, z : float
        The position of the text.
    text : str
        The text string to display.
    zdir : {'x', 'y', 'z', None, 3-tuple}
        The direction of the text. See `.get_dir_vector` for a description of
        the values.

    Other Parameters
    ----------------
    **kwargs
         All other parameters are passed on to `~matplotlib.text.Text`.
    """

    def __init__(self, x=0, y=0, z=0, text='', zdir='z', **kwargs):
        if False:
            return 10
        mtext.Text.__init__(self, x, y, text, **kwargs)
        self.set_3d_properties(z, zdir)

    def get_position_3d(self):
        if False:
            while True:
                i = 10
        'Return the (x, y, z) position of the text.'
        return (self._x, self._y, self._z)

    def set_position_3d(self, xyz, zdir=None):
        if False:
            print('Hello World!')
        "\n        Set the (*x*, *y*, *z*) position of the text.\n\n        Parameters\n        ----------\n        xyz : (float, float, float)\n            The position in 3D space.\n        zdir : {'x', 'y', 'z', None, 3-tuple}\n            The direction of the text. If unspecified, the *zdir* will not be\n            changed. See `.get_dir_vector` for a description of the values.\n        "
        super().set_position(xyz[:2])
        self.set_z(xyz[2])
        if zdir is not None:
            self._dir_vec = get_dir_vector(zdir)

    def set_z(self, z):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the *z* position of the text.\n\n        Parameters\n        ----------\n        z : float\n        '
        self._z = z
        self.stale = True

    def set_3d_properties(self, z=0, zdir='z'):
        if False:
            while True:
                i = 10
        "\n        Set the *z* position and direction of the text.\n\n        Parameters\n        ----------\n        z : float\n            The z-position in 3D space.\n        zdir : {'x', 'y', 'z', 3-tuple}\n            The direction of the text. Default: 'z'.\n            See `.get_dir_vector` for a description of the values.\n        "
        self._z = z
        self._dir_vec = get_dir_vector(zdir)
        self.stale = True

    @artist.allow_rasterization
    def draw(self, renderer):
        if False:
            i = 10
            return i + 15
        position3d = np.array((self._x, self._y, self._z))
        proj = proj3d._proj_trans_points([position3d, position3d + self._dir_vec], self.axes.M)
        dx = proj[0][1] - proj[0][0]
        dy = proj[1][1] - proj[1][0]
        angle = math.degrees(math.atan2(dy, dx))
        with cbook._setattr_cm(self, _x=proj[0][0], _y=proj[1][0], _rotation=_norm_text_angle(angle)):
            mtext.Text.draw(self, renderer)
        self.stale = False

    def get_tightbbox(self, renderer=None):
        if False:
            for i in range(10):
                print('nop')
        return None

def text_2d_to_3d(obj, z=0, zdir='z'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert a `.Text` to a `.Text3D` object.\n\n    Parameters\n    ----------\n    z : float\n        The z-position in 3D space.\n    zdir : {'x', 'y', 'z', 3-tuple}\n        The direction of the text. Default: 'z'.\n        See `.get_dir_vector` for a description of the values.\n    "
    obj.__class__ = Text3D
    obj.set_3d_properties(z, zdir)

class Line3D(lines.Line2D):
    """
    3D line object.

    .. note:: Use `get_data_3d` to obtain the data associated with the line.
            `~.Line2D.get_data`, `~.Line2D.get_xdata`, and `~.Line2D.get_ydata` return
            the x- and y-coordinates of the projected 2D-line, not the x- and y-data of
            the 3D-line. Similarly, use `set_data_3d` to set the data, not
            `~.Line2D.set_data`, `~.Line2D.set_xdata`, and `~.Line2D.set_ydata`.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n\n        Parameters\n        ----------\n        xs : array-like\n            The x-data to be plotted.\n        ys : array-like\n            The y-data to be plotted.\n        zs : array-like\n            The z-data to be plotted.\n        *args, **kwargs\n            Additional arguments are passed to `~matplotlib.lines.Line2D`.\n        '
        super().__init__([], [], *args, **kwargs)
        self.set_data_3d(xs, ys, zs)

    def set_3d_properties(self, zs=0, zdir='z'):
        if False:
            return 10
        "\n        Set the *z* position and direction of the line.\n\n        Parameters\n        ----------\n        zs : float or array of floats\n            The location along the *zdir* axis in 3D space to position the\n            line.\n        zdir : {'x', 'y', 'z'}\n            Plane to plot line orthogonal to. Default: 'z'.\n            See `.get_dir_vector` for a description of the values.\n        "
        xs = self.get_xdata()
        ys = self.get_ydata()
        zs = cbook._to_unmasked_float_array(zs).ravel()
        zs = np.broadcast_to(zs, len(xs))
        self._verts3d = juggle_axes(xs, ys, zs, zdir)
        self.stale = True

    def set_data_3d(self, *args):
        if False:
            print('Hello World!')
        '\n        Set the x, y and z data\n\n        Parameters\n        ----------\n        x : array-like\n            The x-data to be plotted.\n        y : array-like\n            The y-data to be plotted.\n        z : array-like\n            The z-data to be plotted.\n\n        Notes\n        -----\n        Accepts x, y, z arguments or a single array-like (x, y, z)\n        '
        if len(args) == 1:
            args = args[0]
        for (name, xyz) in zip('xyz', args):
            if not np.iterable(xyz):
                raise RuntimeError(f'{name} must be a sequence')
        self._verts3d = args
        self.stale = True

    def get_data_3d(self):
        if False:
            print('Hello World!')
        '\n        Get the current data\n\n        Returns\n        -------\n        verts3d : length-3 tuple or array-like\n            The current data as a tuple or array-like.\n        '
        return self._verts3d

    @artist.allow_rasterization
    def draw(self, renderer):
        if False:
            i = 10
            return i + 15
        (xs3d, ys3d, zs3d) = self._verts3d
        (xs, ys, zs) = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_data(xs, ys)
        super().draw(renderer)
        self.stale = False

def line_2d_to_3d(line, zs=0, zdir='z'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert a `.Line2D` to a `.Line3D` object.\n\n    Parameters\n    ----------\n    zs : float\n        The location along the *zdir* axis in 3D space to position the line.\n    zdir : {'x', 'y', 'z'}\n        Plane to plot line orthogonal to. Default: 'z'.\n        See `.get_dir_vector` for a description of the values.\n    "
    line.__class__ = Line3D
    line.set_3d_properties(zs, zdir)

def _path_to_3d_segment(path, zs=0, zdir='z'):
    if False:
        i = 10
        return i + 15
    'Convert a path to a 3D segment.'
    zs = np.broadcast_to(zs, len(path))
    pathsegs = path.iter_segments(simplify=False, curves=False)
    seg = [(x, y, z) for (((x, y), code), z) in zip(pathsegs, zs)]
    seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    return seg3d

def _paths_to_3d_segments(paths, zs=0, zdir='z'):
    if False:
        return 10
    'Convert paths from a collection object to 3D segments.'
    if not np.iterable(zs):
        zs = np.broadcast_to(zs, len(paths))
    elif len(zs) != len(paths):
        raise ValueError('Number of z-coordinates does not match paths.')
    segs = [_path_to_3d_segment(path, pathz, zdir) for (path, pathz) in zip(paths, zs)]
    return segs

def _path_to_3d_segment_with_codes(path, zs=0, zdir='z'):
    if False:
        while True:
            i = 10
    'Convert a path to a 3D segment with path codes.'
    zs = np.broadcast_to(zs, len(path))
    pathsegs = path.iter_segments(simplify=False, curves=False)
    seg_codes = [((x, y, z), code) for (((x, y), code), z) in zip(pathsegs, zs)]
    if seg_codes:
        (seg, codes) = zip(*seg_codes)
        seg3d = [juggle_axes(x, y, z, zdir) for (x, y, z) in seg]
    else:
        seg3d = []
        codes = []
    return (seg3d, list(codes))

def _paths_to_3d_segments_with_codes(paths, zs=0, zdir='z'):
    if False:
        i = 10
        return i + 15
    '\n    Convert paths from a collection object to 3D segments with path codes.\n    '
    zs = np.broadcast_to(zs, len(paths))
    segments_codes = [_path_to_3d_segment_with_codes(path, pathz, zdir) for (path, pathz) in zip(paths, zs)]
    if segments_codes:
        (segments, codes) = zip(*segments_codes)
    else:
        (segments, codes) = ([], [])
    return (list(segments), list(codes))

class Collection3D(Collection):
    """A collection of 3D paths."""

    def do_3d_projection(self):
        if False:
            while True:
                i = 10
        'Project the points according to renderer matrix.'
        xyzs_list = [proj3d.proj_transform(*vs.T, self.axes.M) for (vs, _) in self._3dverts_codes]
        self._paths = [mpath.Path(np.column_stack([xs, ys]), cs) for ((xs, ys, _), (_, cs)) in zip(xyzs_list, self._3dverts_codes)]
        zs = np.concatenate([zs for (_, _, zs) in xyzs_list])
        return zs.min() if len(zs) else 1000000000.0

def collection_2d_to_3d(col, zs=0, zdir='z'):
    if False:
        print('Hello World!')
    'Convert a `.Collection` to a `.Collection3D` object.'
    zs = np.broadcast_to(zs, len(col.get_paths()))
    col._3dverts_codes = [(np.column_stack(juggle_axes(*np.column_stack([p.vertices, np.broadcast_to(z, len(p.vertices))]).T, zdir)), p.codes) for (p, z) in zip(col.get_paths(), zs)]
    col.__class__ = cbook._make_class_factory(Collection3D, '{}3D')(type(col))

class Line3DCollection(LineCollection):
    """
    A collection of 3D lines.
    """

    def set_sort_zpos(self, val):
        if False:
            print('Hello World!')
        'Set the position to use for z-sorting.'
        self._sort_zpos = val
        self.stale = True

    def set_segments(self, segments):
        if False:
            return 10
        '\n        Set 3D segments.\n        '
        self._segments3d = segments
        super().set_segments([])

    def do_3d_projection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Project the points according to renderer matrix.\n        '
        xyslist = [proj3d._proj_trans_points(points, self.axes.M) for points in self._segments3d]
        segments_2d = [np.column_stack([xs, ys]) for (xs, ys, zs) in xyslist]
        LineCollection.set_segments(self, segments_2d)
        minz = 1000000000.0
        for (xs, ys, zs) in xyslist:
            minz = min(minz, min(zs))
        return minz

def line_collection_2d_to_3d(col, zs=0, zdir='z'):
    if False:
        print('Hello World!')
    'Convert a `.LineCollection` to a `.Line3DCollection` object.'
    segments3d = _paths_to_3d_segments(col.get_paths(), zs, zdir)
    col.__class__ = Line3DCollection
    col.set_segments(segments3d)

class Patch3D(Patch):
    """
    3D patch object.
    """

    def __init__(self, *args, zs=(), zdir='z', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parameters\n        ----------\n        verts :\n        zs : float\n            The location along the *zdir* axis in 3D space to position the\n            patch.\n        zdir : {'x', 'y', 'z'}\n            Plane to plot patch orthogonal to. Default: 'z'.\n            See `.get_dir_vector` for a description of the values.\n        "
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)

    def set_3d_properties(self, verts, zs=0, zdir='z'):
        if False:
            i = 10
            return i + 15
        "\n        Set the *z* position and direction of the patch.\n\n        Parameters\n        ----------\n        verts :\n        zs : float\n            The location along the *zdir* axis in 3D space to position the\n            patch.\n        zdir : {'x', 'y', 'z'}\n            Plane to plot patch orthogonal to. Default: 'z'.\n            See `.get_dir_vector` for a description of the values.\n        "
        zs = np.broadcast_to(zs, len(verts))
        self._segment3d = [juggle_axes(x, y, z, zdir) for ((x, y), z) in zip(verts, zs)]

    def get_path(self):
        if False:
            i = 10
            return i + 15
        return self._path2d

    def do_3d_projection(self):
        if False:
            while True:
                i = 10
        s = self._segment3d
        (xs, ys, zs) = zip(*s)
        (vxs, vys, vzs, vis) = proj3d.proj_transform_clip(xs, ys, zs, self.axes.M)
        self._path2d = mpath.Path(np.column_stack([vxs, vys]))
        return min(vzs)

class PathPatch3D(Patch3D):
    """
    3D PathPatch object.
    """

    def __init__(self, path, *, zs=(), zdir='z', **kwargs):
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        path :\n        zs : float\n            The location along the *zdir* axis in 3D space to position the\n            path patch.\n        zdir : {'x', 'y', 'z', 3-tuple}\n            Plane to plot path patch orthogonal to. Default: 'z'.\n            See `.get_dir_vector` for a description of the values.\n        "
        Patch.__init__(self, **kwargs)
        self.set_3d_properties(path, zs, zdir)

    def set_3d_properties(self, path, zs=0, zdir='z'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the *z* position and direction of the path patch.\n\n        Parameters\n        ----------\n        path :\n        zs : float\n            The location along the *zdir* axis in 3D space to position the\n            path patch.\n        zdir : {'x', 'y', 'z', 3-tuple}\n            Plane to plot path patch orthogonal to. Default: 'z'.\n            See `.get_dir_vector` for a description of the values.\n        "
        Patch3D.set_3d_properties(self, path.vertices, zs=zs, zdir=zdir)
        self._code3d = path.codes

    def do_3d_projection(self):
        if False:
            i = 10
            return i + 15
        s = self._segment3d
        (xs, ys, zs) = zip(*s)
        (vxs, vys, vzs, vis) = proj3d.proj_transform_clip(xs, ys, zs, self.axes.M)
        self._path2d = mpath.Path(np.column_stack([vxs, vys]), self._code3d)
        return min(vzs)

def _get_patch_verts(patch):
    if False:
        i = 10
        return i + 15
    'Return a list of vertices for the path of a patch.'
    trans = patch.get_patch_transform()
    path = patch.get_path()
    polygons = path.to_polygons(trans)
    return polygons[0] if len(polygons) else np.array([])

def patch_2d_to_3d(patch, z=0, zdir='z'):
    if False:
        for i in range(10):
            print('nop')
    'Convert a `.Patch` to a `.Patch3D` object.'
    verts = _get_patch_verts(patch)
    patch.__class__ = Patch3D
    patch.set_3d_properties(verts, z, zdir)

def pathpatch_2d_to_3d(pathpatch, z=0, zdir='z'):
    if False:
        for i in range(10):
            print('nop')
    'Convert a `.PathPatch` to a `.PathPatch3D` object.'
    path = pathpatch.get_path()
    trans = pathpatch.get_patch_transform()
    mpath = trans.transform_path(path)
    pathpatch.__class__ = PathPatch3D
    pathpatch.set_3d_properties(mpath, z, zdir)

class Patch3DCollection(PatchCollection):
    """
    A collection of 3D patches.
    """

    def __init__(self, *args, zs=0, zdir='z', depthshade=True, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Create a collection of flat 3D patches with its normal vector\n        pointed in *zdir* direction, and located at *zs* on the *zdir*\n        axis. 'zs' can be a scalar or an array-like of the same length as\n        the number of patches in the collection.\n\n        Constructor arguments are the same as for\n        :class:`~matplotlib.collections.PatchCollection`. In addition,\n        keywords *zs=0* and *zdir='z'* are available.\n\n        Also, the keyword argument *depthshade* is available to indicate\n        whether to shade the patches in order to give the appearance of depth\n        (default is *True*). This is typically desired in scatter plots.\n        "
        self._depthshade = depthshade
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)

    def get_depthshade(self):
        if False:
            for i in range(10):
                print('nop')
        return self._depthshade

    def set_depthshade(self, depthshade):
        if False:
            i = 10
            return i + 15
        '\n        Set whether depth shading is performed on collection members.\n\n        Parameters\n        ----------\n        depthshade : bool\n            Whether to shade the patches in order to give the appearance of\n            depth.\n        '
        self._depthshade = depthshade
        self.stale = True

    def set_sort_zpos(self, val):
        if False:
            for i in range(10):
                print('nop')
        'Set the position to use for z-sorting.'
        self._sort_zpos = val
        self.stale = True

    def set_3d_properties(self, zs, zdir):
        if False:
            i = 10
            return i + 15
        "\n        Set the *z* positions and direction of the patches.\n\n        Parameters\n        ----------\n        zs : float or array of floats\n            The location or locations to place the patches in the collection\n            along the *zdir* axis.\n        zdir : {'x', 'y', 'z'}\n            Plane to plot patches orthogonal to.\n            All patches must have the same direction.\n            See `.get_dir_vector` for a description of the values.\n        "
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            (xs, ys) = offsets.T
        else:
            xs = []
            ys = []
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        self._z_markers_idx = slice(-1)
        self._vzs = None
        self.stale = True

    def do_3d_projection(self):
        if False:
            print('Hello World!')
        (xs, ys, zs) = self._offsets3d
        (vxs, vys, vzs, vis) = proj3d.proj_transform_clip(xs, ys, zs, self.axes.M)
        self._vzs = vzs
        super().set_offsets(np.column_stack([vxs, vys]))
        if vzs.size > 0:
            return min(vzs)
        else:
            return np.nan

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        if False:
            return 10
        color_array = _zalpha(color_array, self._vzs) if self._vzs is not None and self._depthshade else color_array
        if len(color_array) > 1:
            color_array = color_array[self._z_markers_idx]
        return mcolors.to_rgba_array(color_array, self._alpha)

    def get_facecolor(self):
        if False:
            for i in range(10):
                print('nop')
        return self._maybe_depth_shade_and_sort_colors(super().get_facecolor())

    def get_edgecolor(self):
        if False:
            print('Hello World!')
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        return self._maybe_depth_shade_and_sort_colors(super().get_edgecolor())

class Path3DCollection(PathCollection):
    """
    A collection of 3D paths.
    """

    def __init__(self, *args, zs=0, zdir='z', depthshade=True, **kwargs):
        if False:
            print('Hello World!')
        "\n        Create a collection of flat 3D paths with its normal vector\n        pointed in *zdir* direction, and located at *zs* on the *zdir*\n        axis. 'zs' can be a scalar or an array-like of the same length as\n        the number of paths in the collection.\n\n        Constructor arguments are the same as for\n        :class:`~matplotlib.collections.PathCollection`. In addition,\n        keywords *zs=0* and *zdir='z'* are available.\n\n        Also, the keyword argument *depthshade* is available to indicate\n        whether to shade the patches in order to give the appearance of depth\n        (default is *True*). This is typically desired in scatter plots.\n        "
        self._depthshade = depthshade
        self._in_draw = False
        super().__init__(*args, **kwargs)
        self.set_3d_properties(zs, zdir)
        self._offset_zordered = None

    def draw(self, renderer):
        if False:
            print('Hello World!')
        with self._use_zordered_offset():
            with cbook._setattr_cm(self, _in_draw=True):
                super().draw(renderer)

    def set_sort_zpos(self, val):
        if False:
            return 10
        'Set the position to use for z-sorting.'
        self._sort_zpos = val
        self.stale = True

    def set_3d_properties(self, zs, zdir):
        if False:
            print('Hello World!')
        "\n        Set the *z* positions and direction of the paths.\n\n        Parameters\n        ----------\n        zs : float or array of floats\n            The location or locations to place the paths in the collection\n            along the *zdir* axis.\n        zdir : {'x', 'y', 'z'}\n            Plane to plot paths orthogonal to.\n            All paths must have the same direction.\n            See `.get_dir_vector` for a description of the values.\n        "
        self.update_scalarmappable()
        offsets = self.get_offsets()
        if len(offsets) > 0:
            (xs, ys) = offsets.T
        else:
            xs = []
            ys = []
        self._offsets3d = juggle_axes(xs, ys, np.atleast_1d(zs), zdir)
        self._sizes3d = self._sizes
        self._linewidths3d = np.array(self._linewidths)
        (xs, ys, zs) = self._offsets3d
        self._z_markers_idx = slice(-1)
        self._vzs = None
        self.stale = True

    def set_sizes(self, sizes, dpi=72.0):
        if False:
            for i in range(10):
                print('nop')
        super().set_sizes(sizes, dpi)
        if not self._in_draw:
            self._sizes3d = sizes

    def set_linewidth(self, lw):
        if False:
            print('Hello World!')
        super().set_linewidth(lw)
        if not self._in_draw:
            self._linewidths3d = np.array(self._linewidths)

    def get_depthshade(self):
        if False:
            i = 10
            return i + 15
        return self._depthshade

    def set_depthshade(self, depthshade):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set whether depth shading is performed on collection members.\n\n        Parameters\n        ----------\n        depthshade : bool\n            Whether to shade the patches in order to give the appearance of\n            depth.\n        '
        self._depthshade = depthshade
        self.stale = True

    def do_3d_projection(self):
        if False:
            for i in range(10):
                print('nop')
        (xs, ys, zs) = self._offsets3d
        (vxs, vys, vzs, vis) = proj3d.proj_transform_clip(xs, ys, zs, self.axes.M)
        z_markers_idx = self._z_markers_idx = np.argsort(vzs)[::-1]
        self._vzs = vzs
        if len(self._sizes3d) > 1:
            self._sizes = self._sizes3d[z_markers_idx]
        if len(self._linewidths3d) > 1:
            self._linewidths = self._linewidths3d[z_markers_idx]
        PathCollection.set_offsets(self, np.column_stack((vxs, vys)))
        vzs = vzs[z_markers_idx]
        vxs = vxs[z_markers_idx]
        vys = vys[z_markers_idx]
        self._offset_zordered = np.column_stack((vxs, vys))
        return np.min(vzs) if vzs.size else np.nan

    @contextmanager
    def _use_zordered_offset(self):
        if False:
            for i in range(10):
                print('nop')
        if self._offset_zordered is None:
            yield
        else:
            old_offset = self._offsets
            super().set_offsets(self._offset_zordered)
            try:
                yield
            finally:
                self._offsets = old_offset

    def _maybe_depth_shade_and_sort_colors(self, color_array):
        if False:
            print('Hello World!')
        color_array = _zalpha(color_array, self._vzs) if self._vzs is not None and self._depthshade else color_array
        if len(color_array) > 1:
            color_array = color_array[self._z_markers_idx]
        return mcolors.to_rgba_array(color_array, self._alpha)

    def get_facecolor(self):
        if False:
            while True:
                i = 10
        return self._maybe_depth_shade_and_sort_colors(super().get_facecolor())

    def get_edgecolor(self):
        if False:
            return 10
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        return self._maybe_depth_shade_and_sort_colors(super().get_edgecolor())

def patch_collection_2d_to_3d(col, zs=0, zdir='z', depthshade=True):
    if False:
        print('Hello World!')
    '\n    Convert a `.PatchCollection` into a `.Patch3DCollection` object\n    (or a `.PathCollection` into a `.Path3DCollection` object).\n\n    Parameters\n    ----------\n    zs : float or array of floats\n        The location or locations to place the patches in the collection along\n        the *zdir* axis. Default: 0.\n    zdir : {\'x\', \'y\', \'z\'}\n        The axis in which to place the patches. Default: "z".\n        See `.get_dir_vector` for a description of the values.\n    depthshade\n        Whether to shade the patches to give a sense of depth. Default: *True*.\n\n    '
    if isinstance(col, PathCollection):
        col.__class__ = Path3DCollection
        col._offset_zordered = None
    elif isinstance(col, PatchCollection):
        col.__class__ = Patch3DCollection
    col._depthshade = depthshade
    col._in_draw = False
    col.set_3d_properties(zs, zdir)

class Poly3DCollection(PolyCollection):
    """
    A collection of 3D polygons.

    .. note::
        **Filling of 3D polygons**

        There is no simple definition of the enclosed surface of a 3D polygon
        unless the polygon is planar.

        In practice, Matplotlib fills the 2D projection of the polygon. This
        gives a correct filling appearance only for planar polygons. For all
        other polygons, you'll find orientations in which the edges of the
        polygon intersect in the projection. This will lead to an incorrect
        visualization of the 3D area.

        If you need filled areas, it is recommended to create them via
        `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`, which creates a
        triangulation and thus generates consistent surfaces.
    """

    def __init__(self, verts, *args, zsort='average', shade=False, lightsource=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        verts : list of (N, 3) array-like\n            The sequence of polygons [*verts0*, *verts1*, ...] where each\n            element *verts_i* defines the vertices of polygon *i* as a 2D\n            array-like of shape (N, 3).\n        zsort : {'average', 'min', 'max'}, default: 'average'\n            The calculation method for the z-order.\n            See `~.Poly3DCollection.set_zsort` for details.\n        shade : bool, default: False\n            Whether to shade *facecolors* and *edgecolors*. When activating\n            *shade*, *facecolors* and/or *edgecolors* must be provided.\n\n            .. versionadded:: 3.7\n\n        lightsource : `~matplotlib.colors.LightSource`, optional\n            The lightsource to use when *shade* is True.\n\n            .. versionadded:: 3.7\n\n        *args, **kwargs\n            All other parameters are forwarded to `.PolyCollection`.\n\n        Notes\n        -----\n        Note that this class does a bit of magic with the _facecolors\n        and _edgecolors properties.\n        "
        if shade:
            normals = _generate_normals(verts)
            facecolors = kwargs.get('facecolors', None)
            if facecolors is not None:
                kwargs['facecolors'] = _shade_colors(facecolors, normals, lightsource)
            edgecolors = kwargs.get('edgecolors', None)
            if edgecolors is not None:
                kwargs['edgecolors'] = _shade_colors(edgecolors, normals, lightsource)
            if facecolors is None and edgecolors is None:
                raise ValueError('You must provide facecolors, edgecolors, or both for shade to work.')
        super().__init__(verts, *args, **kwargs)
        if isinstance(verts, np.ndarray):
            if verts.ndim != 3:
                raise ValueError('verts must be a list of (N, 3) array-like')
        elif any((len(np.shape(vert)) != 2 for vert in verts)):
            raise ValueError('verts must be a list of (N, 3) array-like')
        self.set_zsort(zsort)
        self._codes3d = None
    _zsort_functions = {'average': np.average, 'min': np.min, 'max': np.max}

    def set_zsort(self, zsort):
        if False:
            return 10
        "\n        Set the calculation method for the z-order.\n\n        Parameters\n        ----------\n        zsort : {'average', 'min', 'max'}\n            The function applied on the z-coordinates of the vertices in the\n            viewer's coordinate system, to determine the z-order.\n        "
        self._zsortfunc = self._zsort_functions[zsort]
        self._sort_zpos = None
        self.stale = True

    def get_vector(self, segments3d):
        if False:
            print('Hello World!')
        'Optimize points for projection.'
        if len(segments3d):
            (xs, ys, zs) = np.vstack(segments3d).T
        else:
            (xs, ys, zs) = ([], [], [])
        ones = np.ones(len(xs))
        self._vec = np.array([xs, ys, zs, ones])
        indices = [0, *np.cumsum([len(segment) for segment in segments3d])]
        self._segslices = [*map(slice, indices[:-1], indices[1:])]

    def set_verts(self, verts, closed=True):
        if False:
            i = 10
            return i + 15
        '\n        Set 3D vertices.\n\n        Parameters\n        ----------\n        verts : list of (N, 3) array-like\n            The sequence of polygons [*verts0*, *verts1*, ...] where each\n            element *verts_i* defines the vertices of polygon *i* as a 2D\n            array-like of shape (N, 3).\n        closed : bool, default: True\n            Whether the polygon should be closed by adding a CLOSEPOLY\n            connection at the end.\n        '
        self.get_vector(verts)
        super().set_verts([], False)
        self._closed = closed

    def set_verts_and_codes(self, verts, codes):
        if False:
            print('Hello World!')
        'Set 3D vertices with path codes.'
        self.set_verts(verts, closed=False)
        self._codes3d = codes

    def set_3d_properties(self):
        if False:
            print('Hello World!')
        self.update_scalarmappable()
        self._sort_zpos = None
        self.set_zsort('average')
        self._facecolor3d = PolyCollection.get_facecolor(self)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)
        self._alpha3d = PolyCollection.get_alpha(self)
        self.stale = True

    def set_sort_zpos(self, val):
        if False:
            i = 10
            return i + 15
        'Set the position to use for z-sorting.'
        self._sort_zpos = val
        self.stale = True

    def do_3d_projection(self):
        if False:
            print('Hello World!')
        '\n        Perform the 3D projection for this object.\n        '
        if self._A is not None:
            self.update_scalarmappable()
            if self._face_is_mapped:
                self._facecolor3d = self._facecolors
            if self._edge_is_mapped:
                self._edgecolor3d = self._edgecolors
        (txs, tys, tzs) = proj3d._proj_transform_vec(self._vec, self.axes.M)
        xyzlist = [(txs[sl], tys[sl], tzs[sl]) for sl in self._segslices]
        cface = self._facecolor3d
        cedge = self._edgecolor3d
        if len(cface) != len(xyzlist):
            cface = cface.repeat(len(xyzlist), axis=0)
        if len(cedge) != len(xyzlist):
            if len(cedge) == 0:
                cedge = cface
            else:
                cedge = cedge.repeat(len(xyzlist), axis=0)
        if xyzlist:
            z_segments_2d = sorted(((self._zsortfunc(zs), np.column_stack([xs, ys]), fc, ec, idx) for (idx, ((xs, ys, zs), fc, ec)) in enumerate(zip(xyzlist, cface, cedge))), key=lambda x: x[0], reverse=True)
            (_, segments_2d, self._facecolors2d, self._edgecolors2d, idxs) = zip(*z_segments_2d)
        else:
            segments_2d = []
            self._facecolors2d = np.empty((0, 4))
            self._edgecolors2d = np.empty((0, 4))
            idxs = []
        if self._codes3d is not None:
            codes = [self._codes3d[idx] for idx in idxs]
            PolyCollection.set_verts_and_codes(self, segments_2d, codes)
        else:
            PolyCollection.set_verts(self, segments_2d, self._closed)
        if len(self._edgecolor3d) != len(cface):
            self._edgecolors2d = self._edgecolor3d
        if self._sort_zpos is not None:
            zvec = np.array([[0], [0], [self._sort_zpos], [1]])
            ztrans = proj3d._proj_transform_vec(zvec, self.axes.M)
            return ztrans[2][0]
        elif tzs.size > 0:
            return np.min(tzs)
        else:
            return np.nan

    def set_facecolor(self, colors):
        if False:
            print('Hello World!')
        super().set_facecolor(colors)
        self._facecolor3d = PolyCollection.get_facecolor(self)

    def set_edgecolor(self, colors):
        if False:
            while True:
                i = 10
        super().set_edgecolor(colors)
        self._edgecolor3d = PolyCollection.get_edgecolor(self)

    def set_alpha(self, alpha):
        if False:
            return 10
        artist.Artist.set_alpha(self, alpha)
        try:
            self._facecolor3d = mcolors.to_rgba_array(self._facecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        try:
            self._edgecolors = mcolors.to_rgba_array(self._edgecolor3d, self._alpha)
        except (AttributeError, TypeError, IndexError):
            pass
        self.stale = True

    def get_facecolor(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '_facecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return np.asarray(self._facecolors2d)

    def get_edgecolor(self):
        if False:
            return 10
        if not hasattr(self, '_edgecolors2d'):
            self.axes.M = self.axes.get_proj()
            self.do_3d_projection()
        return np.asarray(self._edgecolors2d)

def poly_collection_2d_to_3d(col, zs=0, zdir='z'):
    if False:
        return 10
    "\n    Convert a `.PolyCollection` into a `.Poly3DCollection` object.\n\n    Parameters\n    ----------\n    zs : float or array of floats\n        The location or locations to place the polygons in the collection along\n        the *zdir* axis. Default: 0.\n    zdir : {'x', 'y', 'z'}\n        The axis in which to place the patches. Default: 'z'.\n        See `.get_dir_vector` for a description of the values.\n    "
    (segments_3d, codes) = _paths_to_3d_segments_with_codes(col.get_paths(), zs, zdir)
    col.__class__ = Poly3DCollection
    col.set_verts_and_codes(segments_3d, codes)
    col.set_3d_properties()

def juggle_axes(xs, ys, zs, zdir):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reorder coordinates so that 2D *xs*, *ys* can be plotted in the plane\n    orthogonal to *zdir*. *zdir* is normally 'x', 'y' or 'z'. However, if\n    *zdir* starts with a '-' it is interpreted as a compensation for\n    `rotate_axes`.\n    "
    if zdir == 'x':
        return (zs, xs, ys)
    elif zdir == 'y':
        return (xs, zs, ys)
    elif zdir[0] == '-':
        return rotate_axes(xs, ys, zs, zdir)
    else:
        return (xs, ys, zs)

def rotate_axes(xs, ys, zs, zdir):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reorder coordinates so that the axes are rotated with *zdir* along\n    the original z axis. Prepending the axis with a '-' does the\n    inverse transform, so *zdir* can be 'x', '-x', 'y', '-y', 'z' or '-z'.\n    "
    if zdir in ('x', '-y'):
        return (ys, zs, xs)
    elif zdir in ('-x', 'y'):
        return (zs, xs, ys)
    else:
        return (xs, ys, zs)

def _zalpha(colors, zs):
    if False:
        for i in range(10):
            print('nop')
    'Modify the alphas of the color list according to depth.'
    if len(colors) == 0 or len(zs) == 0:
        return np.zeros((0, 4))
    norm = Normalize(min(zs), max(zs))
    sats = 1 - norm(zs) * 0.7
    rgba = np.broadcast_to(mcolors.to_rgba_array(colors), (len(zs), 4))
    return np.column_stack([rgba[:, :3], rgba[:, 3] * sats])

def _generate_normals(polygons):
    if False:
        while True:
            i = 10
    '\n    Compute the normals of a list of polygons, one normal per polygon.\n\n    Normals point towards the viewer for a face with its vertices in\n    counterclockwise order, following the right hand rule.\n\n    Uses three points equally spaced around the polygon. This method assumes\n    that the points are in a plane. Otherwise, more than one shade is required,\n    which is not supported.\n\n    Parameters\n    ----------\n    polygons : list of (M_i, 3) array-like, or (..., M, 3) array-like\n        A sequence of polygons to compute normals for, which can have\n        varying numbers of vertices. If the polygons all have the same\n        number of vertices and array is passed, then the operation will\n        be vectorized.\n\n    Returns\n    -------\n    normals : (..., 3) array\n        A normal vector estimated for the polygon.\n    '
    if isinstance(polygons, np.ndarray):
        n = polygons.shape[-2]
        (i1, i2, i3) = (0, n // 3, 2 * n // 3)
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for (poly_i, ps) in enumerate(polygons):
            n = len(ps)
            (i1, i2, i3) = (0, n // 3, 2 * n // 3)
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]
    return np.cross(v1, v2)

def _shade_colors(color, normals, lightsource=None):
    if False:
        return 10
    '\n    Shade *color* using normal vectors given by *normals*,\n    assuming a *lightsource* (using default position if not given).\n    *color* can also be an array of the same length as *normals*.\n    '
    if lightsource is None:
        lightsource = mcolors.LightSource(azdeg=225, altdeg=19.4712)
    with np.errstate(invalid='ignore'):
        shade = normals / np.linalg.norm(normals, axis=1, keepdims=True) @ lightsource.direction
    mask = ~np.isnan(shade)
    if mask.any():
        in_norm = mcolors.Normalize(-1, 1)
        out_norm = mcolors.Normalize(0.3, 1).inverse

        def norm(x):
            if False:
                i = 10
                return i + 15
            return out_norm(in_norm(x))
        shade[~mask] = 0
        color = mcolors.to_rgba_array(color)
        alpha = color[:, 3]
        colors = norm(shade)[:, np.newaxis] * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()
    return colors