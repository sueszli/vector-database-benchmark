import numpy as np
from matplotlib import _docstring
from matplotlib.contour import ContourSet
from matplotlib.tri._triangulation import Triangulation

@_docstring.dedent_interpd
class TriContourSet(ContourSet):
    """
    Create and store a set of contour lines or filled regions for
    a triangular grid.

    This class is typically not instantiated directly by the user but by
    `~.Axes.tricontour` and `~.Axes.tricontourf`.

    %(contour_set_attributes)s
    """

    def __init__(self, ax, *args, **kwargs):
        if False:
            return 10
        '\n        Draw triangular grid contour lines or filled regions,\n        depending on whether keyword arg *filled* is False\n        (default) or True.\n\n        The first argument of the initializer must be an `~.axes.Axes`\n        object.  The remaining arguments and keyword arguments\n        are described in the docstring of `~.Axes.tricontour`.\n        '
        super().__init__(ax, *args, **kwargs)

    def _process_args(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Process args and kwargs.\n        '
        if isinstance(args[0], TriContourSet):
            C = args[0]._contour_generator
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
            self._mins = args[0]._mins
            self._maxs = args[0]._maxs
        else:
            from matplotlib import _tri
            (tri, z) = self._contour_args(args, kwargs)
            C = _tri.TriContourGenerator(tri.get_cpp_triangulation(), z)
            self._mins = [tri.x.min(), tri.y.min()]
            self._maxs = [tri.x.max(), tri.y.max()]
        self._contour_generator = C
        return kwargs

    def _contour_args(self, args, kwargs):
        if False:
            print('Hello World!')
        (tri, args, kwargs) = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
        (z, *args) = args
        z = np.ma.asarray(z)
        if z.shape != tri.x.shape:
            raise ValueError('z array must have same length as triangulation x and y arrays')
        z_check = z[np.unique(tri.get_masked_triangles())]
        if np.ma.is_masked(z_check):
            raise ValueError('z must not contain masked points within the triangulation')
        if not np.isfinite(z_check).all():
            raise ValueError('z array must not contain non-finite values within the triangulation')
        z = np.ma.masked_invalid(z, copy=False)
        self.zmax = float(z_check.max())
        self.zmin = float(z_check.min())
        if self.logscale and self.zmin <= 0:
            func = 'contourf' if self.filled else 'contour'
            raise ValueError(f'Cannot {func} log of negative values.')
        self._process_contour_level_args(args, z.dtype)
        return (tri, z)
_docstring.interpd.update(_tricontour_doc='\nDraw contour %%(type)s on an unstructured triangular grid.\n\nCall signatures::\n\n    %%(func)s(triangulation, z, [levels], ...)\n    %%(func)s(x, y, z, [levels], *, [triangles=triangles], [mask=mask], ...)\n\nThe triangular grid can be specified either by passing a `.Triangulation`\nobject as the first parameter, or by passing the points *x*, *y* and\noptionally the *triangles* and a *mask*. See `.Triangulation` for an\nexplanation of these parameters. If neither of *triangulation* or\n*triangles* are given, the triangulation is calculated on the fly.\n\nIt is possible to pass *triangles* positionally, i.e.\n``%%(func)s(x, y, triangles, z, ...)``. However, this is discouraged. For more\nclarity, pass *triangles* via keyword argument.\n\nParameters\n----------\ntriangulation : `.Triangulation`, optional\n    An already created triangular grid.\n\nx, y, triangles, mask\n    Parameters defining the triangular grid. See `.Triangulation`.\n    This is mutually exclusive with specifying *triangulation*.\n\nz : array-like\n    The height values over which the contour is drawn.  Color-mapping is\n    controlled by *cmap*, *norm*, *vmin*, and *vmax*.\n\n    .. note::\n        All values in *z* must be finite. Hence, nan and inf values must\n        either be removed or `~.Triangulation.set_mask` be used.\n\nlevels : int or array-like, optional\n    Determines the number and positions of the contour lines / regions.\n\n    If an int *n*, use `~matplotlib.ticker.MaxNLocator`, which tries to\n    automatically choose no more than *n+1* "nice" contour levels between\n    between minimum and maximum numeric values of *Z*.\n\n    If array-like, draw contour lines at the specified levels.  The values must\n    be in increasing order.\n\nReturns\n-------\n`~matplotlib.tri.TriContourSet`\n\nOther Parameters\n----------------\ncolors : color string or sequence of colors, optional\n    The colors of the levels, i.e., the contour %%(type)s.\n\n    The sequence is cycled for the levels in ascending order. If the sequence\n    is shorter than the number of levels, it is repeated.\n\n    As a shortcut, single color strings may be used in place of one-element\n    lists, i.e. ``\'red\'`` instead of ``[\'red\']`` to color all levels with the\n    same color. This shortcut does only work for color strings, not for other\n    ways of specifying colors.\n\n    By default (value *None*), the colormap specified by *cmap* will be used.\n\nalpha : float, default: 1\n    The alpha blending value, between 0 (transparent) and 1 (opaque).\n\n%(cmap_doc)s\n\n    This parameter is ignored if *colors* is set.\n\n%(norm_doc)s\n\n    This parameter is ignored if *colors* is set.\n\n%(vmin_vmax_doc)s\n\n    If *vmin* or *vmax* are not given, the default color scaling is based on\n    *levels*.\n\n    This parameter is ignored if *colors* is set.\n\norigin : {*None*, \'upper\', \'lower\', \'image\'}, default: None\n    Determines the orientation and exact position of *z* by specifying the\n    position of ``z[0, 0]``.  This is only relevant, if *X*, *Y* are not given.\n\n    - *None*: ``z[0, 0]`` is at X=0, Y=0 in the lower left corner.\n    - \'lower\': ``z[0, 0]`` is at X=0.5, Y=0.5 in the lower left corner.\n    - \'upper\': ``z[0, 0]`` is at X=N+0.5, Y=0.5 in the upper left corner.\n    - \'image\': Use the value from :rc:`image.origin`.\n\nextent : (x0, x1, y0, y1), optional\n    If *origin* is not *None*, then *extent* is interpreted as in `.imshow`: it\n    gives the outer pixel boundaries. In this case, the position of z[0, 0] is\n    the center of the pixel, not a corner. If *origin* is *None*, then\n    (*x0*, *y0*) is the position of z[0, 0], and (*x1*, *y1*) is the position\n    of z[-1, -1].\n\n    This argument is ignored if *X* and *Y* are specified in the call to\n    contour.\n\nlocator : ticker.Locator subclass, optional\n    The locator is used to determine the contour levels if they are not given\n    explicitly via *levels*.\n    Defaults to `~.ticker.MaxNLocator`.\n\nextend : {\'neither\', \'both\', \'min\', \'max\'}, default: \'neither\'\n    Determines the ``%%(func)s``-coloring of values that are outside the\n    *levels* range.\n\n    If \'neither\', values outside the *levels* range are not colored.  If \'min\',\n    \'max\' or \'both\', color the values below, above or below and above the\n    *levels* range.\n\n    Values below ``min(levels)`` and above ``max(levels)`` are mapped to the\n    under/over values of the `.Colormap`. Note that most colormaps do not have\n    dedicated colors for these by default, so that the over and under values\n    are the edge values of the colormap.  You may want to set these values\n    explicitly using `.Colormap.set_under` and `.Colormap.set_over`.\n\n    .. note::\n\n        An existing `.TriContourSet` does not get notified if properties of its\n        colormap are changed. Therefore, an explicit call to\n        `.ContourSet.changed()` is needed after modifying the colormap. The\n        explicit call can be left out, if a colorbar is assigned to the\n        `.TriContourSet` because it internally calls `.ContourSet.changed()`.\n\nxunits, yunits : registered units, optional\n    Override axis units by specifying an instance of a\n    :class:`matplotlib.units.ConversionInterface`.\n\nantialiased : bool, optional\n    Enable antialiasing, overriding the defaults.  For\n    filled contours, the default is *True*.  For line contours,\n    it is taken from :rc:`lines.antialiased`.' % _docstring.interpd.params)

@_docstring.Substitution(func='tricontour', type='lines')
@_docstring.dedent_interpd
def tricontour(ax, *args, **kwargs):
    if False:
        print('Hello World!')
    "\n    %(_tricontour_doc)s\n\n    linewidths : float or array-like, default: :rc:`contour.linewidth`\n        The line width of the contour lines.\n\n        If a number, all levels will be plotted with this linewidth.\n\n        If a sequence, the levels in ascending order will be plotted with\n        the linewidths in the order specified.\n\n        If None, this falls back to :rc:`lines.linewidth`.\n\n    linestyles : {*None*, 'solid', 'dashed', 'dashdot', 'dotted'}, optional\n        If *linestyles* is *None*, the default is 'solid' unless the lines are\n        monochrome.  In that case, negative contours will take their linestyle\n        from :rc:`contour.negative_linestyle` setting.\n\n        *linestyles* can also be an iterable of the above strings specifying a\n        set of linestyles to be used. If this iterable is shorter than the\n        number of contour levels it will be repeated as necessary.\n    "
    kwargs['filled'] = False
    return TriContourSet(ax, *args, **kwargs)

@_docstring.Substitution(func='tricontourf', type='regions')
@_docstring.dedent_interpd
def tricontourf(ax, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    %(_tricontour_doc)s\n\n    hatches : list[str], optional\n        A list of crosshatch patterns to use on the filled areas.\n        If None, no hatching will be added to the contour.\n        Hatching is supported in the PostScript, PDF, SVG and Agg\n        backends only.\n\n    Notes\n    -----\n    `.tricontourf` fills intervals that are closed at the top; that is, for\n    boundaries *z1* and *z2*, the filled region is::\n\n        z1 < Z <= z2\n\n    except for the lowest interval, which is closed on both sides (i.e. it\n    includes the lowest value).\n    '
    kwargs['filled'] = True
    return TriContourSet(ax, *args, **kwargs)