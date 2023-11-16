import numpy as np
from scipy._lib.decorator import decorator as _decorator
__all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d']

@_decorator
def _held_figure(func, obj, ax=None, **kw):
    if False:
        i = 10
        return i + 15
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        return func(obj, ax=ax, **kw)
    was_held = getattr(ax, 'ishold', lambda : True)()
    if was_held:
        return func(obj, ax=ax, **kw)
    try:
        ax.hold(True)
        return func(obj, ax=ax, **kw)
    finally:
        ax.hold(was_held)

def _adjust_bounds(ax, points):
    if False:
        for i in range(10):
            print('nop')
    margin = 0.1 * np.ptp(points, axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])

@_held_figure
def delaunay_plot_2d(tri, ax=None):
    if False:
        i = 10
        return i + 15
    '\n    Plot the given Delaunay triangulation in 2-D\n\n    Parameters\n    ----------\n    tri : scipy.spatial.Delaunay instance\n        Triangulation to plot\n    ax : matplotlib.axes.Axes instance, optional\n        Axes to plot on\n\n    Returns\n    -------\n    fig : matplotlib.figure.Figure instance\n        Figure for the plot\n\n    See Also\n    --------\n    Delaunay\n    matplotlib.pyplot.triplot\n\n    Notes\n    -----\n    Requires Matplotlib.\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.spatial import Delaunay, delaunay_plot_2d\n\n    The Delaunay triangulation of a set of random points:\n\n    >>> rng = np.random.default_rng()\n    >>> points = rng.random((30, 2))\n    >>> tri = Delaunay(points)\n\n    Plot it:\n\n    >>> _ = delaunay_plot_2d(tri)\n    >>> plt.show()\n\n    '
    if tri.points.shape[1] != 2:
        raise ValueError('Delaunay triangulation is not 2-D')
    (x, y) = tri.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri.simplices.copy())
    _adjust_bounds(ax, tri.points)
    return ax.figure

@_held_figure
def convex_hull_plot_2d(hull, ax=None):
    if False:
        while True:
            i = 10
    '\n    Plot the given convex hull diagram in 2-D\n\n    Parameters\n    ----------\n    hull : scipy.spatial.ConvexHull instance\n        Convex hull to plot\n    ax : matplotlib.axes.Axes instance, optional\n        Axes to plot on\n\n    Returns\n    -------\n    fig : matplotlib.figure.Figure instance\n        Figure for the plot\n\n    See Also\n    --------\n    ConvexHull\n\n    Notes\n    -----\n    Requires Matplotlib.\n\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.spatial import ConvexHull, convex_hull_plot_2d\n\n    The convex hull of a random set of points:\n\n    >>> rng = np.random.default_rng()\n    >>> points = rng.random((30, 2))\n    >>> hull = ConvexHull(points)\n\n    Plot it:\n\n    >>> _ = convex_hull_plot_2d(hull)\n    >>> plt.show()\n\n    '
    from matplotlib.collections import LineCollection
    if hull.points.shape[1] != 2:
        raise ValueError('Convex hull is not 2-D')
    ax.plot(hull.points[:, 0], hull.points[:, 1], 'o')
    line_segments = [hull.points[simplex] for simplex in hull.simplices]
    ax.add_collection(LineCollection(line_segments, colors='k', linestyle='solid'))
    _adjust_bounds(ax, hull.points)
    return ax.figure

@_held_figure
def voronoi_plot_2d(vor, ax=None, **kw):
    if False:
        return 10
    "\n    Plot the given Voronoi diagram in 2-D\n\n    Parameters\n    ----------\n    vor : scipy.spatial.Voronoi instance\n        Diagram to plot\n    ax : matplotlib.axes.Axes instance, optional\n        Axes to plot on\n    show_points : bool, optional\n        Add the Voronoi points to the plot.\n    show_vertices : bool, optional\n        Add the Voronoi vertices to the plot.\n    line_colors : string, optional\n        Specifies the line color for polygon boundaries\n    line_width : float, optional\n        Specifies the line width for polygon boundaries\n    line_alpha : float, optional\n        Specifies the line alpha for polygon boundaries\n    point_size : float, optional\n        Specifies the size of points\n\n    Returns\n    -------\n    fig : matplotlib.figure.Figure instance\n        Figure for the plot\n\n    See Also\n    --------\n    Voronoi\n\n    Notes\n    -----\n    Requires Matplotlib.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy.spatial import Voronoi, voronoi_plot_2d\n\n    Create a set of points for the example:\n\n    >>> rng = np.random.default_rng()\n    >>> points = rng.random((10,2))\n\n    Generate the Voronoi diagram for the points:\n\n    >>> vor = Voronoi(points)\n\n    Use `voronoi_plot_2d` to plot the diagram:\n\n    >>> fig = voronoi_plot_2d(vor)\n\n    Use `voronoi_plot_2d` to plot the diagram again, with some settings\n    customized:\n\n    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',\n    ...                       line_width=2, line_alpha=0.6, point_size=2)\n    >>> plt.show()\n\n    "
    from matplotlib.collections import LineCollection
    if vor.points.shape[1] != 2:
        raise ValueError('Voronoi diagram is not 2-D')
    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], '.', markersize=point_size)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')
    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)
    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)
    finite_segments = []
    infinite_segments = []
    for (pointidx, simplex) in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if vor.furthest_site:
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()
            infinite_segments.append([vor.vertices[i], far_point])
    ax.add_collection(LineCollection(finite_segments, colors=line_colors, lw=line_width, alpha=line_alpha, linestyle='solid'))
    ax.add_collection(LineCollection(infinite_segments, colors=line_colors, lw=line_width, alpha=line_alpha, linestyle='dashed'))
    _adjust_bounds(ax, vor.points)
    return ax.figure