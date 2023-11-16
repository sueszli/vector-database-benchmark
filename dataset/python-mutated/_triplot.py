import numpy as np
from matplotlib.tri._triangulation import Triangulation
import matplotlib.cbook as cbook
import matplotlib.lines as mlines

def triplot(ax, *args, **kwargs):
    if False:
        return 10
    '\n    Draw an unstructured triangular grid as lines and/or markers.\n\n    Call signatures::\n\n      triplot(triangulation, ...)\n      triplot(x, y, [triangles], *, [mask=mask], ...)\n\n    The triangular grid can be specified either by passing a `.Triangulation`\n    object as the first parameter, or by passing the points *x*, *y* and\n    optionally the *triangles* and a *mask*. If neither of *triangulation* or\n    *triangles* are given, the triangulation is calculated on the fly.\n\n    Parameters\n    ----------\n    triangulation : `.Triangulation`\n        An already created triangular grid.\n    x, y, triangles, mask\n        Parameters defining the triangular grid. See `.Triangulation`.\n        This is mutually exclusive with specifying *triangulation*.\n    other_parameters\n        All other args and kwargs are forwarded to `~.Axes.plot`.\n\n    Returns\n    -------\n    lines : `~matplotlib.lines.Line2D`\n        The drawn triangles edges.\n    markers : `~matplotlib.lines.Line2D`\n        The drawn marker nodes.\n    '
    import matplotlib.axes
    (tri, args, kwargs) = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    (x, y, edges) = (tri.x, tri.y, tri.edges)
    fmt = args[0] if args else ''
    (linestyle, marker, color) = matplotlib.axes._base._process_plot_format(fmt)
    kw = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    for (key, val) in zip(('linestyle', 'marker', 'color'), (linestyle, marker, color)):
        if val is not None:
            kw.setdefault(key, val)
    linestyle = kw['linestyle']
    kw_lines = {**kw, 'marker': 'None', 'zorder': kw.get('zorder', 1)}
    if linestyle not in [None, 'None', '', ' ']:
        tri_lines_x = np.insert(x[edges], 2, np.nan, axis=1)
        tri_lines_y = np.insert(y[edges], 2, np.nan, axis=1)
        tri_lines = ax.plot(tri_lines_x.ravel(), tri_lines_y.ravel(), **kw_lines)
    else:
        tri_lines = ax.plot([], [], **kw_lines)
    marker = kw['marker']
    kw_markers = {**kw, 'linestyle': 'None'}
    kw_markers.pop('label', None)
    if marker not in [None, 'None', '', ' ']:
        tri_markers = ax.plot(x, y, **kw_markers)
    else:
        tri_markers = ax.plot([], [], **kw_markers)
    return tri_lines + tri_markers