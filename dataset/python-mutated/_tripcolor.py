import numpy as np
from matplotlib import _api
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.tri._triangulation import Triangulation

def tripcolor(ax, *args, alpha=1.0, norm=None, cmap=None, vmin=None, vmax=None, shading='flat', facecolors=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Create a pseudocolor plot of an unstructured triangular grid.\n\n    Call signatures::\n\n      tripcolor(triangulation, c, *, ...)\n      tripcolor(x, y, c, *, [triangles=triangles], [mask=mask], ...)\n\n    The triangular grid can be specified either by passing a `.Triangulation`\n    object as the first parameter, or by passing the points *x*, *y* and\n    optionally the *triangles* and a *mask*. See `.Triangulation` for an\n    explanation of these parameters.\n\n    It is possible to pass the triangles positionally, i.e.\n    ``tripcolor(x, y, triangles, c, ...)``. However, this is discouraged.\n    For more clarity, pass *triangles* via keyword argument.\n\n    If neither of *triangulation* or *triangles* are given, the triangulation\n    is calculated on the fly. In this case, it does not make sense to provide\n    colors at the triangle faces via *c* or *facecolors* because there are\n    multiple possible triangulations for a group of points and you don't know\n    which triangles will be constructed.\n\n    Parameters\n    ----------\n    triangulation : `.Triangulation`\n        An already created triangular grid.\n    x, y, triangles, mask\n        Parameters defining the triangular grid. See `.Triangulation`.\n        This is mutually exclusive with specifying *triangulation*.\n    c : array-like\n        The color values, either for the points or for the triangles. Which one\n        is automatically inferred from the length of *c*, i.e. does it match\n        the number of points or the number of triangles. If there are the same\n        number of points and triangles in the triangulation it is assumed that\n        color values are defined at points; to force the use of color values at\n        triangles use the keyword argument ``facecolors=c`` instead of just\n        ``c``.\n        This parameter is position-only.\n    facecolors : array-like, optional\n        Can be used alternatively to *c* to specify colors at the triangle\n        faces. This parameter takes precedence over *c*.\n    shading : {'flat', 'gouraud'}, default: 'flat'\n        If  'flat' and the color values *c* are defined at points, the color\n        values used for each triangle are from the mean c of the triangle's\n        three points. If *shading* is 'gouraud' then color values must be\n        defined at points.\n    other_parameters\n        All other parameters are the same as for `~.Axes.pcolor`.\n    "
    _api.check_in_list(['flat', 'gouraud'], shading=shading)
    (tri, args, kwargs) = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
    if facecolors is not None:
        if args:
            _api.warn_external('Positional parameter c has no effect when the keyword facecolors is given')
        point_colors = None
        if len(facecolors) != len(tri.triangles):
            raise ValueError('The length of facecolors must match the number of triangles')
    else:
        if not args:
            raise TypeError("tripcolor() missing 1 required positional argument: 'c'; or 1 required keyword-only argument: 'facecolors'")
        elif len(args) > 1:
            raise TypeError(f'Unexpected positional parameters: {args[1:]!r}')
        c = np.asarray(args[0])
        if len(c) == len(tri.x):
            point_colors = c
            facecolors = None
        elif len(c) == len(tri.triangles):
            point_colors = None
            facecolors = c
        else:
            raise ValueError('The length of c must match either the number of points or the number of triangles')
    linewidths = (0.25,)
    if 'linewidth' in kwargs:
        kwargs['linewidths'] = kwargs.pop('linewidth')
    kwargs.setdefault('linewidths', linewidths)
    edgecolors = 'none'
    if 'edgecolor' in kwargs:
        kwargs['edgecolors'] = kwargs.pop('edgecolor')
    ec = kwargs.setdefault('edgecolors', edgecolors)
    if 'antialiased' in kwargs:
        kwargs['antialiaseds'] = kwargs.pop('antialiased')
    if 'antialiaseds' not in kwargs and ec.lower() == 'none':
        kwargs['antialiaseds'] = False
    if shading == 'gouraud':
        if facecolors is not None:
            raise ValueError("shading='gouraud' can only be used when the colors are specified at the points, not at the faces.")
        collection = TriMesh(tri, alpha=alpha, array=point_colors, cmap=cmap, norm=norm, **kwargs)
    else:
        maskedTris = tri.get_masked_triangles()
        verts = np.stack((tri.x[maskedTris], tri.y[maskedTris]), axis=-1)
        if facecolors is None:
            colors = point_colors[maskedTris].mean(axis=1)
        elif tri.mask is not None:
            colors = facecolors[~tri.mask]
        else:
            colors = facecolors
        collection = PolyCollection(verts, alpha=alpha, array=colors, cmap=cmap, norm=norm, **kwargs)
    collection._scale_norm(norm, vmin, vmax)
    ax.grid(False)
    minx = tri.x.min()
    maxx = tri.x.max()
    miny = tri.y.min()
    maxy = tri.y.max()
    corners = ((minx, miny), (maxx, maxy))
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(collection)
    return collection