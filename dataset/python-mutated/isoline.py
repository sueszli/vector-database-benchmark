from __future__ import division
import numpy as np
from .line import LineVisual
from ..color import ColorArray
from ..color.colormap import _normalize, get_colormap

def iso_mesh_line(vertices, tris, vertex_data, levels):
    if False:
        print('Hello World!')
    'Generate an isocurve from vertex data in a surface mesh.\n\n    Parameters\n    ----------\n    vertices : ndarray, shape (Nv, 3)\n        Vertex coordinates.\n    tris : ndarray, shape (Nf, 3)\n        Indices of triangular element into the vertices array.\n    vertex_data : ndarray, shape (Nv,)\n        data at vertex.\n    levels : ndarray, shape (Nl,)\n        Levels at which to generate an isocurve\n\n    Returns\n    -------\n    lines : ndarray, shape (Nvout, 3)\n        Vertex coordinates for lines points\n    connects : ndarray, shape (Ne, 2)\n        Indices of line element into the vertex array.\n    vertex_level: ndarray, shape (Nvout,)\n        level for vertex in lines\n\n    Notes\n    -----\n    Uses a marching squares algorithm to generate the isolines.\n    '
    lines = None
    connects = None
    vertex_level = None
    level_index = None
    if not all([isinstance(x, np.ndarray) for x in (vertices, tris, vertex_data, levels)]):
        raise ValueError('all inputs must be numpy arrays')
    if vertices.shape[1] <= 3:
        verts = vertices
    elif vertices.shape[1] == 4:
        verts = vertices[:, :-1]
    else:
        verts = None
    if verts is not None and tris.shape[1] == 3 and (vertex_data.shape[0] == verts.shape[0]):
        edges = np.vstack((tris.reshape(-1), np.roll(tris, -1, axis=1).reshape(-1))).T
        edge_datas = vertex_data[edges]
        edge_coors = verts[edges].reshape(tris.shape[0] * 3, 2, 3)
        for lev in levels:
            index = edge_datas >= lev
            index = index[:, 0] ^ index[:, 1]
            edge_datas_Ok = edge_datas[index, :]
            xyz = edge_coors[index]
            ratio = np.array([(lev - edge_datas_Ok[:, 0]) / (edge_datas_Ok[:, 1] - edge_datas_Ok[:, 0])])
            point = xyz[:, 0, :] + ratio.T * (xyz[:, 1, :] - xyz[:, 0, :])
            nbr = point.shape[0] // 2
            if connects is not None:
                connect = np.arange(0, nbr * 2).reshape((nbr, 2)) + len(lines)
                connects = np.append(connects, connect, axis=0)
                lines = np.append(lines, point, axis=0)
                vertex_level = np.append(vertex_level, np.zeros(len(point)) + lev)
                level_index = np.append(level_index, np.array(len(point)))
            else:
                lines = point
                connects = np.arange(0, nbr * 2).reshape((nbr, 2))
                vertex_level = np.zeros(len(point)) + lev
                level_index = np.array(len(point))
            vertex_level = vertex_level.reshape((vertex_level.size, 1))
    return (lines, connects, vertex_level, level_index)

class IsolineVisual(LineVisual):
    """Isocurves of a tri mesh with data at vertices at different levels.

    Parameters
    ----------
    vertices : ndarray, shape (Nv, 3) | None
        Vertex coordinates.
    tris : ndarray, shape (Nf, 3) | None
        Indices into the vertex array.
    data : ndarray, shape (Nv,) | None
        scalar at vertices
    levels : ndarray, shape (Nlev,) | None
        The levels at which the isocurve is constructed from "data".
    color_lev : Color, tuple, colormap name or array
        The color to use when drawing the line. If an array is given, it
        must be of shape (Nlev, 4) and provide one rgba color by level.
    **kwargs : dict
        Keyword arguments to pass to `LineVisual`.
    """

    def __init__(self, vertices=None, tris=None, data=None, levels=None, color_lev=None, **kwargs):
        if False:
            while True:
                i = 10
        self._data = None
        self._vertices = None
        self._tris = None
        self._levels = levels
        self._color_lev = color_lev
        self._need_color_update = True
        self._need_recompute = True
        self._v = None
        self._c = None
        self._vl = None
        self._li = None
        self._lc = None
        self._cl = None
        self._update_color_lev = False
        kwargs['antialias'] = False
        LineVisual.__init__(self, method='gl', **kwargs)
        self.set_data(vertices=vertices, tris=tris, data=data)

    @property
    def levels(self):
        if False:
            print('Hello World!')
        'The threshold at which the isocurves are constructed from the data.'
        return self._levels

    @levels.setter
    def levels(self, levels):
        if False:
            for i in range(10):
                print('nop')
        self._levels = levels
        self._need_recompute = True
        self.update()

    @property
    def data(self):
        if False:
            i = 10
            return i + 15
        'The mesh data'
        return (self._vertices, self._tris, self._data)

    def set_data(self, vertices=None, tris=None, data=None):
        if False:
            print('Hello World!')
        'Set the data\n\n        Parameters\n        ----------\n        vertices : ndarray, shape (Nv, 3) | None\n            Vertex coordinates.\n        tris : ndarray, shape (Nf, 3) | None\n            Indices into the vertex array.\n        data : ndarray, shape (Nv,) | None\n            scalar at vertices\n        '
        if data is not None:
            self._data = data
            self._need_recompute = True
        if vertices is not None:
            self._vertices = vertices
            self._need_recompute = True
        if tris is not None:
            self._tris = tris
            self._need_recompute = True
        self.update()

    @property
    def color(self):
        if False:
            print('Hello World!')
        return self._color_lev

    def set_color(self, color):
        if False:
            for i in range(10):
                print('nop')
        'Set the color\n\n        Parameters\n        ----------\n        color : instance of Color\n            The color to use.\n        '
        if color is not None:
            self._color_lev = color
            self._need_color_update = True
            self.update()

    def _levels_to_colors(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            f_color_levs = get_colormap(self._color_lev)
        except (KeyError, TypeError):
            colors = ColorArray(self._color_lev).rgba
        else:
            lev = _normalize(self._levels, self._levels.min(), self._levels.max())
            colors = f_color_levs.map(lev[:, np.newaxis])
        if len(colors) == 1:
            colors = colors * np.ones((len(self._levels), 1))
        if len(colors) != len(self._levels):
            raise TypeError('Color/level mismatch. Color must be of shape (Nlev, ...) and provide one color per level')
        self._lc = colors

    def _compute_iso_color(self):
        if False:
            for i in range(10):
                print('nop')
        'Compute LineVisual color from level index and corresponding level color'
        level_color = []
        colors = self._lc
        for (i, index) in enumerate(self._li):
            level_color.append(np.zeros((index, 4)) + colors[i])
        self._cl = np.vstack(level_color)

    def _prepare_draw(self, view):
        if False:
            print('Hello World!')
        if self._data is None or self._levels is None or self._tris is None or (self._vertices is None) or (self._color_lev is None):
            return False
        if self._need_recompute:
            (self._v, self._c, self._vl, self._li) = iso_mesh_line(self._vertices, self._tris, self._data, self._levels)
            self._levels_to_colors()
            self._compute_iso_color()
            LineVisual.set_data(self, pos=self._v, connect=self._c, color=self._cl)
            self._need_recompute = False
        if self._need_color_update:
            self._levels_to_colors()
            self._compute_iso_color()
            LineVisual.set_data(self, color=self._cl)
            self._update_color_lev = False
        return LineVisual._prepare_draw(self, view)