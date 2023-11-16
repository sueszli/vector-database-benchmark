from __future__ import annotations
import typing
from collections import Counter
from contextlib import suppress
from warnings import warn
import numpy as np
from ..doctools import document
from ..exceptions import PlotnineWarning
from ..utils import SIZE_FACTOR, make_line_segments, match, to_rgba
from .geom import geom
if typing.TYPE_CHECKING:
    from typing import Any, Literal, Sequence
    import numpy.typing as npt
    import pandas as pd
    from matplotlib.path import Path
    from plotnine.iapi import panel_view
    from plotnine.typing import Axes, Coord, DrawingArea, Layer, TupleFloat2

@document
class geom_path(geom):
    """
    Connected points

    {usage}

    Parameters
    ----------
    {common_parameters}
    lineend : str (default: butt)
        Line end style, of of *butt*, *round* or *projecting.*
        This option is applied for solid linetypes.
    linejoin : str (default: round)
        Line join style, one of *round*, *miter* or *bevel*.
        This option is applied for solid linetypes.
    arrow : plotnine.geoms.geom_path.arrow (default: None)
        Arrow specification. Default is no arrow.

    See Also
    --------
    plotnine.geoms.arrow : for adding arrowhead(s) to paths.
    """
    DEFAULT_AES = {'alpha': 1, 'color': 'black', 'linetype': 'solid', 'size': 0.5}
    REQUIRED_AES = {'x', 'y'}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False, 'lineend': 'butt', 'linejoin': 'round', 'arrow': None}

    def handle_na(self, data: pd.DataFrame) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15

        def keep(x: Sequence[float]) -> npt.NDArray[np.bool_]:
            if False:
                for i in range(10):
                    print('nop')
            first = match([False], x, nomatch=1, start=0)[0]
            last = len(x) - match([False], x[::-1], nomatch=1, start=0)[0]
            bool_idx = np.hstack([np.repeat(False, first), np.repeat(True, last - first), np.repeat(False, len(x) - last)])
            return bool_idx
        bool_idx = data[['x', 'y', 'size', 'color', 'linetype']].isna().apply(keep, axis=0)
        bool_idx = np.all(bool_idx, axis=1)
        n1 = len(data)
        data = data[bool_idx]
        data.reset_index(drop=True, inplace=True)
        n2 = len(data)
        if n2 != n1 and (not self.params['na_rm']):
            msg = 'geom_path: Removed {} rows containing missing values.'
            warn(msg.format(n1 - n2), PlotnineWarning)
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            while True:
                i = 10
        if not any(data['group'].duplicated()):
            warn('geom_path: Each group consist of only one observation. Do you need to adjust the group aesthetic?', PlotnineWarning)
        c = Counter(data['group'])
        counts = np.array([c[v] for v in data['group']])
        data = data[counts >= 2]
        if len(data) < 2:
            return
        data = data.sort_values('group', kind='mergesort')
        data.reset_index(drop=True, inplace=True)
        cols = {'color', 'size', 'linetype', 'alpha', 'group'}
        cols = cols & set(data.columns)
        num_unique_rows = len(data.drop_duplicates(cols))
        ngroup = len(np.unique(data['group'].to_numpy()))
        constant = num_unique_rows == ngroup
        params['constant'] = constant
        if not constant:
            self.draw_group(data, panel_params, coord, ax, **params)
        else:
            for (_, gdata) in data.groupby('group'):
                gdata.reset_index(inplace=True, drop=True)
                self.draw_group(gdata, panel_params, coord, ax, **params)

    @staticmethod
    def draw_group(data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            return 10
        data = coord.transform(data, panel_params, munch=True)
        data['size'] *= SIZE_FACTOR
        if 'constant' in params:
            constant: bool = params.pop('constant')
        else:
            constant = len(np.unique(data['group'].to_numpy())) == 1
        if not constant:
            _draw_segments(data, ax, **params)
        else:
            _draw_lines(data, ax, **params)
        if 'arrow' in params and params['arrow']:
            params['arrow'].draw(data, panel_params, coord, ax, constant=constant, **params)

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: Layer) -> DrawingArea:
        if False:
            print('Hello World!')
        '\n        Draw a horizontal line in the box\n\n        Parameters\n        ----------\n        data : Series\n            Data Row\n        da : DrawingArea\n            Canvas\n        lyr : layer\n            Layer\n\n        Returns\n        -------\n        out : DrawingArea\n        '
        from matplotlib.lines import Line2D
        data['size'] *= SIZE_FACTOR
        x = [0, da.width]
        y = [0.5 * da.height] * 2
        color = to_rgba(data['color'], data['alpha'])
        key = Line2D(x, y, linestyle=data['linetype'], linewidth=data['size'], color=color, solid_capstyle='butt', antialiased=False)
        da.add_artist(key)
        return da

class arrow:
    """
    Define arrow (actually an arrowhead)

    This is used to define arrow heads for
    :class:`.geom_path`.

    Parameters
    ----------
    angle : int | float
        angle in degrees between the tail a
        single edge.
    length : int | float
        of the edge in "inches"
    ends : str in ``['last', 'first', 'both']``
        At which end of the line to draw the
        arrowhead
    type : str in ``['open', 'closed']``
        When it is closed, it is also filled
    """

    def __init__(self, angle: float=30, length: float=0.2, ends: Literal['first', 'last', 'both']='last', type: Literal['open', 'closed']='open'):
        if False:
            i = 10
            return i + 15
        self.angle = angle
        self.length = length
        self.ends = ends
        self.type = type

    def draw(self, data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, constant: bool=True, **params: Any):
        if False:
            return 10
        "\n        Draw arrows at the end(s) of the lines\n\n        Parameters\n        ----------\n        data : dataframe\n            Data to be plotted by this geom. This is the\n            dataframe created in the plot_build pipeline.\n        panel_params : panel_view\n            The scale information as may be required by the\n            axes. At this point, that information is about\n            ranges, ticks and labels. Attributes are of interest\n            to the geom are::\n\n                'panel_params.x.range'  # tuple\n                'panel_params.y.range'  # tuple\n\n        coord : coord\n            Coordinate (e.g. coord_cartesian) system of the\n            geom.\n        ax : axes\n            Axes on which to plot.\n        constant: bool\n            If the path attributes vary along the way. If false,\n            the arrows are per segment of the path\n        params : dict\n            Combined parameters for the geom and stat. Also\n            includes the 'zorder'.\n        "
        first = self.ends in ('first', 'both')
        last = self.ends in ('last', 'both')
        data = data.sort_values('group', kind='mergesort')
        data['color'] = to_rgba(data['color'], data['alpha'])
        if self.type == 'open':
            data['facecolor'] = 'none'
        else:
            data['facecolor'] = data['color']
        if not constant:
            from matplotlib.collections import PathCollection
            idx1: list[int] = []
            idx2: list[int] = []
            for (_, df) in data.groupby('group'):
                idx1.extend(df.index[:-1].to_list())
                idx2.extend(df.index[1:].to_list())
            d = {'zorder': params['zorder'], 'rasterized': params['raster'], 'edgecolor': data.loc[idx1, 'color'], 'facecolor': data.loc[idx1, 'facecolor'], 'linewidth': data.loc[idx1, 'size'], 'linestyle': data.loc[idx1, 'linetype']}
            x1 = data.loc[idx1, 'x'].to_numpy()
            y1 = data.loc[idx1, 'y'].to_numpy()
            x2 = data.loc[idx2, 'x'].to_numpy()
            y2 = data.loc[idx2, 'y'].to_numpy()
            if first:
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                coll = PathCollection(paths, **d)
                ax.add_collection(coll)
            if last:
                (x1, y1, x2, y2) = (x2, y2, x1, y1)
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                coll = PathCollection(paths, **d)
                ax.add_collection(coll)
        else:
            from matplotlib.patches import PathPatch
            d = {'zorder': params['zorder'], 'rasterized': params['raster'], 'edgecolor': data['color'].iloc[0], 'facecolor': data['facecolor'].iloc[0], 'linewidth': data['size'].iloc[0], 'linestyle': data['linetype'].iloc[0], 'joinstyle': 'round', 'capstyle': 'butt'}
            if first:
                (x1, x2) = data['x'].iloc[0:2]
                (y1, y2) = data['y'].iloc[0:2]
                (x1, y1, x2, y2) = (np.array([i]) for i in (x1, y1, x2, y2))
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                patch = PathPatch(paths[0], **d)
                ax.add_artist(patch)
            if last:
                (x1, x2) = data['x'].iloc[-2:]
                (y1, y2) = data['y'].iloc[-2:]
                (x1, y1, x2, y2) = (x2, y2, x1, y1)
                (x1, y1, x2, y2) = (np.array([i]) for i in (x1, y1, x2, y2))
                paths = self.get_paths(x1, y1, x2, y2, panel_params, coord, ax)
                patch = PathPatch(paths[0], **d)
                ax.add_artist(patch)

    def get_paths(self, x1: npt.ArrayLike, y1: npt.ArrayLike, x2: npt.ArrayLike, y2: npt.ArrayLike, panel_params: panel_view, coord: Coord, ax: Axes) -> list[Path]:
        if False:
            print('Hello World!')
        "\n        Compute paths that create the arrow heads\n\n        Parameters\n        ----------\n        x1, y1, x2, y2 : array_like\n            List of points that define the tails of the arrows.\n            The arrow heads will be at x1, y1. If you need them\n            at x2, y2 reverse the input.\n\n        panel_params : panel_view\n            The scale information as may be required by the\n            axes. At this point, that information is about\n            ranges, ticks and labels. Attributes are of interest\n            to the geom are::\n\n                'panel_params.x.range'  # tuple\n                'panel_params.y.range'  # tuple\n\n        coord : coord\n            Coordinate (e.g. coord_cartesian) system of the\n            geom.\n        ax : axes\n            Axes on which to plot.\n\n        Returns\n        -------\n        out : list of Path\n            Paths that create arrow heads\n        "
        from matplotlib.path import Path
        dummy = (0, 0)
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.STOP]
        (width, height) = _axes_get_size_inches(ax)
        ranges = coord.range(panel_params)
        width_ = np.ptp(ranges.x)
        height_ = np.ptp(ranges.y)
        lx = self.length * width_ / width
        ly = self.length * height_ / height
        a = self.angle * np.pi / 180
        (xdiff, ydiff) = (x2 - x1, y2 - y1)
        rotations = np.arctan2(ydiff / ly, xdiff / lx)
        v1x = x1 + lx * np.cos(rotations + a)
        v1y = y1 + ly * np.sin(rotations + a)
        v2x = x1 + lx * np.cos(rotations - a)
        v2y = y1 + ly * np.sin(rotations - a)
        paths = []
        for t in zip(v1x, v1y, x1, y1, v2x, v2y):
            verts = [t[:2], t[2:4], t[4:], dummy]
            paths.append(Path(verts, codes))
        return paths

def _draw_segments(data: pd.DataFrame, ax: Axes, **params: Any):
    if False:
        print('Hello World!')
    '\n    Draw independent line segments between all the\n    points\n    '
    from matplotlib.collections import LineCollection
    color = to_rgba(data['color'], data['alpha'])
    indices: list[int] = []
    _segments = []
    for (_, df) in data.groupby('group'):
        idx = df.index
        indices.extend(idx[:-1].to_list())
        x = data['x'].iloc[idx]
        y = data['y'].iloc[idx]
        _segments.append(make_line_segments(x, y, ispath=True))
    segments = np.vstack(_segments)
    if color is None:
        edgecolor = color
    else:
        edgecolor = [color[i] for i in indices]
    linewidth = data.loc[indices, 'size']
    linestyle = data.loc[indices, 'linetype']
    coll = LineCollection(segments, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, zorder=params['zorder'], rasterized=params['raster'])
    ax.add_collection(coll)

def _draw_lines(data: pd.DataFrame, ax: Axes, **params: Any):
    if False:
        while True:
            i = 10
    '\n    Draw a path with the same characteristics from the\n    first point to the last point\n    '
    from matplotlib.lines import Line2D
    color = to_rgba(data['color'].iloc[0], data['alpha'].iloc[0])
    join_style = _get_joinstyle(data, params)
    lines = Line2D(data['x'], data['y'], color=color, linewidth=data['size'].iloc[0], linestyle=data['linetype'].iloc[0], zorder=params['zorder'], rasterized=params['raster'], **join_style)
    ax.add_artist(lines)

def _get_joinstyle(data: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    with suppress(KeyError):
        if params['linejoin'] == 'mitre':
            params['linejoin'] = 'miter'
    with suppress(KeyError):
        if params['lineend'] == 'square':
            params['lineend'] = 'projecting'
    joinstyle = params.get('linejoin', 'miter')
    capstyle = params.get('lineend', 'butt')
    d = {}
    if data['linetype'].iloc[0] == 'solid':
        d['solid_joinstyle'] = joinstyle
        d['solid_capstyle'] = capstyle
    elif data['linetype'].iloc[0] == 'dashed':
        d['dash_joinstyle'] = joinstyle
        d['dash_capstyle'] = capstyle
    return d

def _axes_get_size_inches(ax: Axes) -> TupleFloat2:
    if False:
        for i in range(10):
            print('nop')
    '\n    Size of axes in inches\n\n    Parameters\n    ----------\n    ax : axes\n        Axes\n\n    Returns\n    -------\n    out : tuple[float, float]\n        (width, height) of ax in inches\n    '
    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return (bbox.width, bbox.height)