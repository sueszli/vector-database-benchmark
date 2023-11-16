from __future__ import annotations
import typing
import numpy as np
from ..doctools import document
from ..utils import SIZE_FACTOR, to_rgba
from .geom import geom
if typing.TYPE_CHECKING:
    from typing import Any
    import pandas as pd
    from plotnine.iapi import panel_view
    from plotnine.typing import Axes, Coord, DrawingArea, Layer

@document
class geom_polygon(geom):
    """
    Polygon, a filled path

    {usage}

    Parameters
    ----------
    {common_parameters}

    Notes
    -----
    All paths in the same ``group`` aesthetic value make up a polygon.
    """
    DEFAULT_AES = {'alpha': 1, 'color': None, 'fill': '#333333', 'linetype': 'solid', 'size': 0.5}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False}
    REQUIRED_AES = {'x', 'y'}

    def handle_na(self, data: pd.DataFrame) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            i = 10
            return i + 15
        '\n        Plot all groups\n        '
        self.draw_group(data, panel_params, coord, ax, **params)

    @staticmethod
    def draw_group(data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            for i in range(10):
                print('nop')
        from matplotlib.collections import PolyCollection
        data = coord.transform(data, panel_params, munch=True)
        data['size'] *= SIZE_FACTOR
        verts = []
        facecolor = []
        edgecolor = []
        linestyle = []
        linewidth = []
        grouper = data.groupby('group', sort=False)
        for (group, df) in grouper:
            fill = to_rgba(df['fill'].iloc[0], df['alpha'].iloc[0])
            verts.append(tuple(zip(df['x'], df['y'])))
            facecolor.append('none' if fill is None else fill)
            edgecolor.append(df['color'].iloc[0] or 'none')
            linestyle.append(df['linetype'].iloc[0])
            linewidth.append(df['size'].iloc[0])
        col = PolyCollection(verts, facecolors=facecolor, edgecolors=edgecolor, linestyles=linestyle, linewidths=linewidth, zorder=params['zorder'], rasterized=params['raster'])
        ax.add_collection(col)

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: Layer) -> DrawingArea:
        if False:
            while True:
                i = 10
        '\n        Draw a rectangle in the box\n\n        Parameters\n        ----------\n        data : Series\n            Data Row\n        da : DrawingArea\n            Canvas\n        lyr : layer\n            Layer\n\n        Returns\n        -------\n        out : DrawingArea\n        '
        from matplotlib.patches import Rectangle
        data['size'] *= SIZE_FACTOR
        linewidth = np.min([data['size'], da.width / 4, da.height / 4])
        if data['color'] is None:
            linewidth = 0
        facecolor = to_rgba(data['fill'], data['alpha'])
        if facecolor is None:
            facecolor = 'none'
        rect = Rectangle((0 + linewidth / 2, 0 + linewidth / 2), width=da.width - linewidth, height=da.height - linewidth, linewidth=linewidth, linestyle=data['linetype'], facecolor=facecolor, edgecolor=data['color'], capstyle='projecting')
        da.add_artist(rect)
        return da