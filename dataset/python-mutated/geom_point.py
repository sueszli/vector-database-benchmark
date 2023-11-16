from __future__ import annotations
import typing
import numpy as np
from ..doctools import document
from ..scales.scale_shape import FILLED_SHAPES
from ..utils import SIZE_FACTOR, to_rgba
from .geom import geom
if typing.TYPE_CHECKING:
    from typing import Any
    import pandas as pd
    from plotnine.iapi import panel_view
    from plotnine.typing import Axes, Coord, DrawingArea, Layer

@document
class geom_point(geom):
    """
    Plot points (Scatter plot)

    {usage}

    Parameters
    ----------
    {common_parameters}
    """
    DEFAULT_AES = {'alpha': 1, 'color': 'black', 'fill': None, 'shape': 'o', 'size': 1.5, 'stroke': 0.5}
    REQUIRED_AES = {'x', 'y'}
    NON_MISSING_AES = {'color', 'shape', 'size'}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False}

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            print('Hello World!')
        '\n        Plot all groups\n        '
        self.draw_group(data, panel_params, coord, ax, **params)

    @staticmethod
    def draw_group(data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            print('Hello World!')
        data = coord.transform(data, panel_params)
        units = 'shape'
        for (_, udata) in data.groupby(units, dropna=False):
            udata.reset_index(inplace=True, drop=True)
            geom_point.draw_unit(udata, panel_params, coord, ax, **params)

    @staticmethod
    def draw_unit(data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            print('Hello World!')
        size = (data['size'] + data['stroke']) ** 2 * np.pi
        stroke = data['stroke'] * SIZE_FACTOR
        color = to_rgba(data['color'], data['alpha'])
        shape = data.loc[0, 'shape']
        if shape in FILLED_SHAPES:
            if all((c is None for c in data['fill'])):
                fill = color
            else:
                fill = to_rgba(data['fill'], data['alpha'])
        else:
            fill = color
            color = None
        ax.scatter(x=data['x'], y=data['y'], s=size, facecolor=fill, edgecolor=color, linewidth=stroke, marker=shape, zorder=params['zorder'], rasterized=params['raster'])

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: Layer) -> DrawingArea:
        if False:
            return 10
        '\n        Draw a point in the box\n\n        Parameters\n        ----------\n        data : Series\n            Data Row\n        da : DrawingArea\n            Canvas\n        lyr : layer\n            Layer\n\n        Returns\n        -------\n        out : DrawingArea\n        '
        from matplotlib.lines import Line2D
        if data['fill'] is None:
            data['fill'] = data['color']
        size = (data['size'] + data['stroke']) * SIZE_FACTOR
        stroke = data['stroke'] * SIZE_FACTOR
        fill = to_rgba(data['fill'], data['alpha'])
        color = to_rgba(data['color'], data['alpha'])
        key = Line2D([0.5 * da.width], [0.5 * da.height], marker=data['shape'], markersize=size, markerfacecolor=fill, markeredgecolor=color, markeredgewidth=stroke)
        da.add_artist(key)
        return da