from __future__ import annotations
import typing
from warnings import warn
import numpy as np
from ..doctools import document
from ..exceptions import PlotnineWarning
from ..utils import groupby_apply, resolution, to_rgba
from .geom import geom
if typing.TYPE_CHECKING:
    from typing import Any
    import pandas as pd
    from plotnine.iapi import panel_view
    from plotnine.typing import Axes, Coord, DrawingArea, FloatSeries, Layer

@document
class geom_dotplot(geom):
    """
    Dot plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    stackdir : str (default: up)
        Direction in which to stack the dots. Options are
        :py:`['up', 'down', 'center', 'centerwhole']`
    stackratio : float (default: 1)
        How close to stack the dots. If value is less than 1,
        the dots overlap, if greater than 1 they are spaced.
    dotsize : float (default: 1)
        Diameter of dots relative to ``binwidth``.
    stackgroups : bool (default: False)
        If :py:`True`, the dots are stacked across groups.

    See Also
    --------
    plotnine.stats.stat_bindot
    """
    DEFAULT_AES = {'alpha': 1, 'color': 'black', 'fill': 'black'}
    REQUIRED_AES = {'x', 'y'}
    NON_MISSING_AES = {'size', 'shape'}
    DEFAULT_PARAMS = {'stat': 'bindot', 'position': 'identity', 'na_rm': False, 'stackdir': 'up', 'stackratio': 1, 'dotsize': 1, 'stackgroups': False}

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        gp = self.params
        sp = self._stat.params
        if gp['position'] == 'stack':
            warn("position='stack' doesn't work properly with geom_dotplot. Use stackgroups=True instead.", PlotnineWarning)
        if gp['stackgroups'] and sp['method'] == 'dotdensity' and (sp['binpositions'] == 'bygroup'):
            warn("geom_dotplot called with stackgroups=TRUE and method='dotdensity'. You probably want to set binpositions='all'", PlotnineWarning)
        if 'width' not in data:
            if sp['width']:
                data['width'] = sp['width']
            else:
                data['width'] = resolution(data['x'], False) * 0.9
        if gp['stackdir'] in (None, 'up'):

            def stackdots(a: FloatSeries) -> FloatSeries:
                if False:
                    print('Hello World!')
                return a - 0.5
            stackaxismin: float = 0
            stackaxismax: float = 1
        elif gp['stackdir'] == 'down':

            def stackdots(a: FloatSeries) -> FloatSeries:
                if False:
                    for i in range(10):
                        print('nop')
                return -a + 0.5
            stackaxismin = -1
            stackaxismax = 0
        elif gp['stackdir'] == 'center':

            def stackdots(a: FloatSeries) -> FloatSeries:
                if False:
                    for i in range(10):
                        print('nop')
                return a - 1 - np.max(a - 1) / 2
            stackaxismin = -0.5
            stackaxismax = 0.5
        elif gp['stackdir'] == 'centerwhole':

            def stackdots(a: FloatSeries) -> FloatSeries:
                if False:
                    while True:
                        i = 10
                return a - 1 - np.floor(np.max(a - 1) / 2)
            stackaxismin = -0.5
            stackaxismax = 0.5
        else:
            raise ValueError(f"Invalid value stackdir={gp['stackdir']}")
        idx = [i for (i, c) in enumerate(data['count']) for j in range(int(c))]
        data = data.iloc[idx]
        data.reset_index(inplace=True, drop=True)
        groupvars = [sp['binaxis'], 'PANEL']
        if not gp['stackgroups']:
            groupvars.append('group')

        def func(df: pd.DataFrame) -> pd.DataFrame:
            if False:
                print('Hello World!')
            df['countidx'] = range(1, len(df) + 1)
            df['stackpos'] = stackdots(df['countidx'])
            return df
        data = groupby_apply(data, groupvars, func)
        if sp['binaxis'] == 'x':
            data['xmin'] = data['x'] - data['binwidth'] / 2
            data['xmax'] = data['x'] + data['binwidth'] / 2
            data['ymin'] = stackaxismin
            data['ymax'] = stackaxismax
            data['y'] = 0
        elif sp['binaxis'] == 'y':

            def func(df: pd.DataFrame) -> pd.DataFrame:
                if False:
                    while True:
                        i = 10
                df['ymin'] = df['y'].min() - data['binwidth'][0] / 2
                df['ymax'] = df['y'].max() + data['binwidth'][0] / 2
                return df
            data = groupby_apply(data, 'group', func)
            data['xmin'] = data['x'] + data['width'] * stackaxismin
            data['xmax'] = data['x'] + data['width'] * stackaxismax
        return data

    @staticmethod
    def draw_group(data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            i = 10
            return i + 15
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Ellipse
        data = coord.transform(data, panel_params)
        fill = to_rgba(data['fill'], data['alpha'])
        color = to_rgba(data['color'], data['alpha'])
        ranges = coord.range(panel_params)
        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        (ax_width, ax_height) = (bbox.width, bbox.height)
        factor = ax_width / ax_height * np.ptp(ranges.y) / np.ptp(ranges.x)
        size = data.loc[0, 'binwidth'] * params['dotsize']
        offsets = data['stackpos'] * params['stackratio']
        if params['binaxis'] == 'x':
            (width, height) = (size, size * factor)
            (xpos, ypos) = (data['x'], data['y'] + height * offsets)
        elif params['binaxis'] == 'y':
            (width, height) = (size / factor, size)
            (xpos, ypos) = (data['x'] + width * offsets, data['y'])
        else:
            raise ValueError(f"Invalid valid value binaxis={params['binaxis']}")
        circles = []
        for xy in zip(xpos, ypos):
            patch = Ellipse(xy, width=width, height=height)
            circles.append(patch)
        coll = PatchCollection(circles, edgecolors=color, facecolors=fill, rasterized=params['raster'])
        ax.add_collection(coll)

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: Layer) -> DrawingArea:
        if False:
            return 10
        '\n        Draw a point in the box\n\n        Parameters\n        ----------\n        data : Series\n            Data Row\n        da : DrawingArea\n            Canvas\n        lyr : layer\n            Layer\n\n        Returns\n        -------\n        out : DrawingArea\n        '
        from matplotlib.lines import Line2D
        fill = to_rgba(data['fill'], data['alpha'])
        key = Line2D([0.5 * da.width], [0.5 * da.height], marker='o', markersize=da.width / 2, markerfacecolor=fill, markeredgecolor=data['color'])
        da.add_artist(key)
        return da