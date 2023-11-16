from __future__ import annotations
import typing
from copy import copy
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..utils import groupby_apply, pivot_apply
from .position_dodge import position_dodge
if typing.TYPE_CHECKING:
    from plotnine.typing import IntArray

class position_dodge2(position_dodge):
    """
    Dodge overlaps and place objects side-by-side

    This is an enhanced version of
    :class:`~plotnine.positions.position_dodge` that can deal
    with rectangular overlaps that do not share a lower x border.

    Parameters
    ----------
    width: float
        Dodging width, when different to the width of the
        individual elements. This is useful when you want
        to align narrow geoms with wider geoms
    preserve: str in ``['total', 'single']``
        Should dodging preserve the total width of all elements
        at a position, or the width of a single element?
    padding : float
        Padding between elements at the same position.
        Elements are shrunk by this proportion to allow space
        between them (Default: 0.1)
    reverse : bool
        Reverse the default ordering of the groups. This is
        useful if you're rotating both the plot and legend.
        (Default: False)
    """
    REQUIRED_AES = {'x'}

    def __init__(self, width=None, preserve='total', padding=0.1, reverse=False):
        if False:
            i = 10
            return i + 15
        self.params = {'width': width, 'preserve': preserve, 'padding': padding, 'reverse': reverse}

    def setup_params(self, data):
        if False:
            i = 10
            return i + 15
        if 'xmin' not in data and 'xmax' not in data and (self.params['width'] is None):
            msg = 'Width not defined. Set with `position_dodge2(width = ?)`'
            raise PlotnineError(msg)
        params = copy(self.params)
        if params['preserve'] == 'total':
            params['n'] = None
        elif 'x' in data:

            def max_x_values(gdf):
                if False:
                    for i in range(10):
                        print('nop')
                n = gdf['x'].value_counts().max()
                return pd.DataFrame({'n': [n]})
            res = groupby_apply(data, 'PANEL', max_x_values)
            params['n'] = res['n'].max()
        else:

            def _find_x_overlaps(gdf):
                if False:
                    i = 10
                    return i + 15
                return pd.DataFrame({'n': find_x_overlaps(gdf)})
            res = groupby_apply(data, 'PANEL', _find_x_overlaps)
            params['n'] = res['n'].max()
        return params

    @classmethod
    def compute_panel(cls, data, scales, params):
        if False:
            i = 10
            return i + 15
        return cls.collide2(data, params=params)

    @staticmethod
    def strategy(data, params):
        if False:
            return 10
        padding = params['padding']
        n = params['n']
        if not all((col in data.columns for col in ['xmin', 'xmax'])):
            data['xmin'] = data['x']
            data['xmax'] = data['x']
        data['xid'] = find_x_overlaps(data)
        res1 = pivot_apply(data, 'xmin', 'xid', np.min)
        res2 = pivot_apply(data, 'xmax', 'xid', np.max)
        data['newx'] = (res1 + res2)[data['xid'].to_numpy()].to_numpy() / 2
        if n is None:
            n = data['xid'].value_counts().to_numpy()
            n = n[data.loc[:, 'xid'] - 1]
            data['new_width'] = (data['xmax'] - data['xmin']) / n
        else:
            data['new_width'] = (data['xmax'] - data['xmin']) / n

        def sum_new_width(gdf):
            if False:
                i = 10
                return i + 15
            return pd.DataFrame({'size': [gdf['new_width'].sum()], 'newx': gdf['newx'].iloc[0]})
        group_sizes = groupby_apply(data, 'newx', sum_new_width)
        starts = group_sizes['newx'] - group_sizes['size'] / 2
        for (i, start) in enumerate(starts, start=1):
            bool_idx = data['xid'] == i
            divisions = np.cumsum(np.hstack([start, data.loc[bool_idx, 'new_width']]))
            data.loc[bool_idx, 'xmin'] = divisions[:-1]
            data.loc[bool_idx, 'xmax'] = divisions[1:]
        data['x'] = (data['xmin'] + data['xmax']) / 2
        if data['xid'].duplicated().any():
            pad_width = data['new_width'] * (1 - padding)
            data['xmin'] = data['x'] - pad_width / 2
            data['xmax'] = data['x'] + pad_width / 2
        data = data.drop(columns=['xid', 'newx', 'new_width'], errors='ignore')
        return data

def find_x_overlaps(df: pd.DataFrame) -> IntArray:
    if False:
        print('Hello World!')
    '\n    Find overlapping regions along the x axis\n    '
    n = len(df)
    overlaps = np.zeros(n, dtype=int)
    overlaps[0] = 1
    counter = 1
    for i in range(1, n):
        if df['xmin'].iloc[i] >= df['xmax'].iloc[i - 1]:
            counter += 1
        overlaps[i] = counter
    return overlaps