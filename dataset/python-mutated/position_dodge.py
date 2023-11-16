from contextlib import suppress
from copy import copy
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..utils import groupby_apply, match
from .position import position

class position_dodge(position):
    """
    Dodge overlaps and place objects side-by-side

    Parameters
    ----------
    width: float
        Dodging width, when different to the width of the
        individual elements. This is useful when you want
        to align narrow geoms with wider geoms
    preserve: str in ``['total', 'single']``
        Should dodging preserve the total width of all elements
        at a position, or the width of a single element?
    """
    REQUIRED_AES = {'x'}

    def __init__(self, width=None, preserve='total'):
        if False:
            print('Hello World!')
        self.params = {'width': width, 'preserve': preserve}

    def setup_data(self, data, params):
        if False:
            print('Hello World!')
        has_xmin_xmax = 'xmin' in data and 'xmax' in data
        if 'x' not in data and has_xmin_xmax:
            data['x'] = (data['xmin'] + data['xmax']) / 2
        return super().setup_data(data, params)

    def setup_params(self, data):
        if False:
            print('Hello World!')
        if 'xmin' not in data and 'xmax' not in data and (self.params['width'] is None):
            msg = 'Width not defined. Set with `position_dodge(width = ?)`'
            raise PlotnineError(msg)
        params = copy(self.params)
        if params['preserve'] == 'total':
            params['n'] = None
        else:

            def max_xmin_values(gdf):
                if False:
                    print('Hello World!')
                try:
                    n = gdf['xmin'].value_counts().max()
                except KeyError:
                    n = gdf['x'].value_counts().max()
                return pd.DataFrame({'n': [n]})
            res = groupby_apply(data, 'PANEL', max_xmin_values)
            params['n'] = res['n'].max()
        return params

    @classmethod
    def compute_panel(cls, data, scales, params):
        if False:
            print('Hello World!')
        return cls.collide(data, params=params)

    @staticmethod
    def strategy(data, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dodge overlapping interval\n\n        Assumes that each set has the same horizontal position.\n        '
        width = params['width']
        with suppress(TypeError):
            iter(width)
            width = np.asarray(width)
            width = width[data.index]
        udata_group = data['group'].drop_duplicates()
        n = params.get('n', None)
        if n is None:
            n = len(udata_group)
        if n == 1:
            return data
        if not all((col in data.columns for col in ['xmin', 'xmax'])):
            data['xmin'] = data['x']
            data['xmax'] = data['x']
        d_width = np.max(data['xmax'] - data['xmin'])
        udata_group = udata_group.sort_values()
        groupidx = match(data['group'], udata_group)
        groupidx = np.asarray(groupidx) + 1
        data['x'] = data['x'] + width * ((groupidx - 0.5) / n - 0.5)
        data['xmin'] = data['x'] - d_width / n / 2
        data['xmax'] = data['x'] + d_width / n / 2
        return data