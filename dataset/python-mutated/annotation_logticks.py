from __future__ import annotations
import typing
import warnings
import numpy as np
import pandas as pd
from ..coords import coord_flip
from ..exceptions import PlotnineWarning
from ..scales.scale_continuous import scale_continuous as ScaleContinuous
from ..utils import log
from .annotate import annotate
from .geom_path import geom_path
from .geom_rug import geom_rug
if typing.TYPE_CHECKING:
    from typing import Any, Literal, Optional, Sequence
    from typing_extensions import TypeGuard
    from plotnine.iapi import panel_view
    from plotnine.typing import AnyArray, Axes, Coord, Geom, Layout, Scale, Trans, TupleFloat2, TupleFloat3

class _geom_logticks(geom_rug):
    """
    Internal geom implementing drawing of annotation_logticks
    """
    DEFAULT_AES = {}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False, 'sides': 'bl', 'alpha': 1, 'color': 'black', 'size': 0.5, 'linetype': 'solid', 'lengths': (0.036, 0.0225, 0.012), 'base': 10}
    draw_legend = staticmethod(geom_path.draw_legend)

    def draw_layer(self, data: pd.DataFrame, layout: Layout, coord: Coord, **params: Any):
        if False:
            print('Hello World!')
        '\n        Draw ticks on every panel\n        '
        for pid in layout.layout['PANEL']:
            ploc = pid - 1
            panel_params = layout.panel_params[ploc]
            ax = layout.axs[ploc]
            self.draw_panel(data, panel_params, coord, ax, **params)

    @staticmethod
    def _check_log_scale(base: Optional[float], sides: str, panel_params: panel_view, coord: Coord) -> TupleFloat2:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check the log transforms\n\n        Parameters\n        ----------\n        base : float or None\n            Base of the logarithm in which the ticks will be\n            calculated. If ``None``, the base of the log transform\n            the scale will be used.\n        sides : str (default: bl)\n            Sides onto which to draw the marks. Any combination\n            chosen from the characters ``btlr``, for *bottom*, *top*,\n            *left* or *right* side marks. If ``coord_flip()`` is used,\n            these are the sides *before* the flip.\n        panel_params : panel_view\n            ``x`` and ``y`` view scale values.\n        coord : coord\n            Coordinate (e.g. coord_cartesian) system of the geom.\n\n        Returns\n        -------\n        out : tuple\n            The bases (base_x, base_y) to use when generating the ticks.\n        '

        def is_log_trans(t: Trans) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return hasattr(t, 'base') and t.__class__.__name__.startswith('log')

        def get_base(sc, ubase: Optional[float]) -> float:
            if False:
                while True:
                    i = 10
            ae = sc.aesthetics[0]
            if not isinstance(sc, ScaleContinuous) or not is_log_trans(sc.trans):
                warnings.warn(f'annotation_logticks for {ae}-axis which does not have a log scale. The logticks may not make sense.', PlotnineWarning)
                return 10 if ubase is None else ubase
            base = sc.trans.base
            if ubase is not None and base != ubase:
                warnings.warn(f'The x-axis is log transformed in base={base} ,but the annotation_logticks are computed in base={ubase}', PlotnineWarning)
                return ubase
            return base
        (base_x, base_y) = (10, 10)
        x_scale = panel_params.x.scale
        y_scale = panel_params.y.scale
        if isinstance(coord, coord_flip):
            (x_scale, y_scale) = (y_scale, x_scale)
            (base_x, base_y) = (base_y, base_x)
        if 't' in sides or 'b' in sides:
            base_x = get_base(x_scale, base)
        if 'l' in sides or 'r' in sides:
            base_y = get_base(y_scale, base)
        return (base_x, base_y)

    @staticmethod
    def _calc_ticks(value_range: TupleFloat2, base: float) -> tuple[AnyArray, AnyArray, AnyArray]:
        if False:
            return 10
        '\n        Calculate tick marks within a range\n\n        Parameters\n        ----------\n        value_range: tuple\n            Range for which to calculate ticks.\n\n        base : number\n            Base of logarithm\n\n        Returns\n        -------\n        out: tuple\n            (major, middle, minor) tick locations\n        '

        def _minor(x: Sequence[Any], mid_idx: int) -> AnyArray:
            if False:
                for i in range(10):
                    print('nop')
            return np.hstack([x[1:mid_idx], x[mid_idx + 1:-1]])
        low = np.floor(value_range[0])
        high = np.ceil(value_range[1])
        arr = base ** np.arange(low, float(high + 1))
        n_ticks = int(np.round(base) - 1)
        breaks = [log(np.linspace(b1, b2, n_ticks + 1), base) for (b1, b2) in list(zip(arr, arr[1:]))]
        major = np.array([x[0] for x in breaks] + [breaks[-1][-1]])
        if n_ticks % 2:
            mid_idx = n_ticks // 2
            middle = np.array([x[mid_idx] for x in breaks])
            minor = np.hstack([_minor(x, mid_idx) for x in breaks])
        else:
            middle = np.array([])
            minor = np.hstack([x[1:-1] for x in breaks])
        return (major, middle, minor)

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            for i in range(10):
                print('nop')
        sides = params['sides']
        lengths = params['lengths']
        _aesthetics = {'size': params['size'], 'color': params['color'], 'alpha': params['alpha'], 'linetype': params['linetype']}

        def _draw(geom: Geom, axis: Literal['x', 'y'], tick_positions: tuple[AnyArray, AnyArray, AnyArray]):
            if False:
                i = 10
                return i + 15
            for (position, length) in zip(tick_positions, lengths):
                data = pd.DataFrame({axis: position, **_aesthetics})
                geom.draw_group(data, panel_params, coord, ax, length=length, **params)
        if isinstance(coord, coord_flip):
            tick_range_x = panel_params.y.range
            tick_range_y = panel_params.x.range
        else:
            tick_range_x = panel_params.x.range
            tick_range_y = panel_params.y.range
        (base_x, base_y) = self._check_log_scale(params['base'], sides, panel_params, coord)
        if 'b' in sides or 't' in sides:
            tick_positions = self._calc_ticks(tick_range_x, base_x)
            _draw(self, 'x', tick_positions)
        if 'l' in sides or 'r' in sides:
            tick_positions = self._calc_ticks(tick_range_y, base_y)
            _draw(self, 'y', tick_positions)

class annotation_logticks(annotate):
    """
    Marginal log ticks.

    If added to a plot that does not have a log10 axis
    on the respective side, a warning will be issued.

    Parameters
    ----------
    sides : str (default: bl)
        Sides onto which to draw the marks. Any combination
        chosen from the characters ``btlr``, for *bottom*, *top*,
        *left* or *right* side marks. If ``coord_flip()`` is used,
        these are the sides *after* the flip.
    alpha : float (default: 1)
        Transparency of the ticks
    color : str | tuple (default: 'black')
        Colour of the ticks
    size : float
        Thickness of the ticks
    linetype : 'solid' | 'dashed' | 'dashdot' | 'dotted' | tuple
        Type of line. Default is *solid*.
    lengths: tuple (default (0.036, 0.0225, 0.012))
        length of the ticks drawn for full / half / tenth
        ticks relative to panel size
    base : float (default: None)
        Base of the logarithm in which the ticks will be
        calculated. If ``None``, the base used to log transform
        the scale will be used.
    """

    def __init__(self, sides: str='bl', alpha: float=1, color: str | tuple[float, ...]='black', size: float=0.5, linetype: str | tuple[float, ...]='solid', lengths: TupleFloat3=(0.036, 0.0225, 0.012), base: float | None=None):
        if False:
            for i in range(10):
                print('nop')
        if len(lengths) != 3:
            raise ValueError('length for annotation_logticks must be a tuple of 3 floats')
        self._annotation_geom = _geom_logticks(sides=sides, alpha=alpha, color=color, size=size, linetype=linetype, lengths=lengths, base=base)

def is_continuous_scale(sc: Scale) -> TypeGuard[ScaleContinuous]:
    if False:
        while True:
            i = 10
    return isinstance(sc, ScaleContinuous)