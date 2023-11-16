from __future__ import annotations
import typing
from ..doctools import document
from .geom import geom
if typing.TYPE_CHECKING:
    from typing import Any
    import pandas as pd
    from plotnine.iapi import panel_view
    from plotnine.typing import Axes, Coord

@document
class geom_blank(geom):
    """
    An empty plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    """
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False}

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            while True:
                i = 10
        pass

    def handle_na(self, data: pd.DataFrame) -> pd.DataFrame:
        if False:
            return 10
        return data