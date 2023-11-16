from __future__ import annotations
from typing import Dict, Union
from .. import optional_features
from ..element import Element
try:
    import plotly.graph_objects as go
    optional_features.register('plotly')
except ImportError:
    pass

class Plotly(Element, component='plotly.vue', libraries=['lib/plotly/plotly.min.js']):

    def __init__(self, figure: Union[Dict, go.Figure]) -> None:
        if False:
            while True:
                i = 10
        'Plotly Element\n\n        Renders a Plotly chart.\n        There are two ways to pass a Plotly figure for rendering, see parameter `figure`:\n\n        * Pass a `go.Figure` object, see https://plotly.com/python/\n\n        * Pass a Python `dict` object with keys `data`, `layout`, `config` (optional), see https://plotly.com/javascript/\n\n        For best performance, use the declarative `dict` approach for creating a Plotly chart.\n\n        :param figure: Plotly figure to be rendered. Can be either a `go.Figure` instance, or\n                       a `dict` object with keys `data`, `layout`, `config` (optional).\n        '
        if not optional_features.has('plotly'):
            raise ImportError('Plotly is not installed. Please run "pip install nicegui[plotly]".')
        super().__init__()
        self.figure = figure
        self.update()

    def update_figure(self, figure: Union[Dict, go.Figure]):
        if False:
            print('Hello World!')
        'Overrides figure instance of this Plotly chart and updates chart on client side.'
        self.figure = figure
        self.update()

    def update(self) -> None:
        if False:
            return 10
        self._props['options'] = self._get_figure_json()
        super().update()

    def _get_figure_json(self) -> Dict:
        if False:
            i = 10
            return i + 15
        if isinstance(self.figure, go.Figure):
            return self.figure.to_plotly_json()
        if isinstance(self.figure, dict):
            return self.figure
        raise ValueError(f'Plotly figure is of unknown type "{self.figure.__class__.__name__}".')