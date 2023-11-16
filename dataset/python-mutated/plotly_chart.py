"""Streamlit support for Plotly charts."""
import json
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, List, Set, Union, cast
from typing_extensions import Final, Literal, TypeAlias
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.PlotlyChart_pb2 import PlotlyChart as PlotlyChartProto
from streamlit.runtime.legacy_caching import caching
from streamlit.runtime.metrics_util import gather_metrics
if TYPE_CHECKING:
    import matplotlib
    import plotly.graph_objs as go
    from plotly.basedatatypes import BaseFigure
    from streamlit.delta_generator import DeltaGenerator
try:
    import plotly.io as pio
    import streamlit.elements.lib.streamlit_plotly_theme
    pio.templates.default = 'streamlit'
except ModuleNotFoundError:
    pass
LOGGER: Final = get_logger(__name__)
SharingMode: TypeAlias = Literal['streamlit', 'private', 'public', 'secret']
SHARING_MODES: Set[SharingMode] = {'streamlit', 'private', 'public', 'secret'}
_AtomicFigureOrData: TypeAlias = Union['go.Figure', 'go.Data']
FigureOrData: TypeAlias = Union[_AtomicFigureOrData, List[_AtomicFigureOrData], Dict[str, _AtomicFigureOrData], 'BaseFigure', 'matplotlib.figure.Figure']

class PlotlyMixin:

    @gather_metrics('plotly_chart')
    def plotly_chart(self, figure_or_data: FigureOrData, use_container_width: bool=False, sharing: SharingMode='streamlit', theme: Union[None, Literal['streamlit']]='streamlit', **kwargs: Any) -> 'DeltaGenerator':
        if False:
            return 10
        'Display an interactive Plotly chart.\n\n        Plotly is a charting library for Python. The arguments to this function\n        closely follow the ones for Plotly\'s `plot()` function. You can find\n        more about Plotly at https://plot.ly/python.\n\n        To show Plotly charts in Streamlit, call `st.plotly_chart` wherever you\n        would call Plotly\'s `py.plot` or `py.iplot`.\n\n        Parameters\n        ----------\n        figure_or_data : plotly.graph_objs.Figure, plotly.graph_objs.Data,\n            dict/list of plotly.graph_objs.Figure/Data\n\n            See https://plot.ly/python/ for examples of graph descriptions.\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over the figure\'s native `width` value.\n\n        sharing : "streamlit", "private", "secret", or "public"\n            Use "streamlit" to insert the plot and all its dependencies\n            directly in the Streamlit app using plotly\'s offline mode (default).\n            Use any other sharing mode to send the chart to Plotly chart studio, which\n            requires an account. See https://plot.ly/python/chart-studio/ for more information.\n\n        theme : "streamlit" or None\n            The theme of the chart. Currently, we only support "streamlit" for the Streamlit\n            defined design or None to fallback to the default behavior of the library.\n\n        **kwargs\n            Any argument accepted by Plotly\'s `plot()` function.\n\n        Example\n        -------\n        The example below comes straight from the examples at\n        https://plot.ly/python:\n\n        >>> import streamlit as st\n        >>> import numpy as np\n        >>> import plotly.figure_factory as ff\n        >>>\n        >>> # Add histogram data\n        >>> x1 = np.random.randn(200) - 2\n        >>> x2 = np.random.randn(200)\n        >>> x3 = np.random.randn(200) + 2\n        >>>\n        >>> # Group data together\n        >>> hist_data = [x1, x2, x3]\n        >>>\n        >>> group_labels = [\'Group 1\', \'Group 2\', \'Group 3\']\n        >>>\n        >>> # Create distplot with custom bin_size\n        >>> fig = ff.create_distplot(\n        ...         hist_data, group_labels, bin_size=[.1, .25, .5])\n        >>>\n        >>> # Plot!\n        >>> st.plotly_chart(fig, use_container_width=True)\n\n        .. output::\n           https://doc-plotly-chart.streamlit.app/\n           height: 400px\n\n        '
        plotly_chart_proto = PlotlyChartProto()
        if theme != 'streamlit' and theme != None:
            raise StreamlitAPIException(f'You set theme="{theme}" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.')
        marshall(plotly_chart_proto, figure_or_data, use_container_width, sharing, theme, **kwargs)
        return self.dg._enqueue('plotly_chart', plotly_chart_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            print('Hello World!')
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

def marshall(proto: PlotlyChartProto, figure_or_data: FigureOrData, use_container_width: bool, sharing: SharingMode, theme: Union[None, Literal['streamlit']], **kwargs: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Marshall a proto with a Plotly spec.\n\n    See DeltaGenerator.plotly_chart for docs.\n    '
    import plotly.tools
    if type_util.is_type(figure_or_data, 'matplotlib.figure.Figure'):
        figure = plotly.tools.mpl_to_plotly(figure_or_data)
    else:
        figure = plotly.tools.return_figure_from_figure_or_data(figure_or_data, validate_figure=True)
    if not isinstance(sharing, str) or sharing.lower() not in SHARING_MODES:
        raise ValueError('Invalid sharing mode for Plotly chart: %s' % sharing)
    proto.use_container_width = use_container_width
    if sharing == 'streamlit':
        import plotly.utils
        config = dict(kwargs.get('config', {}))
        config.setdefault('showLink', kwargs.get('show_link', False))
        config.setdefault('linkText', kwargs.get('link_text', False))
        proto.figure.spec = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
        proto.figure.config = json.dumps(config)
    else:
        url = _plot_to_url_or_load_cached_url(figure, sharing=sharing, auto_open=False, **kwargs)
        proto.url = _get_embed_url(url)
    proto.theme = theme or ''

@caching.cache
def _plot_to_url_or_load_cached_url(*args: Any, **kwargs: Any) -> 'go.Figure':
    if False:
        i = 10
        return i + 15
    "Call plotly.plot wrapped in st.cache.\n\n    This is so we don't unnecessarily upload data to Plotly's SASS if nothing\n    changed since the previous upload.\n    "
    try:
        import chart_studio.plotly as ply
    except ImportError:
        import plotly.plotly as ply
    return ply.plot(*args, **kwargs)

def _get_embed_url(url: str) -> str:
    if False:
        i = 10
        return i + 15
    parsed_url = urllib.parse.urlparse(url)
    parsed_embed_url = parsed_url._replace(path=parsed_url.path + '.embed')
    return urllib.parse.urlunparse(parsed_embed_url)