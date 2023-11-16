"""A Python wrapper around Bokeh."""
import hashlib
import json
from typing import TYPE_CHECKING, cast
from typing_extensions import Final
from streamlit.errors import StreamlitAPIException
from streamlit.proto.BokehChart_pb2 import BokehChart as BokehChartProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
if TYPE_CHECKING:
    from bokeh.plotting.figure import Figure
    from streamlit.delta_generator import DeltaGenerator
ST_BOKEH_VERSION: Final = '2.4.3'

class BokehMixin:

    @gather_metrics('bokeh_chart')
    def bokeh_chart(self, figure: 'Figure', use_container_width: bool=False) -> 'DeltaGenerator':
        if False:
            print('Hello World!')
        "Display an interactive Bokeh chart.\n\n        Bokeh is a charting library for Python. The arguments to this function\n        closely follow the ones for Bokeh's `show` function. You can find\n        more about Bokeh at https://bokeh.pydata.org.\n\n        To show Bokeh charts in Streamlit, call `st.bokeh_chart`\n        wherever you would call Bokeh's `show`.\n\n        Parameters\n        ----------\n        figure : bokeh.plotting.figure.Figure\n            A Bokeh figure to plot.\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over Bokeh's native `width` value.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>> from bokeh.plotting import figure\n        >>>\n        >>> x = [1, 2, 3, 4, 5]\n        >>> y = [6, 7, 2, 4, 5]\n        >>>\n        >>> p = figure(\n        ...     title='simple line example',\n        ...     x_axis_label='x',\n        ...     y_axis_label='y')\n        ...\n        >>> p.line(x, y, legend_label='Trend', line_width=2)\n        >>>\n        >>> st.bokeh_chart(p, use_container_width=True)\n\n        .. output::\n           https://doc-bokeh-chart.streamlit.app/\n           height: 700px\n\n        "
        import bokeh
        if bokeh.__version__ != ST_BOKEH_VERSION:
            raise StreamlitAPIException(f'Streamlit only supports Bokeh version {ST_BOKEH_VERSION}, but you have version {bokeh.__version__} installed. Please run `pip install --force-reinstall --no-deps bokeh=={ST_BOKEH_VERSION}` to install the correct version.')
        delta_path = self.dg._get_delta_path_str()
        element_id = hashlib.md5(delta_path.encode(), **HASHLIB_KWARGS).hexdigest()
        bokeh_chart_proto = BokehChartProto()
        marshall(bokeh_chart_proto, figure, use_container_width, element_id)
        return self.dg._enqueue('bokeh_chart', bokeh_chart_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            print('Hello World!')
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

def marshall(proto: BokehChartProto, figure: 'Figure', use_container_width: bool, element_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Construct a Bokeh chart object.\n\n    See DeltaGenerator.bokeh_chart for docs.\n    '
    from bokeh.embed import json_item
    data = json_item(figure)
    proto.figure = json.dumps(data)
    proto.use_container_width = use_container_width
    proto.element_id = element_id