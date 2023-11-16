"""Streamlit support for GraphViz charts."""
import hashlib
from typing import TYPE_CHECKING, Union, cast
from typing_extensions import Final, TypeAlias
from streamlit import type_util
from streamlit.errors import StreamlitAPIException
from streamlit.logger import get_logger
from streamlit.proto.GraphVizChart_pb2 import GraphVizChart as GraphVizChartProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.util import HASHLIB_KWARGS
if TYPE_CHECKING:
    import graphviz
    from streamlit.delta_generator import DeltaGenerator
LOGGER: Final = get_logger(__name__)
FigureOrDot: TypeAlias = Union['graphviz.Graph', 'graphviz.Digraph', str]

class GraphvizMixin:

    @gather_metrics('graphviz_chart')
    def graphviz_chart(self, figure_or_dot: FigureOrDot, use_container_width: bool=False) -> 'DeltaGenerator':
        if False:
            while True:
                i = 10
        "Display a graph using the dagre-d3 library.\n\n        Parameters\n        ----------\n        figure_or_dot : graphviz.dot.Graph, graphviz.dot.Digraph, str\n            The Graphlib graph object or dot string to display\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over the figure's native `width` value.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>> import graphviz\n        >>>\n        >>> # Create a graphlib graph object\n        >>> graph = graphviz.Digraph()\n        >>> graph.edge('run', 'intr')\n        >>> graph.edge('intr', 'runbl')\n        >>> graph.edge('runbl', 'run')\n        >>> graph.edge('run', 'kernel')\n        >>> graph.edge('kernel', 'zombie')\n        >>> graph.edge('kernel', 'sleep')\n        >>> graph.edge('kernel', 'runmem')\n        >>> graph.edge('sleep', 'swap')\n        >>> graph.edge('swap', 'runswap')\n        >>> graph.edge('runswap', 'new')\n        >>> graph.edge('runswap', 'runmem')\n        >>> graph.edge('new', 'runmem')\n        >>> graph.edge('sleep', 'runmem')\n        >>>\n        >>> st.graphviz_chart(graph)\n\n        Or you can render the chart from the graph using GraphViz's Dot\n        language:\n\n        >>> st.graphviz_chart('''\n            digraph {\n                run -> intr\n                intr -> runbl\n                runbl -> run\n                run -> kernel\n                kernel -> zombie\n                kernel -> sleep\n                kernel -> runmem\n                sleep -> swap\n                swap -> runswap\n                runswap -> new\n                runswap -> runmem\n                new -> runmem\n                sleep -> runmem\n            }\n        ''')\n\n        .. output::\n           https://doc-graphviz-chart.streamlit.app/\n           height: 600px\n\n        "
        delta_path = self.dg._get_delta_path_str()
        element_id = hashlib.md5(delta_path.encode(), **HASHLIB_KWARGS).hexdigest()
        graphviz_chart_proto = GraphVizChartProto()
        marshall(graphviz_chart_proto, figure_or_dot, use_container_width, element_id)
        return self.dg._enqueue('graphviz_chart', graphviz_chart_proto)

    @property
    def dg(self) -> 'DeltaGenerator':
        if False:
            return 10
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

def marshall(proto: GraphVizChartProto, figure_or_dot: FigureOrDot, use_container_width: bool, element_id: str) -> None:
    if False:
        print('Hello World!')
    'Construct a GraphViz chart object.\n\n    See DeltaGenerator.graphviz_chart for docs.\n    '
    if type_util.is_graphviz_chart(figure_or_dot):
        dot = figure_or_dot.source
        engine = figure_or_dot.engine
    elif isinstance(figure_or_dot, str):
        dot = figure_or_dot
        engine = 'dot'
    else:
        raise StreamlitAPIException('Unhandled type for graphviz chart: %s' % type(figure_or_dot))
    proto.spec = dot
    proto.engine = engine
    proto.use_container_width = use_container_width
    proto.element_id = element_id