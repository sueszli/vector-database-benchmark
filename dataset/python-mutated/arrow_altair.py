"""A Python wrapper around Altair.
Altair is a Python visualization library based on Vega-Lite,
a nice JSON schema for expressing graphs and charts.
"""
from __future__ import annotations
from contextlib import nullcontext
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional, Sequence, Tuple, Union, cast
import pandas as pd
from pandas.api.types import infer_dtype, is_integer_dtype
from typing_extensions import Literal
import streamlit.elements.arrow_vega_lite as arrow_vega_lite
from streamlit import type_util
from streamlit.color_util import Color, is_color_like, is_color_tuple_like, is_hex_color_like, to_css_color
from streamlit.elements.altair_utils import AddRowsMetadata
from streamlit.elements.arrow import Data
from streamlit.elements.utils import last_index_for_melted_dataframes
from streamlit.errors import Error, StreamlitAPIException
from streamlit.proto.ArrowVegaLiteChart_pb2 import ArrowVegaLiteChart as ArrowVegaLiteChartProto
from streamlit.runtime.metrics_util import gather_metrics
if TYPE_CHECKING:
    import altair as alt
    from streamlit.delta_generator import DeltaGenerator

class ChartType(Enum):
    AREA = {'mark_type': 'area'}
    BAR = {'mark_type': 'bar'}
    LINE = {'mark_type': 'line'}
    SCATTER = {'mark_type': 'circle'}
COLOR_LEGEND_SETTINGS = dict(titlePadding=5, offset=5, orient='bottom')
SIZE_LEGEND_SETTINGS = dict(titlePadding=0.5, offset=5, orient='bottom')
SEPARATED_INDEX_COLUMN_TITLE = 'index'
MELTED_Y_COLUMN_TITLE = 'value'
MELTED_COLOR_COLUMN_TITLE = 'color'
PROTECTION_SUFFIX = '--p5bJXXpQgvPz6yvQMFiy'
SEPARATED_INDEX_COLUMN_NAME = SEPARATED_INDEX_COLUMN_TITLE + PROTECTION_SUFFIX
MELTED_Y_COLUMN_NAME = MELTED_Y_COLUMN_TITLE + PROTECTION_SUFFIX
MELTED_COLOR_COLUMN_NAME = MELTED_COLOR_COLUMN_TITLE + PROTECTION_SUFFIX
NON_EXISTENT_COLUMN_NAME = 'DOES_NOT_EXIST' + PROTECTION_SUFFIX

class ArrowAltairMixin:

    @gather_metrics('line_chart')
    def line_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | List[Color] | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> DeltaGenerator:
        if False:
            print('Hello World!')
        'Display a line chart.\n\n        This is syntax-sugar around ``st.altair_chart``. The main difference\n        is this command uses the data\'s own column and indices to figure out\n        the chart\'s spec. As a result this is easier to use for many "just plot\n        this" scenarios, while being less customizable.\n\n        If ``st.line_chart`` does not guess the data specification\n        correctly, try specifying your desired chart using ``st.altair_chart``.\n\n        Parameters\n        ----------\n        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, dict or None\n            Data to be plotted.\n\n        x : str or None\n            Column name to use for the x-axis. If None, uses the data index for the x-axis.\n\n        y : str, Sequence of str, or None\n            Column name(s) to use for the y-axis. If a Sequence of strings,\n            draws several series on the same chart by melting your wide-format\n            table into a long-format table behind the scenes. If None, draws\n            the data of all remaining columns as data series.\n\n        color : str, tuple, Sequence of str, Sequence of tuple, or None\n            The color to use for different lines in this chart.\n\n            For a line chart with just one line, this can be:\n\n            * None, to use the default color.\n            * A hex string like "#ffaa00" or "#ffaa0088".\n            * An RGB or RGBA tuple with the red, green, blue, and alpha\n              components specified as ints from 0 to 255 or floats from 0.0 to\n              1.0.\n\n            For a line chart with multiple lines, where the dataframe is in\n            long format (that is, y is None or just one column), this can be:\n\n            * None, to use the default colors.\n            * The name of a column in the dataset. Data points will be grouped\n              into lines of the same color based on the value of this column.\n              In addition, if the values in this column match one of the color\n              formats above (hex string or color tuple), then that color will\n              be used.\n\n              For example: if the dataset has 1000 rows, but this column only\n              contains the values "adult", "child", and "baby", then those 1000\n              datapoints will be grouped into three lines whose colors will be\n              automatically selected from the default palette.\n\n              But, if for the same 1000-row dataset, this column contained\n              the values "#ffaa00", "#f0f", "#0000ff", then then those 1000\n              datapoints would still be grouped into three lines, but their\n              colors would be "#ffaa00", "#f0f", "#0000ff" this time around.\n\n            For a line chart with multiple lines, where the dataframe is in\n            wide format (that is, y is a Sequence of columns), this can be:\n\n            * None, to use the default colors.\n            * A list of string colors or color tuples to be used for each of\n              the lines in the chart. This list should have the same length\n              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``\n              for three lines).\n\n        width : int\n            The chart width in pixels. If 0, selects the width automatically.\n\n        height : int\n            The chart height in pixels. If 0, selects the height automatically.\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over the width argument.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])\n        >>>\n        >>> st.line_chart(chart_data)\n\n        .. output::\n           https://doc-line-chart.streamlit.app/\n           height: 440px\n\n        You can also choose different columns to use for x and y, as well as set\n        the color dynamically based on a 3rd column (assuming your dataframe is in\n        long format):\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(\n        ...    {\n        ...        "col1": np.random.randn(20),\n        ...        "col2": np.random.randn(20),\n        ...        "col3": np.random.choice(["A", "B", "C"], 20),\n        ...    }\n        ... )\n        >>>\n        >>> st.line_chart(chart_data, x="col1", y="col2", color="col3")\n\n        .. output::\n           https://doc-line-chart1.streamlit.app/\n           height: 440px\n\n        Finally, if your dataframe is in wide format, you can group multiple\n        columns under the y argument to show multiple lines with different\n        colors:\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])\n        >>>\n        >>> st.line_chart(\n        ...    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional\n        ... )\n\n        .. output::\n           https://doc-line-chart2.streamlit.app/\n           height: 440px\n\n        '
        proto = ArrowVegaLiteChartProto()
        (chart, add_rows_metadata) = _generate_chart(chart_type=ChartType.LINE, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=None, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_line_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('area_chart')
    def area_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | List[Color] | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> DeltaGenerator:
        if False:
            while True:
                i = 10
        'Display an area chart.\n\n        This is syntax-sugar around ``st.altair_chart``. The main difference\n        is this command uses the data\'s own column and indices to figure out\n        the chart\'s spec. As a result this is easier to use for many "just plot\n        this" scenarios, while being less customizable.\n\n        If ``st.area_chart`` does not guess the data specification\n        correctly, try specifying your desired chart using ``st.altair_chart``.\n\n        Parameters\n        ----------\n        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, or dict\n            Data to be plotted.\n\n        x : str or None\n            Column name to use for the x-axis. If None, uses the data index for the x-axis.\n\n        y : str, Sequence of str, or None\n            Column name(s) to use for the y-axis. If a Sequence of strings,\n            draws several series on the same chart by melting your wide-format\n            table into a long-format table behind the scenes. If None, draws\n            the data of all remaining columns as data series.\n\n        color : str, tuple, Sequence of str, Sequence of tuple, or None\n            The color to use for different series in this chart.\n\n            For an area chart with just 1 series, this can be:\n\n            * None, to use the default color.\n            * A hex string like "#ffaa00" or "#ffaa0088".\n            * An RGB or RGBA tuple with the red, green, blue, and alpha\n              components specified as ints from 0 to 255 or floats from 0.0 to\n              1.0.\n\n            For an area chart with multiple series, where the dataframe is in\n            long format (that is, y is None or just one column), this can be:\n\n            * None, to use the default colors.\n            * The name of a column in the dataset. Data points will be grouped\n              into series of the same color based on the value of this column.\n              In addition, if the values in this column match one of the color\n              formats above (hex string or color tuple), then that color will\n              be used.\n\n              For example: if the dataset has 1000 rows, but this column only\n              contains the values "adult", "child", and "baby", then those 1000\n              datapoints will be grouped into three series whose colors will be\n              automatically selected from the default palette.\n\n              But, if for the same 1000-row dataset, this column contained\n              the values "#ffaa00", "#f0f", "#0000ff", then then those 1000\n              datapoints would still be grouped into 3 series, but their\n              colors would be "#ffaa00", "#f0f", "#0000ff" this time around.\n\n            For an area chart with multiple series, where the dataframe is in\n            wide format (that is, y is a Sequence of columns), this can be:\n\n            * None, to use the default colors.\n            * A list of string colors or color tuples to be used for each of\n              the series in the chart. This list should have the same length\n              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``\n              for three lines).\n\n        width : int\n            The chart width in pixels. If 0, selects the width automatically.\n\n        height : int\n            The chart height in pixels. If 0, selects the height automatically.\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over the width argument.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])\n        >>>\n        >>> st.area_chart(chart_data)\n\n        .. output::\n           https://doc-area-chart.streamlit.app/\n           height: 440px\n\n        You can also choose different columns to use for x and y, as well as set\n        the color dynamically based on a 3rd column (assuming your dataframe is in\n        long format):\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(\n        ...    {\n        ...        "col1": np.random.randn(20),\n        ...        "col2": np.random.randn(20),\n        ...        "col3": np.random.choice(["A", "B", "C"], 20),\n        ...    }\n        ... )\n        >>>\n        >>> st.area_chart(chart_data, x="col1", y="col2", color="col3")\n\n        .. output::\n           https://doc-area-chart1.streamlit.app/\n           height: 440px\n\n        Finally, if your dataframe is in wide format, you can group multiple\n        columns under the y argument to show multiple series with different\n        colors:\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])\n        >>>\n        >>> st.area_chart(\n        ...    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional\n        ... )\n\n        .. output::\n           https://doc-area-chart2.streamlit.app/\n           height: 440px\n\n        '
        proto = ArrowVegaLiteChartProto()
        (chart, add_rows_metadata) = _generate_chart(chart_type=ChartType.AREA, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=None, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_area_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('bar_chart')
    def bar_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | List[Color] | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> DeltaGenerator:
        if False:
            return 10
        'Display a bar chart.\n\n        This is syntax-sugar around ``st.altair_chart``. The main difference\n        is this command uses the data\'s own column and indices to figure out\n        the chart\'s spec. As a result this is easier to use for many "just plot\n        this" scenarios, while being less customizable.\n\n        If ``st.bar_chart`` does not guess the data specification\n        correctly, try specifying your desired chart using ``st.altair_chart``.\n\n        Parameters\n        ----------\n        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, or dict\n            Data to be plotted.\n\n        x : str or None\n            Column name to use for the x-axis. If None, uses the data index for the x-axis.\n\n        y : str, Sequence of str, or None\n            Column name(s) to use for the y-axis. If a Sequence of strings,\n            draws several series on the same chart by melting your wide-format\n            table into a long-format table behind the scenes. If None, draws\n            the data of all remaining columns as data series.\n\n        color : str, tuple, Sequence of str, Sequence of tuple, or None\n            The color to use for different series in this chart.\n\n            For a bar chart with just one series, this can be:\n\n            * None, to use the default color.\n            * A hex string like "#ffaa00" or "#ffaa0088".\n            * An RGB or RGBA tuple with the red, green, blue, and alpha\n              components specified as ints from 0 to 255 or floats from 0.0 to\n              1.0.\n\n            For a bar chart with multiple series, where the dataframe is in\n            long format (that is, y is None or just one column), this can be:\n\n            * None, to use the default colors.\n            * The name of a column in the dataset. Data points will be grouped\n              into series of the same color based on the value of this column.\n              In addition, if the values in this column match one of the color\n              formats above (hex string or color tuple), then that color will\n              be used.\n\n              For example: if the dataset has 1000 rows, but this column only\n              contains the values "adult", "child", and "baby", then those 1000\n              datapoints will be grouped into three series whose colors will be\n              automatically selected from the default palette.\n\n              But, if for the same 1000-row dataset, this column contained\n              the values "#ffaa00", "#f0f", "#0000ff", then then those 1000\n              datapoints would still be grouped into 3 series, but their\n              colors would be "#ffaa00", "#f0f", "#0000ff" this time around.\n\n            For a bar chart with multiple series, where the dataframe is in\n            wide format (that is, y is a Sequence of columns), this can be:\n\n            * None, to use the default colors.\n            * A list of string colors or color tuples to be used for each of\n              the series in the chart. This list should have the same length\n              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``\n              for three lines).\n\n        width : int\n            The chart width in pixels. If 0, selects the width automatically.\n\n        height : int\n            The chart height in pixels. If 0, selects the height automatically.\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over the width argument.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])\n        >>>\n        >>> st.bar_chart(chart_data)\n\n        .. output::\n           https://doc-bar-chart.streamlit.app/\n           height: 440px\n\n        You can also choose different columns to use for x and y, as well as set\n        the color dynamically based on a 3rd column (assuming your dataframe is in\n        long format):\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(\n        ...    {\n        ...        "col1": list(range(20)) * 3,\n        ...        "col2": np.random.randn(60),\n        ...        "col3": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,\n        ...    }\n        ... )\n        >>>\n        >>> st.bar_chart(chart_data, x="col1", y="col2", color="col3")\n\n        .. output::\n           https://doc-bar-chart1.streamlit.app/\n           height: 440px\n\n        Finally, if your dataframe is in wide format, you can group multiple\n        columns under the y argument to show multiple series with different\n        colors:\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(\n        ...    {"col1": list(range(20)), "col2": np.random.randn(20), "col3": np.random.randn(20)}\n        ... )\n        >>>\n        >>> st.bar_chart(\n        ...    chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional\n        ... )\n\n        .. output::\n           https://doc-bar-chart2.streamlit.app/\n           height: 440px\n\n        '
        proto = ArrowVegaLiteChartProto()
        (chart, add_rows_metadata) = _generate_chart(chart_type=ChartType.BAR, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=None, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_bar_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('scatter_chart')
    def scatter_chart(self, data: Data=None, *, x: str | None=None, y: str | Sequence[str] | None=None, color: str | Color | List[Color] | None=None, size: str | float | int | None=None, width: int=0, height: int=0, use_container_width: bool=True) -> 'DeltaGenerator':
        if False:
            print('Hello World!')
        'Display a scatterplot chart.\n\n        This is syntax-sugar around ``st.altair_chart``. The main difference\n        is this command uses the data\'s own column and indices to figure out\n        the chart\'s spec. As a result this is easier to use for many "just plot\n        this" scenarios, while being less customizable.\n\n        If ``st.scatter_chart`` does not guess the data specification correctly,\n        try specifying your desired chart using ``st.altair_chart``.\n\n        Parameters\n        ----------\n        data : pandas.DataFrame, pandas.Styler, pyarrow.Table, numpy.ndarray, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table, Iterable, dict or None\n            Data to be plotted.\n\n        x : str or None\n            Column name to use for the x-axis. If None, uses the data index for the x-axis.\n\n        y : str, Sequence of str, or None\n            Column name(s) to use for the y-axis. If a Sequence of strings,\n            draws several series on the same chart by melting your wide-format\n            table into a long-format table behind the scenes. If None, draws\n            the data of all remaining columns as data series.\n\n        color : str, tuple, Sequence of str, Sequence of tuple, or None\n            The color of the circles representing each datapoint.\n\n            This can be:\n\n            * None, to use the default color.\n            * A hex string like "#ffaa00" or "#ffaa0088".\n            * An RGB or RGBA tuple with the red, green, blue, and alpha\n              components specified as ints from 0 to 255 or floats from 0.0 to\n              1.0.\n            * The name of a column in the dataset where the color of that\n              datapoint will come from.\n\n              If the values in this column are in one of the color formats\n              above (hex string or color tuple), then that color will be used.\n\n              Otherwise, the color will be automatically picked from the\n              default palette.\n\n              For example: if the dataset has 1000 rows, but this column only\n              contains the values "adult", "child", and "baby", then those 1000\n              datapoints be shown using three colors from the default palette.\n\n              But if this column only contains floats or ints, then those\n              1000 datapoints will be shown using a colors from a continuous\n              color gradient.\n\n              Finally, if this column only contains the values "#ffaa00",\n              "#f0f", "#0000ff", then then each of those 1000 datapoints will\n              be assigned "#ffaa00", "#f0f", or "#0000ff" as appropriate.\n\n            If the dataframe is in wide format (that is, y is a Sequence of\n            columns), this can also be:\n\n            * A list of string colors or color tuples to be used for each of\n              the series in the chart. This list should have the same length\n              as the number of y values (e.g. ``color=["#fd0", "#f0f", "#04f"]``\n              for three series).\n\n        size : str, float, int, or None\n            The size of the circles representing each point.\n\n            This can be:\n\n            * A number like 100, to specify a single size to use for all\n              datapoints.\n            * The name of the column to use for the size. This allows each\n              datapoint to be represented by a circle of a different size.\n\n        width : int\n            The chart width in pixels. If 0, selects the width automatically.\n\n        height : int\n            The chart height in pixels. If 0, selects the height automatically.\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over the width argument.\n\n        Examples\n        --------\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])\n        >>>\n        >>> st.scatter_chart(chart_data)\n\n        .. output::\n           https://doc-scatter-chart.streamlit.app/\n           height: 440px\n\n        You can also choose different columns to use for x and y, as well as set\n        the color dynamically based on a 3rd column (assuming your dataframe is in\n        long format):\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])\n        >>> chart_data[\'col4\'] = np.random.choice([\'A\',\'B\',\'C\'], 20)\n        >>>\n        >>> st.scatter_chart(\n        ...     chart_data,\n        ...     x=\'col1\',\n        ...     y=\'col2\',\n        ...     color=\'col4\',\n        ...     size=\'col3\',\n        ... )\n\n        .. output::\n           https://doc-scatter-chart1.streamlit.app/\n           height: 440px\n\n        Finally, if your dataframe is in wide format, you can group multiple\n        columns under the y argument to show multiple series with different\n        colors:\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 4), columns=["col1", "col2", "col3", "col4"])\n        >>>\n        >>> st.scatter_chart(\n        ...     chart_data,\n        ...     x=\'col1\',\n        ...     y=[\'col2\', \'col3\'],\n        ...     size=\'col4\',\n        ...     color=[\'#FF0000\', \'#0000FF\'],  # Optional\n        ... )\n\n        .. output::\n           https://doc-scatter-chart2.streamlit.app/\n           height: 440px\n\n        '
        proto = ArrowVegaLiteChartProto()
        (chart, add_rows_metadata) = _generate_chart(chart_type=ChartType.SCATTER, data=data, x_from_user=x, y_from_user=y, color_from_user=color, size_from_user=size, width=width, height=height)
        marshall(proto, chart, use_container_width, theme='streamlit')
        return self.dg._enqueue('arrow_scatter_chart', proto, add_rows_metadata=add_rows_metadata)

    @gather_metrics('altair_chart')
    def altair_chart(self, altair_chart: alt.Chart, use_container_width: bool=False, theme: Literal['streamlit'] | None='streamlit') -> DeltaGenerator:
        if False:
            return 10
        'Display a chart using the Altair library.\n\n        Parameters\n        ----------\n        altair_chart : altair.Chart\n            The Altair chart object to display.\n\n        use_container_width : bool\n            If True, set the chart width to the column width. This takes\n            precedence over Altair\'s native ``width`` value.\n\n        theme : "streamlit" or None\n            The theme of the chart. Currently, we only support "streamlit" for the Streamlit\n            defined design or None to fallback to the default behavior of the library.\n\n        Example\n        -------\n\n        >>> import streamlit as st\n        >>> import pandas as pd\n        >>> import numpy as np\n        >>> import altair as alt\n        >>>\n        >>> chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])\n        >>>\n        >>> c = (\n        ...    alt.Chart(chart_data)\n        ...    .mark_circle()\n        ...    .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])\n        ... )\n        >>>\n        >>> st.altair_chart(c, use_container_width=True)\n\n        .. output::\n           https://doc-vega-lite-chart.streamlit.app/\n           height: 300px\n\n        Examples of Altair charts can be found at\n        https://altair-viz.github.io/gallery/.\n\n        '
        if theme != 'streamlit' and theme != None:
            raise StreamlitAPIException(f'You set theme="{theme}" while Streamlit charts only support theme=”streamlit” or theme=None to fallback to the default library theme.')
        proto = ArrowVegaLiteChartProto()
        marshall(proto, altair_chart, use_container_width=use_container_width, theme=theme)
        return self.dg._enqueue('arrow_vega_lite_chart', proto)

    @property
    def dg(self) -> DeltaGenerator:
        if False:
            print('Hello World!')
        'Get our DeltaGenerator.'
        return cast('DeltaGenerator', self)

def _is_date_column(df: pd.DataFrame, name: Optional[str]) -> bool:
    if False:
        i = 10
        return i + 15
    "True if the column with the given name stores datetime.date values.\n\n    This function just checks the first value in the given column, so\n    it's meaningful only for columns whose values all share the same type.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n    name : str\n        The column name\n\n    Returns\n    -------\n    bool\n\n    "
    if name is None:
        return False
    column = df[name]
    if column.size == 0:
        return False
    return isinstance(column.iloc[0], date)

def _melt_data(df: pd.DataFrame, columns_to_leave_alone: List[str], columns_to_melt: Optional[List[str]], new_y_column_name: str, new_color_column_name: str) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    'Converts a wide-format dataframe to a long-format dataframe.'
    melted_df = pd.melt(df, id_vars=columns_to_leave_alone, value_vars=columns_to_melt, var_name=new_color_column_name, value_name=new_y_column_name)
    y_series = melted_df[new_y_column_name]
    if y_series.dtype == 'object' and 'mixed' in infer_dtype(y_series) and (len(y_series.unique()) > 100):
        raise StreamlitAPIException('The columns used for rendering the chart contain too many values with mixed types. Please select the columns manually via the y parameter.')
    fixed_df = type_util.fix_arrow_incompatible_column_types(melted_df, selected_columns=[*columns_to_leave_alone, new_color_column_name, new_y_column_name])
    return fixed_df

def prep_data(df: pd.DataFrame, x_column: Optional[str], y_column_list: List[str], color_column: Optional[str], size_column: Optional[str]) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str], Optional[str]]:
    if False:
        return 10
    'Prepares the data for charting. This is also used in add_rows.\n\n    Returns the prepared dataframe and the new names of the x column (taking the index reset into\n    consideration) and y, color, and size columns.\n    '
    x_column = _maybe_reset_index_in_place(df, x_column, y_column_list)
    selected_data = _drop_unused_columns(df, x_column, color_column, size_column, *y_column_list)
    _maybe_convert_color_column_in_place(selected_data, color_column)
    (x_column, y_column_list, color_column, size_column) = _convert_col_names_to_str_in_place(selected_data, x_column, y_column_list, color_column, size_column)
    (melted_data, y_column, color_column) = _maybe_melt(selected_data, x_column, y_column_list, color_column, size_column)
    return (melted_data, x_column, y_column, color_column, size_column)

def _generate_chart(chart_type: ChartType, data: Optional[Data], x_from_user: Optional[str]=None, y_from_user: Union[str, Sequence[str], None]=None, color_from_user: Union[str, Color, List[Color], None]=None, size_from_user: Union[str, float, None]=None, width: int=0, height: int=0) -> alt.Chart:
    if False:
        return 10
    "Function to use the chart's type, data columns and indices to figure out the chart's spec."
    import altair as alt
    df = type_util.convert_anything_to_df(data, ensure_copy=True)
    del data
    x_column = _parse_x_column(df, x_from_user)
    y_column_list = _parse_y_columns(df, y_from_user, x_column)
    (color_column, color_value) = _parse_generic_column(df, color_from_user)
    (size_column, size_value) = _parse_generic_column(df, size_from_user)
    add_rows_metadata = AddRowsMetadata(last_index=last_index_for_melted_dataframes(df), columns=dict(x_column=x_column, y_column_list=y_column_list, color_column=color_column, size_column=size_column))
    (df, x_column, y_column, color_column, size_column) = prep_data(df, x_column, y_column_list, color_column, size_column)
    chart = alt.Chart(data=df, mark=chart_type.value['mark_type'], width=width, height=height).encode(x=_get_x_encoding(df, x_column, x_from_user, chart_type), y=_get_y_encoding(df, y_column, y_from_user))
    opacity_enc = _get_opacity_encoding(chart_type, color_column)
    if opacity_enc is not None:
        chart = chart.encode(opacity=opacity_enc)
    color_enc = _get_color_encoding(df, color_value, color_column, y_column_list, color_from_user)
    if color_enc is not None:
        chart = chart.encode(color=color_enc)
    size_enc = _get_size_encoding(chart_type, size_column, size_value)
    if size_enc is not None:
        chart = chart.encode(size=size_enc)
    if x_column is not None and y_column is not None:
        chart = chart.encode(tooltip=_get_tooltip_encoding(x_column, y_column, size_column, color_column, color_enc))
    return (chart.interactive(), add_rows_metadata)

def _maybe_reset_index_in_place(df: pd.DataFrame, x_column: Optional[str], y_column_list: List[str]) -> Optional[str]:
    if False:
        return 10
    if x_column is None and len(y_column_list) > 0:
        if df.index.name is None:
            x_column = SEPARATED_INDEX_COLUMN_NAME
        else:
            x_column = df.index.name
        df.index.name = x_column
        df.reset_index(inplace=True)
    return x_column

def _drop_unused_columns(df: pd.DataFrame, *column_names: Optional[str]) -> pd.DataFrame:
    if False:
        return 10
    "Returns a subset of df, selecting only column_names that aren't None."
    seen = set()
    keep = []
    for x in column_names:
        if x is None:
            continue
        if x in seen:
            continue
        seen.add(x)
        keep.append(x)
    return df[keep]

def _maybe_convert_color_column_in_place(df: pd.DataFrame, color_column: Optional[str]):
    if False:
        print('Hello World!')
    'If needed, convert color column to a format Vega understands.'
    if color_column is None or len(df[color_column]) == 0:
        return
    first_color_datum = df[color_column][0]
    if is_hex_color_like(first_color_datum):
        pass
    elif is_color_tuple_like(first_color_datum):
        df[color_column] = df[color_column].map(to_css_color)
    else:
        pass

def _convert_col_names_to_str_in_place(df: pd.DataFrame, x_column: Optional[str], y_column_list: List[str], color_column: Optional[str], size_column: Optional[str]) -> Tuple[Optional[str], List[str], Optional[str], Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Converts column names to strings, since Vega-Lite does not accept ints, etc.'
    column_names = list(df.columns)
    str_column_names = [str(c) for c in column_names]
    df.columns = pd.Index(str_column_names)
    return (None if x_column is None else str(x_column), [str(c) for c in y_column_list], None if color_column is None else str(color_column), None if size_column is None else str(size_column))

def _parse_generic_column(df: pd.DataFrame, column_or_value: Any) -> Tuple[Optional[str], Any]:
    if False:
        print('Hello World!')
    if isinstance(column_or_value, str) and column_or_value in df.columns:
        column_name = column_or_value
        value = None
    else:
        column_name = None
        value = column_or_value
    return (column_name, value)

def _parse_x_column(df: pd.DataFrame, x_from_user: Optional[str]) -> Optional[str]:
    if False:
        print('Hello World!')
    if x_from_user is None:
        return None
    elif isinstance(x_from_user, str):
        if x_from_user not in df.columns:
            raise StreamlitColumnNotFoundError(df, x_from_user)
        return x_from_user
    else:
        raise StreamlitAPIException(f"x parameter should be a column name (str) or None to use the  dataframe's index. Value given: {x_from_user} (type {type(x_from_user)})")

def _parse_y_columns(df: pd.DataFrame, y_from_user: Union[str, Sequence[str], None], x_column: Union[str, None]) -> List[str]:
    if False:
        i = 10
        return i + 15
    y_column_list: List[str] = []
    if y_from_user is None:
        y_column_list = list(df.columns)
    elif isinstance(y_from_user, str):
        y_column_list = [y_from_user]
    elif type_util.is_sequence(y_from_user):
        y_column_list = list((str(col) for col in y_from_user))
    else:
        raise StreamlitAPIException(f'y parameter should be a column name (str) or list thereof. Value given: {y_from_user} (type {type(y_from_user)})')
    for col in y_column_list:
        if col not in df.columns:
            raise StreamlitColumnNotFoundError(df, col)
    if x_column in y_column_list and (not y_from_user or x_column not in y_from_user):
        y_column_list.remove(x_column)
    return y_column_list

def _get_opacity_encoding(chart_type: ChartType, color_column: Optional[str]) -> Optional[alt.OpacityValue]:
    if False:
        print('Hello World!')
    import altair as alt
    if color_column and chart_type == ChartType.AREA:
        return alt.OpacityValue(0.7)
    return None

def _get_scale(df: pd.DataFrame, column_name: Optional[str]) -> alt.Scale:
    if False:
        for i in range(10):
            print('nop')
    import altair as alt
    if _is_date_column(df, column_name):
        return alt.Scale(type='utc')
    return alt.Scale()

def _get_axis_config(df: pd.DataFrame, column_name: Optional[str], grid: bool) -> alt.Axis:
    if False:
        while True:
            i = 10
    import altair as alt
    if column_name is not None and is_integer_dtype(df[column_name]):
        return alt.Axis(tickMinStep=1, grid=grid)
    return alt.Axis(grid=grid)

def _maybe_melt(df: pd.DataFrame, x_column: Optional[str], y_column_list: List[str], color_column: Optional[str], size_column: Optional[str]) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    if False:
        while True:
            i = 10
    'If multiple columns are set for y, melt the dataframe into long format.'
    y_column: Optional[str]
    if len(y_column_list) == 0:
        y_column = None
    elif len(y_column_list) == 1:
        y_column = y_column_list[0]
    elif x_column is not None:
        y_column = MELTED_Y_COLUMN_NAME
        color_column = MELTED_COLOR_COLUMN_NAME
        columns_to_leave_alone = [x_column]
        if size_column:
            columns_to_leave_alone.append(size_column)
        df = _melt_data(df=df, columns_to_leave_alone=columns_to_leave_alone, columns_to_melt=y_column_list, new_y_column_name=y_column, new_color_column_name=color_column)
    return (df, y_column, color_column)

def _get_x_encoding(df: pd.DataFrame, x_column: Optional[str], x_from_user: Optional[str], chart_type: ChartType) -> alt.X:
    if False:
        print('Hello World!')
    import altair as alt
    if x_column is None:
        x_field = NON_EXISTENT_COLUMN_NAME
        x_title = ''
    elif x_column == SEPARATED_INDEX_COLUMN_NAME:
        x_field = x_column
        x_title = ''
    else:
        x_field = x_column
        if x_from_user is None:
            x_title = ''
        else:
            x_title = x_column
    return alt.X(x_field, title=x_title, type=_get_x_encoding_type(df, chart_type, x_column), scale=_get_scale(df, x_column), axis=_get_axis_config(df, x_column, grid=False))

def _get_y_encoding(df: pd.DataFrame, y_column: Optional[str], y_from_user: Union[str, Sequence[str], None]) -> alt.Y:
    if False:
        i = 10
        return i + 15
    import altair as alt
    if y_column is None:
        y_field = NON_EXISTENT_COLUMN_NAME
        y_title = ''
    elif y_column == MELTED_Y_COLUMN_NAME:
        y_field = y_column
        y_title = ''
    else:
        y_field = y_column
        if y_from_user is None:
            y_title = ''
        else:
            y_title = y_column
    return alt.Y(field=y_field, title=y_title, type=_get_y_encoding_type(df, y_column), scale=_get_scale(df, y_column), axis=_get_axis_config(df, y_column, grid=True))

def _get_color_encoding(df: pd.DataFrame, color_value: Optional[Color], color_column: Optional[str], y_column_list: List[str], color_from_user: Union[str, Color, List[Color], None]) -> alt.Color:
    if False:
        i = 10
        return i + 15
    import altair as alt
    has_color_value = color_value not in [None, [], tuple()]
    if has_color_value:
        if is_color_like(cast(Any, color_value)):
            if len(y_column_list) != 1:
                raise StreamlitColorLengthError([color_value], y_column_list)
            return alt.ColorValue(to_css_color(cast(Any, color_value)))
        elif isinstance(color_value, (list, tuple)):
            color_values = cast(Collection[Color], color_value)
            if len(color_values) != len(y_column_list):
                raise StreamlitColorLengthError(color_values, y_column_list)
            if len(color_value) == 1:
                return alt.ColorValue(to_css_color(cast(Any, color_value[0])))
            else:
                return alt.Color(field=color_column, scale=alt.Scale(range=[to_css_color(c) for c in color_values]), legend=COLOR_LEGEND_SETTINGS, type='nominal', title=' ')
        raise StreamlitInvalidColorError(df, color_from_user)
    elif color_column is not None:
        column_type: Union[str, Tuple[str, List[Any]]]
        if color_column == MELTED_COLOR_COLUMN_NAME:
            column_type = 'nominal'
        else:
            column_type = type_util.infer_vegalite_type(df[color_column])
        color_enc = alt.Color(field=color_column, legend=COLOR_LEGEND_SETTINGS, type=column_type)
        if color_column == MELTED_COLOR_COLUMN_NAME:
            color_enc['title'] = ' '
        elif len(df[color_column]) and is_color_like(df[color_column][0]):
            color_range = [to_css_color(c) for c in df[color_column].unique()]
            color_enc['scale'] = alt.Scale(range=color_range)
            color_enc['legend'] = None
        else:
            pass
        return color_enc
    return None

def _get_size_encoding(chart_type: ChartType, size_column: Optional[str], size_value: Union[str, float, None]) -> alt.Size:
    if False:
        while True:
            i = 10
    import altair as alt
    if chart_type == ChartType.SCATTER:
        if size_column is not None:
            return alt.Size(size_column, legend=SIZE_LEGEND_SETTINGS)
        elif isinstance(size_value, (float, int)):
            return alt.SizeValue(size_value)
        elif size_value is None:
            return alt.SizeValue(100)
        else:
            raise StreamlitAPIException(f'This does not look like a valid size: {repr(size_value)}')
    elif size_column is not None or size_value is not None:
        raise Error(f'Chart type {chart_type.name} does not support size argument. This should never happen!')
    return None

def _get_tooltip_encoding(x_column: str, y_column: str, size_column: Optional[str], color_column: Optional[str], color_enc: alt.Color) -> list[alt.Tooltip]:
    if False:
        for i in range(10):
            print('nop')
    import altair as alt
    tooltip = []
    if x_column == SEPARATED_INDEX_COLUMN_NAME:
        tooltip.append(alt.Tooltip(x_column, title=SEPARATED_INDEX_COLUMN_TITLE))
    else:
        tooltip.append(alt.Tooltip(x_column))
    if y_column == MELTED_Y_COLUMN_NAME:
        tooltip.append(alt.Tooltip(y_column, title=MELTED_Y_COLUMN_TITLE, type='quantitative'))
    else:
        tooltip.append(alt.Tooltip(y_column))
    if color_column and getattr(color_enc, 'legend', True) is not None:
        if color_column == MELTED_COLOR_COLUMN_NAME:
            tooltip.append(alt.Tooltip(color_column, title=MELTED_COLOR_COLUMN_TITLE, type='nominal'))
        else:
            tooltip.append(alt.Tooltip(color_column))
    if size_column:
        tooltip.append(alt.Tooltip(size_column))
    return tooltip

def _get_x_encoding_type(df: pd.DataFrame, chart_type: ChartType, x_column: Optional[str]) -> Union[str, Tuple[str, List[Any]]]:
    if False:
        while True:
            i = 10
    if x_column is None:
        return 'quantitative'
    if chart_type == ChartType.BAR and (not _is_date_column(df, x_column)):
        return 'ordinal'
    return type_util.infer_vegalite_type(df[x_column])

def _get_y_encoding_type(df: pd.DataFrame, y_column: Optional[str]) -> Union[str, Tuple[str, List[Any]]]:
    if False:
        for i in range(10):
            print('nop')
    if y_column:
        return type_util.infer_vegalite_type(df[y_column])
    return 'quantitative'

def marshall(vega_lite_chart: ArrowVegaLiteChartProto, altair_chart: alt.Chart, use_container_width: bool=False, theme: Union[None, Literal['streamlit']]='streamlit', **kwargs: Any) -> None:
    if False:
        print('Hello World!')
    "Marshall chart's data into proto."
    import altair as alt
    datasets = {}

    def id_transform(data) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        'Altair data transformer that returns a fake named dataset with the\n        object id.\n        '
        name = str(id(data))
        datasets[name] = data
        return {'name': name}
    alt.data_transformers.register('id', id_transform)
    with alt.themes.enable('none') if alt.themes.active == 'default' else nullcontext():
        with alt.data_transformers.enable('id'):
            chart_dict = altair_chart.to_dict()
            chart_dict['datasets'] = datasets
            arrow_vega_lite.marshall(vega_lite_chart, chart_dict, use_container_width=use_container_width, theme=theme, **kwargs)

class StreamlitColumnNotFoundError(StreamlitAPIException):

    def __init__(self, df, col_name, *args):
        if False:
            while True:
                i = 10
        available_columns = ', '.join((str(c) for c in list(df.columns)))
        message = f'Data does not have a column named `"{col_name}"`. Available columns are `{available_columns}`'
        super().__init__(message, *args)

class StreamlitInvalidColorError(StreamlitAPIException):

    def __init__(self, df, color_from_user, *args):
        if False:
            print('Hello World!')
        ', '.join((str(c) for c in list(df.columns)))
        message = f'\nThis does not look like a valid color argument: `{color_from_user}`.\n\nThe color argument can be:\n\n* A hex string like "#ffaa00" or "#ffaa0088".\n* An RGB or RGBA tuple with the red, green, blue, and alpha\n  components specified as ints from 0 to 255 or floats from 0.0 to\n  1.0.\n* The name of a column.\n* Or a list of colors, matching the number of y columns to draw.\n        '
        super().__init__(message, *args)

class StreamlitColorLengthError(StreamlitAPIException):

    def __init__(self, color_values, y_column_list, *args):
        if False:
            return 10
        message = f'The list of colors `{color_values}` must have the same length as the list of columns to be colored `{y_column_list}`.'
        super().__init__(message, *args)