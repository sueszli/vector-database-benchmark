"""gr.ScatterPlot() component."""
from __future__ import annotations
from typing import Any, Callable, Literal
import altair as alt
import pandas as pd
from gradio_client.documentation import document, set_documentation_group
from pandas.api.types import is_numeric_dtype
from gradio.components.plot import AltairPlot, AltairPlotData, Plot
set_documentation_group('component')

@document()
class ScatterPlot(Plot):
    """
    Create a scatter plot.

    Preprocessing: this component does *not* accept input.
    Postprocessing: expects a pandas dataframe with the data to plot.

    Demos: scatter_plot
    Guides: creating-a-dashboard-from-bigquery-data
    """
    data_model = AltairPlotData

    def __init__(self, value: pd.DataFrame | Callable | None=None, x: str | None=None, y: str | None=None, *, color: str | None=None, size: str | None=None, shape: str | None=None, title: str | None=None, tooltip: list[str] | str | None=None, x_title: str | None=None, y_title: str | None=None, x_label_angle: float | None=None, y_label_angle: float | None=None, color_legend_title: str | None=None, size_legend_title: str | None=None, shape_legend_title: str | None=None, color_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, size_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, shape_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, height: int | None=None, width: int | None=None, x_lim: list[int | float] | None=None, y_lim: list[int | float] | None=None, caption: str | None=None, interactive: bool | None=True, label: str | None=None, every: float | None=None, show_label: bool | None=None, container: bool=True, scale: int | None=None, min_width: int=160, visible: bool=True, elem_id: str | None=None, elem_classes: list[str] | str | None=None, render: bool=True, show_actions_button: bool=False):
        if False:
            while True:
                i = 10
        "\n        Parameters:\n            value: The pandas dataframe containing the data to display in a scatter plot, or a callable. If callable, the function will be called whenever the app loads to set the initial value of the component.\n            x: Column corresponding to the x axis.\n            y: Column corresponding to the y axis.\n            color: The column to determine the point color. If the column contains numeric data, gradio will interpolate the column data so that small values correspond to light colors and large values correspond to dark values.\n            size: The column used to determine the point size. Should contain numeric data so that gradio can map the data to the point size.\n            shape: The column used to determine the point shape. Should contain categorical data. Gradio will map each unique value to a different shape.\n            title: The title to display on top of the chart.\n            tooltip: The column (or list of columns) to display on the tooltip when a user hovers a point on the plot.\n            x_title: The title given to the x-axis. By default, uses the value of the x parameter.\n            y_title: The title given to the y-axis. By default, uses the value of the y parameter.\n            x_label_angle:  The angle for the x axis labels rotation. Positive values are clockwise, and negative values are counter-clockwise.\n            y_label_angle:  The angle for the y axis labels rotation. Positive values are clockwise, and negative values are counter-clockwise.\n            color_legend_title: The title given to the color legend. By default, uses the value of color parameter.\n            size_legend_title: The title given to the size legend. By default, uses the value of the size parameter.\n            shape_legend_title: The title given to the shape legend. By default, uses the value of the shape parameter.\n            color_legend_position: The position of the color legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.\n            size_legend_position: The position of the size legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.\n            shape_legend_position: The position of the shape legend. If the string value 'none' is passed, this legend is omitted. For other valid position values see: https://vega.github.io/vega/docs/legends/#orientation.\n            height: The height of the plot in pixels.\n            width: The width of the plot in pixels.\n            x_lim: A tuple or list containing the limits for the x-axis, specified as [x_min, x_max].\n            y_lim: A tuple of list containing the limits for the y-axis, specified as [y_min, y_max].\n            caption: The (optional) caption to display below the plot.\n            interactive: Whether users should be able to interact with the plot by panning or zooming with their mouse or trackpad.\n            label: The (optional) label to display on the top left corner of the plot.\n            every:  If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.\n            show_label: Whether the label should be displayed.\n            visible: Whether the plot should be visible.\n            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.\n            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.\n            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.\n            show_actions_button: Whether to show the actions button on the top right corner of the plot.\n        "
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.shape = shape
        self.tooltip = tooltip
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.x_label_angle = x_label_angle
        self.y_label_angle = y_label_angle
        self.color_legend_title = color_legend_title
        self.color_legend_position = color_legend_position
        self.size_legend_title = size_legend_title
        self.size_legend_position = size_legend_position
        self.shape_legend_title = shape_legend_title
        self.shape_legend_position = shape_legend_position
        self.caption = caption
        self.interactive_chart = interactive
        self.width = width
        self.height = height
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.show_actions_button = show_actions_button
        super().__init__(value=value, label=label, every=every, show_label=show_label, container=container, scale=scale, min_width=min_width, visible=visible, elem_id=elem_id, elem_classes=elem_classes, render=render)

    def get_block_name(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'plot'

    @staticmethod
    def create_plot(value: pd.DataFrame, x: str, y: str, color: str | None=None, size: str | None=None, shape: str | None=None, title: str | None=None, tooltip: list[str] | str | None=None, x_title: str | None=None, y_title: str | None=None, x_label_angle: float | None=None, y_label_angle: float | None=None, color_legend_title: str | None=None, size_legend_title: str | None=None, shape_legend_title: str | None=None, color_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, size_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, shape_legend_position: Literal['left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'none'] | None=None, height: int | None=None, width: int | None=None, x_lim: list[int | float] | None=None, y_lim: list[int | float] | None=None, interactive: bool | None=True):
        if False:
            return 10
        'Helper for creating the scatter plot.'
        interactive = True if interactive is None else interactive
        encodings = {'x': alt.X(x, title=x_title or x, scale=AltairPlot.create_scale(x_lim), axis=alt.Axis(labelAngle=x_label_angle) if x_label_angle is not None else alt.Axis()), 'y': alt.Y(y, title=y_title or y, scale=AltairPlot.create_scale(y_lim), axis=alt.Axis(labelAngle=y_label_angle) if y_label_angle is not None else alt.Axis())}
        properties = {}
        if title:
            properties['title'] = title
        if height:
            properties['height'] = height
        if width:
            properties['width'] = width
        if color:
            if is_numeric_dtype(value[color]):
                domain = [value[color].min(), value[color].max()]
                range_ = [0, 1]
                type_ = 'quantitative'
            else:
                domain = value[color].unique().tolist()
                range_ = list(range(len(domain)))
                type_ = 'nominal'
            encodings['color'] = {'field': color, 'type': type_, 'legend': AltairPlot.create_legend(position=color_legend_position, title=color_legend_title or color), 'scale': {'domain': domain, 'range': range_}}
        if tooltip:
            encodings['tooltip'] = tooltip
        if size:
            encodings['size'] = {'field': size, 'type': 'quantitative' if is_numeric_dtype(value[size]) else 'nominal', 'legend': AltairPlot.create_legend(position=size_legend_position, title=size_legend_title or size)}
        if shape:
            encodings['shape'] = {'field': shape, 'type': 'quantitative' if is_numeric_dtype(value[shape]) else 'nominal', 'legend': AltairPlot.create_legend(position=shape_legend_position, title=shape_legend_title or shape)}
        chart = alt.Chart(value).mark_point(clip=True).encode(**encodings).properties(background='transparent', **properties)
        if interactive:
            chart = chart.interactive()
        return chart

    def postprocess(self, value: pd.DataFrame | dict | None) -> AltairPlotData | dict | None:
        if False:
            return 10
        if value is None or isinstance(value, dict):
            return value
        if self.x is None or self.y is None:
            raise ValueError('No value provided for required parameters `x` and `y`.')
        chart = self.create_plot(value=value, x=self.x, y=self.y, color=self.color, size=self.size, shape=self.shape, title=self.title, tooltip=self.tooltip, x_title=self.x_title, y_title=self.y_title, x_label_angle=self.x_label_angle, y_label_angle=self.y_label_angle, color_legend_title=self.color_legend_title, size_legend_title=self.size_legend_title, shape_legend_title=self.size_legend_title, color_legend_position=self.color_legend_position, size_legend_position=self.size_legend_position, shape_legend_position=self.shape_legend_position, interactive=self.interactive_chart, height=self.height, width=self.width, x_lim=self.x_lim, y_lim=self.y_lim)
        return AltairPlotData(type='altair', plot=chart.to_json(), chart='scatter')

    def example_inputs(self) -> Any:
        if False:
            while True:
                i = 10
        return None

    def preprocess(self, payload: AltairPlotData | None) -> AltairPlotData | None:
        if False:
            print('Hello World!')
        return payload