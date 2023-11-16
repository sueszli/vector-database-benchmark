import re
import lux
import numpy as np
import pandas as pd
from lux.utils.date_utils import compute_date_granularity
import altair as alt

class AltairChart:
    """
    AltairChart is a representation of a chart.
    Common utilities for charts that is independent of chart types should go here.

    See Also
    --------
    altair-viz.github.io

    """

    def __init__(self, vis):
        if False:
            print('Hello World!')
        self.vis = vis
        self.data = vis.data
        self.tooltip = True
        self.code = ''
        self.width = 160
        self.height = 150
        self.chart = self.initialize_chart()
        self.encode_color()
        self.add_title()
        self.apply_default_config()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'AltairChart <{str(self.vis)}>'

    def add_tooltip(self):
        if False:
            for i in range(10):
                print('nop')
        if self.tooltip:
            self.chart = self.chart.encode(tooltip=list(self.vis.data.columns))

    def apply_default_config(self):
        if False:
            return 10
        self.chart = self.chart.configure_title(fontWeight=500, fontSize=13, font='Helvetica Neue')
        self.chart = self.chart.configure_axis(titleFontWeight=500, titleFontSize=11, titleFont='Helvetica Neue', labelFontWeight=400, labelFontSize=9, labelFont='Helvetica Neue', labelColor='#505050')
        self.chart = self.chart.configure_legend(titleFontWeight=500, titleFontSize=10, titleFont='Helvetica Neue', labelFontWeight=400, labelFontSize=9, labelFont='Helvetica Neue')
        plotting_scale = lux.config.plotting_scale
        self.chart = self.chart.properties(width=self.width * plotting_scale, height=self.height * plotting_scale)
        self.code += "\nchart = chart.configure_title(fontWeight=500,fontSize=13,font='Helvetica Neue')\n"
        self.code += "chart = chart.configure_axis(titleFontWeight=500,titleFontSize=11,titleFont='Helvetica Neue',\n"
        self.code += "\t\t\t\t\tlabelFontWeight=400,labelFontSize=8,labelFont='Helvetica Neue',labelColor='#505050')\n"
        self.code += "chart = chart.configure_legend(titleFontWeight=500,titleFontSize=10,titleFont='Helvetica Neue',\n"
        self.code += "\t\t\t\t\tlabelFontWeight=400,labelFontSize=8,labelFont='Helvetica Neue')\n"
        self.code += f'chart = chart.properties(width={self.width * plotting_scale},height={self.height * plotting_scale})\n'

    def encode_color(self):
        if False:
            return 10
        color_attr = self.vis.get_attr_by_channel('color')
        if len(color_attr) == 1:
            color_attr_name = color_attr[0].attribute
            color_attr_type = color_attr[0].data_type
            if color_attr_type == 'temporal':
                timeUnit = compute_date_granularity(self.vis.data[color_attr_name])
                self.chart = self.chart.encode(color=alt.Color(str(color_attr_name), type=color_attr_type, timeUnit=timeUnit, title=color_attr_name))
                self.code += f"chart = chart.encode(color=alt.Color('{color_attr_name}',type='{color_attr_type}',timeUnit='{timeUnit}',title='{color_attr_name}'))"
            else:
                self.chart = self.chart.encode(color=alt.Color(str(color_attr_name), type=color_attr_type))
                self.code += f"chart = chart.encode(color=alt.Color('{color_attr_name}',type='{color_attr_type}'))\n"
        elif len(color_attr) > 1:
            raise ValueError('There should not be more than one attribute specified in the same channel.')

    def add_title(self):
        if False:
            for i in range(10):
                print('nop')
        chart_title = self.vis.title
        if chart_title:
            if len(chart_title) > 25:
                chart_title = chart_title[:15] + '...' + chart_title[-10:]
            self.chart = self.chart.encode().properties(title=chart_title)
            if self.code != '':
                self.code += f"chart = chart.encode().properties(title = '{chart_title}')"

    def initialize_chart(self):
        if False:
            i = 10
            return i + 15
        return NotImplemented

    @classmethod
    def sanitize_dataframe(self, df):
        if False:
            for i in range(10):
                print('nop')
        from lux.utils.date_utils import is_timedelta64_series, timedelta64_to_float_seconds
        for attr in df.columns:
            if str(df[attr].dtype) == 'Float64':
                df[attr] = df[attr].astype(np.float64)
            if is_timedelta64_series(df[attr]):
                df[attr] = timedelta64_to_float_seconds(df[attr])
            df = df.rename(columns={attr: str(attr)})
        return df