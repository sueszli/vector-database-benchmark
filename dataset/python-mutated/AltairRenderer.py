import lux
import pandas as pd
from typing import Callable
from lux.vislib.altair.BarChart import BarChart
from lux.vislib.altair.ScatterChart import ScatterChart
from lux.vislib.altair.LineChart import LineChart
from lux.vislib.altair.Histogram import Histogram
from lux.vislib.altair.Heatmap import Heatmap
from lux.vislib.altair.Choropleth import Choropleth

class AltairRenderer:
    """
    Renderer for Charts based on Altair (https://altair-viz.github.io/)
    """

    def __init__(self, output_type='VegaLite'):
        if False:
            while True:
                i = 10
        self.output_type = output_type

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'AltairRenderer'

    def create_vis(self, vis, standalone=True):
        if False:
            while True:
                i = 10
        '\n        Input Vis object and return a visualization specification\n\n        Parameters\n        ----------\n        vis: lux.vis.Vis\n                Input Vis (with data)\n        standalone: bool\n                Flag to determine if outputted code uses user-defined variable names or can be run independently\n        Returns\n        -------\n        chart : altair.Chart\n                Output Altair Chart Object\n        '
        if vis.approx:
            if vis.mark == 'scatter' and vis._postbin:
                vis._mark = 'heatmap'
                lux.config.executor.execute_2D_binning(vis)
            else:
                lux.config.executor.execute([vis], vis._original_df, approx=False)
        if vis.data is not None:
            for attr in list(vis.data.columns):
                if pd.api.types.is_period_dtype(vis.data.dtypes[attr]) or isinstance(vis.data[attr].iloc[0], pd.Period):
                    dateColumn = vis.data[attr]
                    vis.data[attr] = pd.PeriodIndex(dateColumn.values).to_timestamp()
                if pd.api.types.is_interval_dtype(vis.data.dtypes[attr]) or isinstance(vis.data[attr].iloc[0], pd.Interval):
                    vis.data[attr] = vis.data[attr].astype(str)
                if isinstance(attr, str):
                    if '.' in attr:
                        attr_clause = vis.get_attr_by_attr_name(attr)[0]
                        vis._vis_data = vis.data.rename(columns={attr: attr.replace('.', '')})
        if vis.mark == 'histogram':
            chart = Histogram(vis)
        elif vis.mark == 'bar':
            chart = BarChart(vis)
        elif vis.mark == 'scatter':
            chart = ScatterChart(vis)
        elif vis.mark == 'line':
            chart = LineChart(vis)
        elif vis.mark == 'heatmap':
            chart = Heatmap(vis)
        elif vis.mark == 'geographical':
            chart = Choropleth(vis)
        else:
            chart = None
        if chart:
            if lux.config.plotting_style and (lux.config.plotting_backend == 'vegalite' or lux.config.plotting_backend == 'altair'):
                chart.chart = lux.config.plotting_style(chart.chart)
            if self.output_type == 'VegaLite':
                chart_dict = chart.chart.to_dict()
                chart_dict['vislib'] = 'vegalite'
                return chart_dict
            elif self.output_type == 'Altair':
                import inspect
                if lux.config.plotting_style:
                    chart.code += '\n'.join(inspect.getsource(lux.config.plotting_style).split('\n    ')[1:-1])
                chart.code += '\nchart'
                chart.code = chart.code.replace('\n\t\t', '\n')
                var = vis._source
                if var is not None:
                    all_vars = []
                    for f_info in inspect.getouterframes(inspect.currentframe()):
                        local_vars = f_info.frame.f_back
                        if local_vars:
                            callers_local_vars = local_vars.f_locals.items()
                            possible_vars = [var_name for (var_name, var_val) in callers_local_vars if var_val is var]
                            all_vars.extend(possible_vars)
                    for possible_var in all_vars:
                        if possible_var[0] != '_':
                            print(possible_var)
                    found_variable = [possible_var for possible_var in all_vars if possible_var[0] != '_']
                    if len(found_variable) > 0:
                        found_variable = found_variable[0]
                    else:
                        found_variable = 'df'
                else:
                    found_variable = 'df'
                if standalone:
                    chart.code = chart.code.replace('placeholder_variable', f'pd.DataFrame({str(vis.data.to_dict())})')
                else:
                    chart.code = chart.code.replace('placeholder_variable', found_variable)
                return chart.code