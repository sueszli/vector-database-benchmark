import pandas as pd
import matplotlib.pyplot as plt

class MatplotlibChart:
    """
    MatplotlibChart is a representation of a chart.
    Common utilities for charts that is independent of chart types should go here.

    See Also
    --------
    https://matplotlib.org/

    """

    def __init__(self, vis, fig, ax):
        if False:
            i = 10
            return i + 15
        self.vis = vis
        self.data = vis.data
        self.tooltip = True
        self.fig = fig
        self.ax = ax
        self.code = ''
        self.apply_default_config()
        self.chart = self.initialize_chart()
        self.add_title()

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'MatplotlibChart <{str(self.vis)}>'

    def add_tooltip(self):
        if False:
            return 10
        return NotImplemented

    def apply_default_config(self):
        if False:
            print('Hello World!')
        self.code += 'import matplotlib.pyplot as plt\n'
        self.code += 'plt.rcParams.update(\n            {\n                "axes.titlesize": 20,\n                "axes.titleweight": "bold",\n                "axes.labelweight": "bold",\n                "axes.labelsize": 16,\n                "legend.fontsize": 14,\n                "legend.title_fontsize": 15,\n                "xtick.labelsize": 13,\n                "ytick.labelsize": 13,\n            }\n        )\n'

    def encode_color(self):
        if False:
            i = 10
            return i + 15
        return NotImplemented

    def add_title(self):
        if False:
            while True:
                i = 10
        chart_title = self.vis.title
        if chart_title:
            if len(chart_title) > 25:
                chart_title = chart_title[:15] + '...' + chart_title[-10:]
            self.ax.set_title(chart_title)
            self.code += f"ax.set_title('{chart_title}')\n"

    def initialize_chart(self):
        if False:
            print('Hello World!')
        return NotImplemented