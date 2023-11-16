import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph import DateAxisItem
from pyqtgraph.graphicsItems.DateAxisItem import YEAR_SPACING
from tribler.gui.defs import BITTORRENT_BIRTHDAY

class TimeSeriesPlot(pg.PlotWidget):

    def __init__(self, parent, name, series, **kargs):
        if False:
            print('Hello World!')
        axis_items = kargs.pop('axis_items', {'bottom': DateAxisItem('bottom')})
        super().__init__(parent=parent, title=name, axisItems=axis_items, **kargs)
        self.getPlotItem().showGrid(x=True, y=True)
        self.setBackground('#202020')
        self.setAntialiasing(True)
        self.setMenuEnabled(False)
        self.plot_data = {}
        self.plots = []
        self.series = series
        self.last_timestamp = 0
        legend = pg.LegendItem((150, 25 * len(series)), offset=(150, 30))
        legend.setParentItem(self.graphicsItem())
        for serie in series:
            plot = self.plot(**serie)
            self.plots.append(plot)
            legend.addItem(plot, serie['name'])
        self.setLimits(xMin=BITTORRENT_BIRTHDAY, xMax=time.time() + YEAR_SPACING)

    def setup_labels(self):
        if False:
            print('Hello World!')
        pass

    def reset_plot(self):
        if False:
            return 10
        self.plot_data = {}

    def add_data(self, timestamp, data):
        if False:
            return 10
        self.plot_data[timestamp] = data

    def render_plot(self):
        if False:
            i = 10
            return i + 15
        self.plot_data = dict(sorted(self.plot_data.items(), key=lambda x: x[0]))
        for (i, plot) in enumerate(self.plots):
            plot.setData(x=np.array(list(self.plot_data.keys())), y=np.array([data[i] for data in self.plot_data.values()]))