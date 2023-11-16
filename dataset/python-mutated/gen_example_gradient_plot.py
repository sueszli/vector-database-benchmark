"""
generates 'example_gradient_plot.png'
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as exp
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

class MainWindow(pg.GraphicsLayoutWidget):
    """ example application main window """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.resize(420, 400)
        self.show()
        raw = np.linspace(0.0, 2.0, 400)
        y_data1 = ((raw + 0.1) % 1) ** 4
        y_data2 = ((raw + 0.1) % 1) ** 4 - ((raw + 0.6) % 1) ** 4
        cm = pg.colormap.get('CET-L17')
        cm.reverse()
        pen = cm.getPen(span=(0.0, 1.0), width=5)
        curve1 = pg.PlotDataItem(y=y_data1, pen=pen)
        cm = pg.colormap.get('CET-D1')
        cm.setMappingMode('diverging')
        brush = cm.getBrush(span=(-1.0, 1.0))
        curve2 = pg.PlotDataItem(y=y_data2, pen='w', brush=brush, fillLevel=0.0)
        for (idx, curve) in enumerate((curve1, curve2)):
            plot = self.addPlot(row=idx, col=0)
            plot.getAxis('left').setWidth(25)
            plot.addItem(curve)
        self.timer = pg.QtCore.QTimer(singleShot=True)
        self.timer.timeout.connect(self.export)
        self.timer.start(100)

    def export(self):
        if False:
            for i in range(10):
                print('nop')
        print('exporting')
        exporter = exp.ImageExporter(self.scene())
        exporter.parameters()['width'] = 420
        exporter.export('example_gradient_plot.png')
mkQApp('Gradient plotting example')
main_window = MainWindow()
if __name__ == '__main__':
    pg.exec()