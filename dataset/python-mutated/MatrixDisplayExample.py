"""
This example demonstrates ViewBox and AxisItem configuration to plot a correlation matrix.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets, mkQApp

class MainWindow(QtWidgets.QMainWindow):
    """ example application main window """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(MainWindow, self).__init__(*args, **kwargs)
        gr_wid = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(gr_wid)
        self.setWindowTitle('pyqtgraph example: Correlation matrix display')
        self.resize(600, 500)
        self.show()
        corrMatrix = np.array([[1.0, 0.5184571, -0.70188642], [0.5184571, 1.0, -0.86094096], [-0.70188642, -0.86094096, 1.0]])
        columns = ['A', 'B', 'C']
        pg.setConfigOption('imageAxisOrder', 'row-major')
        correlogram = pg.ImageItem()
        tr = QtGui.QTransform().translate(-0.5, -0.5)
        correlogram.setTransform(tr)
        correlogram.setImage(corrMatrix)
        plotItem = gr_wid.addPlot()
        plotItem.invertY(True)
        plotItem.setDefaultPadding(0.0)
        plotItem.addItem(correlogram)
        plotItem.showAxes(True, showValues=(True, True, False, False), size=20)
        ticks = [(idx, label) for (idx, label) in enumerate(columns)]
        for side in ('left', 'top', 'right', 'bottom'):
            plotItem.getAxis(side).setTicks((ticks, []))
        plotItem.getAxis('bottom').setHeight(10)
        colorMap = pg.colormap.get('CET-D1')
        bar = pg.ColorBarItem(values=(-1, 1), colorMap=colorMap)
        bar.setImageItem(correlogram, insert_in=plotItem)
mkQApp('Correlation matrix display')
main_window = MainWindow()
if __name__ == '__main__':
    pg.exec()