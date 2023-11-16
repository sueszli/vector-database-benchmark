"""
generates 'example_false_color_image.png'
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as exp
from pyqtgraph.Qt import QtGui, mkQApp

class MainWindow(pg.GraphicsLayoutWidget):
    """ example application main window """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.resize(420, 400)
        self.show()
        plot = self.addPlot()
        tr = QtGui.QTransform()
        tr.scale(6.0, 3.0)
        tr.translate(-1.5, -1.5)
        img = pg.ImageItem(image=np.eye(3), levels=(0, 1))
        img.setTransform(tr)
        plot.addItem(img)
        plot.showAxes(True)
        plot.invertY(True)
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
        exporter.export('example_imageitem_transform.png')
mkQApp('ImageItem transform example')
main_window = MainWindow()
if __name__ == '__main__':
    pg.exec()