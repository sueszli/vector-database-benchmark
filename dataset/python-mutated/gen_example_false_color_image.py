"""
generates 'example_false_color_image.png'
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as exp
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets, mkQApp

class MainWindow(pg.GraphicsLayoutWidget):
    """ example application main window """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.resize(420, 400)
        self.show()
        plot = self.addPlot()
        data = np.fromfunction(lambda i, j: (1 + 0.3 * np.sin(i)) * i ** 2 + j ** 2, (100, 100))
        noisy_data = data * (1 + 0.2 * np.random.random(data.shape))
        img = pg.ImageItem(image=noisy_data)
        plot.addItem(img)
        cm = pg.colormap.get('CET-L9')
        bar = pg.ColorBarItem(values=(0, 20000), cmap=cm)
        bar.setImageItem(img, insert_in=plot)
        self.timer = pg.QtCore.QTimer(singleShot=True)
        self.timer.timeout.connect(self.export)
        self.timer.start(100)

    def export(self):
        if False:
            i = 10
            return i + 15
        print('exporting')
        exporter = exp.ImageExporter(self.scene())
        exporter.parameters()['width'] = 420
        exporter.export('example_false_color_image.png')
mkQApp('False color image example')
main_window = MainWindow()
if __name__ == '__main__':
    pg.exec()