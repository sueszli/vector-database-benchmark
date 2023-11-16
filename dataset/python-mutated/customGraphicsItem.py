"""
Demonstrate creation of a custom graphic (a candlestick plot)

"""
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

class CandlestickItem(pg.GraphicsObject):

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()

    def generatePicture(self):
        if False:
            i = 10
            return i + 15
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('w'))
        w = (self.data[1][0] - self.data[0][0]) / 3.0
        for (t, open, close, min, max) in self.data:
            p.drawLine(QtCore.QPointF(t, min), QtCore.QPointF(t, max))
            if open > close:
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setBrush(pg.mkBrush('g'))
            p.drawRect(QtCore.QRectF(t - w, open, w * 2, close - open))
        p.end()

    def paint(self, p, *args):
        if False:
            return 10
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if False:
            return 10
        return QtCore.QRectF(self.picture.boundingRect())
data = [(1.0, 10, 13, 5, 15), (2.0, 13, 17, 9, 20), (3.0, 17, 14, 11, 23), (4.0, 14, 15, 5, 19), (5.0, 15, 9, 8, 22), (6.0, 9, 15, 8, 16)]
item = CandlestickItem(data)
plt = pg.plot()
plt.addItem(item)
plt.setWindowTitle('pyqtgraph example: customGraphicsItem')
if __name__ == '__main__':
    pg.exec()