"""
Example demonstrating a variety of scatter plot features.
"""
from collections import namedtuple
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
app = pg.mkQApp('Scatter Plot Item Example')
mw = QtWidgets.QMainWindow()
mw.resize(800, 800)
view = pg.GraphicsLayoutWidget()
mw.setCentralWidget(view)
mw.show()
mw.setWindowTitle('pyqtgraph example: ScatterPlot')
w1 = view.addPlot()
w2 = view.addViewBox()
w2.setAspectLocked(True)
view.nextRow()
w3 = view.addPlot()
w4 = view.addPlot()
print('Generating data, this takes a few seconds...')
clickedPen = pg.mkPen('b', width=2)
lastClicked = []

def clicked(plot, points):
    if False:
        return 10
    global lastClicked
    for p in lastClicked:
        p.resetPen()
    print('clicked points', points)
    for p in points:
        p.setPen(clickedPen)
    lastClicked = points
n = 300
s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
pos = np.random.normal(size=(2, n), scale=1e-05)
spots = [{'pos': pos[:, i], 'data': 1} for i in range(n)] + [{'pos': [0, 0], 'data': 1}]
s1.addPoints(spots)
w1.addItem(s1)
s1.sigClicked.connect(clicked)
TextSymbol = namedtuple('TextSymbol', 'label symbol scale')

def createLabel(label, angle):
    if False:
        print('Hello World!')
    symbol = QtGui.QPainterPath()
    f = QtGui.QFont()
    f.setPointSize(10)
    symbol.addText(0, 0, f, label)
    br = symbol.boundingRect()
    scale = min(1.0 / br.width(), 1.0 / br.height())
    tr = QtGui.QTransform()
    tr.scale(scale, scale)
    tr.rotate(angle)
    tr.translate(-br.x() - br.width() / 2.0, -br.y() - br.height() / 2.0)
    return TextSymbol(label, tr.map(symbol), 0.1 / scale)
random_str = lambda : (''.join([chr(np.random.randint(ord('A'), ord('z'))) for i in range(np.random.randint(1, 5))]), np.random.randint(0, 360))
s2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
pos = np.random.normal(size=(2, n), scale=1e-05)
spots = [{'pos': pos[:, i], 'data': 1, 'brush': pg.intColor(i, n), 'symbol': i % 10, 'size': 5 + i / 10.0} for i in range(n)]
s2.addPoints(spots)
spots = [{'pos': pos[:, i], 'data': 1, 'brush': pg.intColor(i, n), 'symbol': label[1], 'size': label[2] * (5 + i / 10.0)} for (i, label) in [(i, createLabel(*random_str())) for i in range(n)]]
s2.addPoints(spots)
w2.addItem(s2)
s2.sigClicked.connect(clicked)
s3 = pg.ScatterPlotItem(pxMode=False, hoverable=True, hoverPen=pg.mkPen('g'), hoverSize=1e-06)
spots3 = []
for i in range(10):
    for j in range(10):
        spots3.append({'pos': (1e-06 * i, 1e-06 * j), 'size': 1e-06, 'pen': {'color': 'w', 'width': 2}, 'brush': pg.intColor(i * 10 + j, 100)})
s3.addPoints(spots3)
w3.addItem(s3)
s3.sigClicked.connect(clicked)
s4 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 20), hoverable=True, hoverSymbol='s', hoverSize=15, hoverPen=pg.mkPen('r', width=2), hoverBrush=pg.mkBrush('g'))
n = 10000
pos = np.random.normal(size=(2, n), scale=1e-09)
s4.addPoints(x=pos[0], y=pos[1], data=np.arange(n))
w4.addItem(s4)
s4.sigClicked.connect(clicked)
if __name__ == '__main__':
    pg.exec()