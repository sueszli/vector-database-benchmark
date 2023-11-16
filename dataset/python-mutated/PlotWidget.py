"""
Demonstrates use of PlotWidget class. This is little more than a 
GraphicsView with a PlotItem placed in its center.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
app = pg.mkQApp()
mw = QtWidgets.QMainWindow()
mw.setWindowTitle('pyqtgraph example: PlotWidget')
mw.resize(800, 800)
cw = QtWidgets.QWidget()
mw.setCentralWidget(cw)
l = QtWidgets.QVBoxLayout()
cw.setLayout(l)
pw = pg.PlotWidget(name='Plot1')
l.addWidget(pw)
pw2 = pg.PlotWidget(name='Plot2')
l.addWidget(pw2)
pw3 = pg.PlotWidget()
l.addWidget(pw3)
mw.show()
p1 = pw.plot()
p1.setPen((200, 200, 100))
rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 5e-11))
rect.setPen(pg.mkPen(100, 200, 100))
pw.addItem(rect)
pw.setLabel('left', 'Value', units='V')
pw.setLabel('bottom', 'Time', units='s')
pw.setXRange(0, 2)
pw.setYRange(0, 1e-10)

def rand(n):
    if False:
        i = 10
        return i + 15
    data = np.random.random(n)
    data[int(n * 0.1):int(n * 0.13)] += 0.5
    data[int(n * 0.18)] += 2
    data[int(n * 0.1):int(n * 0.13)] *= 5
    data[int(n * 0.18)] *= 20
    data *= 1e-12
    return (data, np.arange(n, n + len(data)) / float(n))

def updateData():
    if False:
        while True:
            i = 10
    (yd, xd) = rand(10000)
    p1.setData(y=yd, x=xd)
t = QtCore.QTimer()
t.timeout.connect(updateData)
t.start(50)
for i in range(0, 5):
    for j in range(0, 3):
        (yd, xd) = rand(10000)
        pw2.plot(y=yd * (j + 1), x=xd, params={'iter': i, 'val': j})
curve = pw3.plot(np.random.normal(size=100) * 1.0, clickable=True)
curve.curve.setClickable(True)
curve.setPen('w')
curve.setShadowPen(pg.mkPen((70, 70, 30), width=6, cosmetic=True))

def clicked():
    if False:
        print('Hello World!')
    print('curve clicked')
curve.sigClicked.connect(clicked)
lr = pg.LinearRegionItem([1, 30], bounds=[0, 100], movable=True)
pw3.addItem(lr)
line = pg.InfiniteLine(angle=90, movable=True)
pw3.addItem(line)
line.setBounds([0, 200])
if __name__ == '__main__':
    pg.exec()