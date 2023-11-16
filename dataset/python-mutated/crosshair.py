"""
Demonstrates some customized mouse interaction by drawing a crosshair that follows 
the mouse.
"""
import numpy as np
import pyqtgraph as pg
app = pg.mkQApp('Crosshair Example')
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('pyqtgraph example: crosshair')
label = pg.LabelItem(justify='right')
win.addItem(label)
p1 = win.addPlot(row=1, col=0)
p1.avgPen = pg.mkPen('#FFFFFF')
p1.avgShadowPen = pg.mkPen('#8080DD', width=10)
p2 = win.addPlot(row=2, col=0)
region = pg.LinearRegionItem()
region.setZValue(10)
p2.addItem(region, ignoreBounds=True)
p1.setAutoVisible(y=True)
data1 = 10000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
data2 = 15000 + 15000 * pg.gaussianFilter(np.random.random(size=10000), 10) + 3000 * np.random.random(size=10000)
p1.plot(data1, pen='r')
p1.plot(data2, pen='g')
p2d = p2.plot(data1, pen='w')
region.setClipItem(p2d)

def update():
    if False:
        print('Hello World!')
    region.setZValue(10)
    (minX, maxX) = region.getRegion()
    p1.setXRange(minX, maxX, padding=0)
region.sigRegionChanged.connect(update)

def updateRegion(window, viewRange):
    if False:
        while True:
            i = 10
    rgn = viewRange[0]
    region.setRegion(rgn)
p1.sigRangeChanged.connect(updateRegion)
region.setRegion([1000, 2000])
vLine = pg.InfiniteLine(angle=90, movable=False)
hLine = pg.InfiniteLine(angle=0, movable=False)
p1.addItem(vLine, ignoreBounds=True)
p1.addItem(hLine, ignoreBounds=True)
vb = p1.vb

def mouseMoved(evt):
    if False:
        print('Hello World!')
    pos = evt
    if p1.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        index = int(mousePoint.x())
        if index > 0 and index < len(data1):
            label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        vLine.setPos(mousePoint.x())
        hLine.setPos(mousePoint.y())
p1.scene().sigMouseMoved.connect(mouseMoved)
if __name__ == '__main__':
    pg.exec()