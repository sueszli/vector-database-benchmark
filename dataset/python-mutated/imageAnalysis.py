"""
Demonstrates common image analysis tools.

Many of the features demonstrated here are already provided by the ImageView
widget, but here we present a lower-level approach that provides finer control
over the user interface.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
pg.setConfigOptions(imageAxisOrder='row-major')
pg.mkQApp()
win = pg.GraphicsLayoutWidget()
win.setWindowTitle('pyqtgraph example: Image Analysis')
p1 = win.addPlot(title='')
img = pg.ImageItem()
p1.addItem(img)
roi = pg.ROI([-8, 14], [6, 5])
roi.addScaleHandle([0.5, 1], [0.5, 0.5])
roi.addScaleHandle([0, 0.5], [0.5, 0.5])
p1.addItem(roi)
roi.setZValue(10)
iso = pg.IsocurveItem(level=0.8, pen='g')
iso.setParentItem(img)
iso.setZValue(5)
hist = pg.HistogramLUTItem()
hist.setImageItem(img)
win.addItem(hist)
isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
hist.vb.addItem(isoLine)
hist.vb.setMouseEnabled(y=False)
isoLine.setValue(0.8)
isoLine.setZValue(1000)
win.nextRow()
p2 = win.addPlot(colspan=2)
p2.setMaximumHeight(250)
win.resize(800, 800)
win.show()
data = np.random.normal(size=(200, 100))
data[20:80, 20:80] += 2.0
data = pg.gaussianFilter(data, (3, 3))
data += np.random.normal(size=(200, 100)) * 0.1
img.setImage(data)
hist.setLevels(data.min(), data.max())
iso.setData(pg.gaussianFilter(data, (2, 2)))
tr = QtGui.QTransform()
img.setTransform(tr.scale(0.2, 0.2).translate(-50, 0))
p1.autoRange()

def updatePlot():
    if False:
        return 10
    global img, roi, data, p2
    selected = roi.getArrayRegion(data, img)
    p2.plot(selected.mean(axis=0), clear=True)
roi.sigRegionChanged.connect(updatePlot)
updatePlot()

def updateIsocurve():
    if False:
        return 10
    global isoLine, iso
    iso.setLevel(isoLine.value())
isoLine.sigDragged.connect(updateIsocurve)

def imageHoverEvent(event):
    if False:
        i = 10
        return i + 15
    'Show the position, pixel, and value under the mouse cursor.\n    '
    if event.isExit():
        p1.setTitle('')
        return
    pos = event.pos()
    (i, j) = (pos.y(), pos.x())
    i = int(np.clip(i, 0, data.shape[0] - 1))
    j = int(np.clip(j, 0, data.shape[1] - 1))
    val = data[i, j]
    ppos = img.mapToParent(pos)
    (x, y) = (ppos.x(), ppos.y())
    p1.setTitle('pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %.3g' % (x, y, i, j, val))
img.hoverEvent = imageHoverEvent
if __name__ == '__main__':
    pg.exec()