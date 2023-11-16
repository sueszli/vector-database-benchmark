"""
Demonstrates very basic use of ImageItem to display image data inside a ViewBox.
"""
from time import perf_counter
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
app = pg.mkQApp('ImageItem Example')
win = pg.GraphicsLayoutWidget()
win.show()
win.setWindowTitle('pyqtgraph example: ImageItem')
view = win.addViewBox()
view.setAspectLocked(True)
img = pg.ImageItem(border='w')
view.addItem(img)
view.setRange(QtCore.QRectF(0, 0, 600, 600))
data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
i = 0
updateTime = perf_counter()
elapsed = 0
timer = QtCore.QTimer()
timer.setSingleShot(True)

def updateData():
    if False:
        for i in range(10):
            print('nop')
    global img, data, i, updateTime, elapsed
    img.setImage(data[i])
    i = (i + 1) % data.shape[0]
    timer.start(1)
    now = perf_counter()
    elapsed_now = now - updateTime
    updateTime = now
    elapsed = elapsed * 0.9 + elapsed_now * 0.1
timer.timeout.connect(updateData)
updateData()
if __name__ == '__main__':
    pg.exec()