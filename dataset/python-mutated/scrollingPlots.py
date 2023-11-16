"""
Various methods of drawing scrolling plots.
"""
from time import perf_counter
import numpy as np
import pyqtgraph as pg
win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('pyqtgraph example: Scrolling Plots')
p1 = win.addPlot()
p2 = win.addPlot()
data1 = np.random.normal(size=300)
curve1 = p1.plot(data1)
curve2 = p2.plot(data1)
ptr1 = 0

def update1():
    if False:
        while True:
            i = 10
    global data1, ptr1
    data1[:-1] = data1[1:]
    data1[-1] = np.random.normal()
    curve1.setData(data1)
    ptr1 += 1
    curve2.setData(data1)
    curve2.setPos(ptr1, 0)
win.nextRow()
p3 = win.addPlot()
p4 = win.addPlot()
p3.setDownsampling(mode='peak')
p4.setDownsampling(mode='peak')
p3.setClipToView(True)
p4.setClipToView(True)
p3.setRange(xRange=[-100, 0])
p3.setLimits(xMax=0)
curve3 = p3.plot()
curve4 = p4.plot()
data3 = np.empty(100)
ptr3 = 0

def update2():
    if False:
        while True:
            i = 10
    global data3, ptr3
    data3[ptr3] = np.random.normal()
    ptr3 += 1
    if ptr3 >= data3.shape[0]:
        tmp = data3
        data3 = np.empty(data3.shape[0] * 2)
        data3[:tmp.shape[0]] = tmp
    curve3.setData(data3[:ptr3])
    curve3.setPos(-ptr3, 0)
    curve4.setData(data3[:ptr3])
chunkSize = 100
maxChunks = 10
startTime = perf_counter()
win.nextRow()
p5 = win.addPlot(colspan=2)
p5.setLabel('bottom', 'Time', 's')
p5.setXRange(-10, 0)
curves = []
data5 = np.empty((chunkSize + 1, 2))
ptr5 = 0

def update3():
    if False:
        print('Hello World!')
    global p5, data5, ptr5, curves
    now = perf_counter()
    for c in curves:
        c.setPos(-(now - startTime), 0)
    i = ptr5 % chunkSize
    if i == 0:
        curve = p5.plot()
        curves.append(curve)
        last = data5[-1]
        data5 = np.empty((chunkSize + 1, 2))
        data5[0] = last
        while len(curves) > maxChunks:
            c = curves.pop(0)
            p5.removeItem(c)
    else:
        curve = curves[-1]
    data5[i + 1, 0] = now - startTime
    data5[i + 1, 1] = np.random.normal()
    curve.setData(x=data5[:i + 2, 0], y=data5[:i + 2, 1])
    ptr5 += 1

def update():
    if False:
        while True:
            i = 10
    update1()
    update2()
    update3()
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)
if __name__ == '__main__':
    pg.exec()