from time import perf_counter
import numpy as np
import pyqtgraph as pg
app = pg.mkQApp()
plt = pg.PlotWidget()
app.processEvents()
plt.show()
plt.enableAutoRange(False, False)

def plot():
    if False:
        i = 10
        return i + 15
    start = perf_counter()
    n = 15
    pts = 100
    x = np.linspace(0, 0.8, pts)
    y = np.random.random(size=pts) * 0.8
    for i in range(n):
        for j in range(n):
            plt.addItem(pg.PlotCurveItem(x=x + i, y=y + j))
    dt = perf_counter() - start
    print(f'Create plots took: {dt * 1000:.3f} ms')
for _ in range(5):
    plt.clear()
    plot()
    app.processEvents()
    plt.autoRange()

def fastPlot():
    if False:
        return 10
    start = perf_counter()
    n = 15
    pts = 100
    x = np.linspace(0, 0.8, pts)
    y = np.random.random(size=pts) * 0.8
    shape = (n, n, pts)
    xdata = np.empty(shape)
    xdata[:] = x + np.arange(shape[1]).reshape((1, -1, 1))
    ydata = np.empty(shape)
    ydata[:] = y + np.arange(shape[0]).reshape((-1, 1, 1))
    conn = np.ones(shape, dtype=bool)
    conn[..., -1] = False
    item = pg.PlotCurveItem()
    item.setData(xdata.ravel(), ydata.ravel(), connect=conn.ravel())
    plt.addItem(item)
    dt = perf_counter() - start
    print('Create plots took: %0.3fms' % (dt * 1000))
for _ in range(5):
    plt.clear()
    fastPlot()
    app.processEvents()
    plt.autoRange()
if __name__ == '__main__':
    pg.exec()