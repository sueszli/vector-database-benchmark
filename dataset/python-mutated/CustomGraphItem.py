"""
Simple example of subclassing GraphItem.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
pg.setConfigOptions(antialias=True)
w = pg.GraphicsLayoutWidget(show=True)
w.setWindowTitle('pyqtgraph example: CustomGraphItem')
v = w.addViewBox()
v.setAspectLocked()

class Graph(pg.GraphItem):

    def __init__(self):
        if False:
            print('Hello World!')
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)

    def setData(self, **kwds):
        if False:
            return 10
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def setTexts(self, text):
        if False:
            print('Hello World!')
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def updateGraph(self):
        if False:
            print('Hello World!')
        pg.GraphItem.setData(self, **self.data)
        for (i, item) in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def mouseDragEvent(self, ev):
        if False:
            print('Hello World!')
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.dragPoint = None
            return
        elif self.dragPoint is None:
            ev.ignore()
            return
        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        self.updateGraph()
        ev.accept()

    def clicked(self, pts):
        if False:
            for i in range(10):
                print('nop')
        print('clicked: %s' % pts)
g = Graph()
v.addItem(g)
pos = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5], [15, 5]], dtype=float)
adj = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [1, 5], [3, 5]])
symbols = ['o', 'o', 'o', 'o', 't', '+']
lines = np.array([(255, 0, 0, 255, 1), (255, 0, 255, 255, 2), (255, 0, 255, 255, 3), (255, 255, 0, 255, 2), (255, 0, 0, 255, 1), (255, 255, 255, 255, 4)], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])
texts = ['Point %d' % i for i in range(6)]
g.setData(pos=pos, adj=adj, pen=lines, size=1, symbol=symbols, pxMode=False, text=texts)
if __name__ == '__main__':
    pg.exec()