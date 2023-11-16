import numpy as np
from ... import ComboBox, PlotDataItem
from ...graphicsItems.ScatterPlotItem import ScatterPlotItem
from ...Qt import QtCore, QtGui, QtWidgets
from ..Node import Node
from .common import CtrlNode

class PlotWidgetNode(Node):
    """Connection to PlotWidget. Will plot arrays, metaarrays, and display event lists."""
    nodeName = 'PlotWidget'
    sigPlotChanged = QtCore.Signal(object)

    def __init__(self, name):
        if False:
            while True:
                i = 10
        Node.__init__(self, name, terminals={'In': {'io': 'in', 'multi': True}})
        self.plot = None
        self.plots = {}
        self.ui = None
        self.items = {}

    def disconnected(self, localTerm, remoteTerm):
        if False:
            for i in range(10):
                print('nop')
        if localTerm is self['In'] and remoteTerm in self.items:
            self.plot.removeItem(self.items[remoteTerm])
            del self.items[remoteTerm]

    def setPlot(self, plot):
        if False:
            return 10
        if plot == self.plot:
            return
        if self.plot is not None:
            for vid in list(self.items.keys()):
                self.plot.removeItem(self.items[vid])
                del self.items[vid]
        self.plot = plot
        self.updateUi()
        self.update()
        self.sigPlotChanged.emit(self)

    def getPlot(self):
        if False:
            i = 10
            return i + 15
        return self.plot

    def process(self, In, display=True):
        if False:
            return 10
        if display and self.plot is not None:
            items = set()
            for (name, vals) in In.items():
                if vals is None:
                    continue
                if type(vals) is not list:
                    vals = [vals]
                for val in vals:
                    vid = id(val)
                    if vid in self.items and self.items[vid].scene() is self.plot.scene():
                        items.add(vid)
                    else:
                        if isinstance(val, QtWidgets.QGraphicsItem):
                            self.plot.addItem(val)
                            item = val
                        else:
                            item = self.plot.plot(val)
                        self.items[vid] = item
                        items.add(vid)
            for vid in list(self.items.keys()):
                if vid not in items:
                    self.plot.removeItem(self.items[vid])
                    del self.items[vid]

    def processBypassed(self, args):
        if False:
            while True:
                i = 10
        if self.plot is None:
            return
        for item in list(self.items.values()):
            self.plot.removeItem(item)
        self.items = {}

    def ctrlWidget(self):
        if False:
            while True:
                i = 10
        if self.ui is None:
            self.ui = ComboBox()
            self.ui.currentIndexChanged.connect(self.plotSelected)
            self.updateUi()
        return self.ui

    def plotSelected(self, index):
        if False:
            print('Hello World!')
        self.setPlot(self.ui.value())

    def setPlotList(self, plots):
        if False:
            for i in range(10):
                print('nop')
        '\n        Specify the set of plots (PlotWidget or PlotItem) that the user may\n        select from.\n        \n        *plots* must be a dictionary of {name: plot} pairs.\n        '
        self.plots = plots
        self.updateUi()

    def updateUi(self):
        if False:
            while True:
                i = 10
        self.ui.setItems(self.plots)
        try:
            self.ui.setValue(self.plot)
        except ValueError:
            pass

class CanvasNode(Node):
    """Connection to a Canvas widget."""
    nodeName = 'CanvasWidget'

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self, name, terminals={'In': {'io': 'in', 'multi': True}})
        self.canvas = None
        self.items = {}

    def disconnected(self, localTerm, remoteTerm):
        if False:
            for i in range(10):
                print('nop')
        if localTerm is self.In and remoteTerm in self.items:
            self.canvas.removeItem(self.items[remoteTerm])
            del self.items[remoteTerm]

    def setCanvas(self, canvas):
        if False:
            for i in range(10):
                print('nop')
        self.canvas = canvas

    def getCanvas(self):
        if False:
            i = 10
            return i + 15
        return self.canvas

    def process(self, In, display=True):
        if False:
            print('Hello World!')
        if display:
            items = set()
            for (name, vals) in In.items():
                if vals is None:
                    continue
                if type(vals) is not list:
                    vals = [vals]
                for val in vals:
                    vid = id(val)
                    if vid in self.items:
                        items.add(vid)
                    else:
                        self.canvas.addItem(val)
                        item = val
                        self.items[vid] = item
                        items.add(vid)
            for vid in list(self.items.keys()):
                if vid not in items:
                    self.canvas.removeItem(self.items[vid])
                    del self.items[vid]

class PlotCurve(CtrlNode):
    """Generates a plot curve from x/y data"""
    nodeName = 'PlotCurve'
    uiTemplate = [('color', 'color')]

    def __init__(self, name):
        if False:
            while True:
                i = 10
        CtrlNode.__init__(self, name, terminals={'x': {'io': 'in'}, 'y': {'io': 'in'}, 'plot': {'io': 'out'}})
        self.item = PlotDataItem()

    def process(self, x, y, display=True):
        if False:
            for i in range(10):
                print('nop')
        if not display:
            return {'plot': None}
        self.item.setData(x, y, pen=self.ctrls['color'].color())
        return {'plot': self.item}

class ScatterPlot(CtrlNode):
    """Generates a scatter plot from a record array or nested dicts"""
    nodeName = 'ScatterPlot'
    uiTemplate = [('x', 'combo', {'values': [], 'index': 0}), ('y', 'combo', {'values': [], 'index': 0}), ('sizeEnabled', 'check', {'value': False}), ('size', 'combo', {'values': [], 'index': 0}), ('absoluteSize', 'check', {'value': False}), ('colorEnabled', 'check', {'value': False}), ('color', 'colormap', {}), ('borderEnabled', 'check', {'value': False}), ('border', 'colormap', {})]

    def __init__(self, name):
        if False:
            print('Hello World!')
        CtrlNode.__init__(self, name, terminals={'input': {'io': 'in'}, 'plot': {'io': 'out'}})
        self.item = ScatterPlotItem()
        self.keys = []

    def process(self, input, display=True):
        if False:
            print('Hello World!')
        if not display:
            return {'plot': None}
        self.updateKeys(input[0])
        x = str(self.ctrls['x'].currentText())
        y = str(self.ctrls['y'].currentText())
        size = str(self.ctrls['size'].currentText())
        pen = QtGui.QPen(QtGui.QColor(0, 0, 0, 0))
        points = []
        for i in input:
            pt = {'pos': (i[x], i[y])}
            if self.ctrls['sizeEnabled'].isChecked():
                pt['size'] = i[size]
            if self.ctrls['borderEnabled'].isChecked():
                pt['pen'] = QtGui.QPen(self.ctrls['border'].getColor(i))
            else:
                pt['pen'] = pen
            if self.ctrls['colorEnabled'].isChecked():
                pt['brush'] = QtGui.QBrush(self.ctrls['color'].getColor(i))
            points.append(pt)
        self.item.setPxMode(not self.ctrls['absoluteSize'].isChecked())
        self.item.setPoints(points)
        return {'plot': self.item}

    def updateKeys(self, data):
        if False:
            return 10
        if isinstance(data, dict):
            keys = list(data.keys())
        elif isinstance(data, list) or isinstance(data, tuple):
            keys = data
        elif isinstance(data, np.ndarray) or isinstance(data, np.void):
            keys = data.dtype.names
        else:
            print('Unknown data type:', type(data), data)
            return
        for c in self.ctrls.values():
            c.blockSignals(True)
        for c in [self.ctrls['x'], self.ctrls['y'], self.ctrls['size']]:
            cur = str(c.currentText())
            c.clear()
            for k in keys:
                c.addItem(k)
                if k == cur:
                    c.setCurrentIndex(c.count() - 1)
        for c in [self.ctrls['color'], self.ctrls['border']]:
            c.setArgList(keys)
        for c in self.ctrls.values():
            c.blockSignals(False)
        self.keys = keys

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        state = CtrlNode.saveState(self)
        return {'keys': self.keys, 'ctrls': state}

    def restoreState(self, state):
        if False:
            while True:
                i = 10
        self.updateKeys(state['keys'])
        CtrlNode.restoreState(self, state['ctrls'])