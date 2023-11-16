"""
This example demonstrates writing a custom Node subclass for use with flowcharts.

We implement a couple of simple image processing nodes.
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart import Flowchart, Node
from pyqtgraph.flowchart.library.common import CtrlNode
from pyqtgraph.Qt import QtWidgets
app = pg.mkQApp('Flowchart Custom Node Example')
win = QtWidgets.QMainWindow()
win.setWindowTitle('pyqtgraph example: FlowchartCustomNode')
cw = QtWidgets.QWidget()
win.setCentralWidget(cw)
layout = QtWidgets.QGridLayout()
cw.setLayout(layout)
fc = Flowchart(terminals={'dataIn': {'io': 'in'}, 'dataOut': {'io': 'out'}})
w = fc.widget()
layout.addWidget(fc.widget(), 0, 0, 2, 1)
v1 = pg.ImageView()
v2 = pg.ImageView()
layout.addWidget(v1, 0, 1)
layout.addWidget(v2, 1, 1)
win.show()
data = np.random.normal(size=(100, 100))
data = 25 * pg.gaussianFilter(data, (5, 5))
data += np.random.normal(size=(100, 100))
data[40:60, 40:60] += 15.0
data[30:50, 30:50] += 15.0
fc.setInput(dataIn=data)

class ImageViewNode(Node):
    """Node that displays image data in an ImageView widget"""
    nodeName = 'ImageView'

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.view = None
        Node.__init__(self, name, terminals={'data': {'io': 'in'}})

    def setView(self, view):
        if False:
            i = 10
            return i + 15
        self.view = view

    def process(self, data, display=True):
        if False:
            while True:
                i = 10
        if display and self.view is not None:
            if data is None:
                self.view.setImage(np.zeros((1, 1)))
            else:
                self.view.setImage(data)

class UnsharpMaskNode(CtrlNode):
    """Return the input data passed through an unsharp mask."""
    nodeName = 'UnsharpMask'
    uiTemplate = [('sigma', 'spin', {'value': 1.0, 'step': 1.0, 'bounds': [0.0, None]}), ('strength', 'spin', {'value': 1.0, 'dec': True, 'step': 0.5, 'minStep': 0.01, 'bounds': [0.0, None]})]

    def __init__(self, name):
        if False:
            while True:
                i = 10
        terminals = {'dataIn': dict(io='in'), 'dataOut': dict(io='out')}
        CtrlNode.__init__(self, name, terminals=terminals)

    def process(self, dataIn, display=True):
        if False:
            while True:
                i = 10
        sigma = self.ctrls['sigma'].value()
        strength = self.ctrls['strength'].value()
        output = dataIn - strength * pg.gaussianFilter(dataIn, (sigma, sigma))
        return {'dataOut': output}
library = fclib.LIBRARY.copy()
library.addNodeType(ImageViewNode, [('Display',)])
library.addNodeType(UnsharpMaskNode, [('Image',), ('Submenu_test', 'submenu2', 'submenu3')])
fc.setLibrary(library)
v1Node = fc.createNode('ImageView', pos=(0, -150))
v1Node.setView(v1)
v2Node = fc.createNode('ImageView', pos=(150, -150))
v2Node.setView(v2)
fNode = fc.createNode('UnsharpMask', pos=(0, 0))
fc.connectTerminals(fc['dataIn'], fNode['dataIn'])
fc.connectTerminals(fc['dataIn'], v1Node['data'])
fc.connectTerminals(fNode['dataOut'], v2Node['data'])
fc.connectTerminals(fNode['dataOut'], fc['dataOut'])
if __name__ == '__main__':
    pg.exec()