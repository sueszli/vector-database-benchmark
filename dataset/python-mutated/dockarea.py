"""
This example demonstrates the use of pyqtgraph's dock widget system.

The dockarea system allows the design of user interfaces which can be rearranged by
the user at runtime. Docks can be moved, resized, stacked, and torn out of the main
window. This is similar in principle to the docking system built into Qt, but 
offers a more deterministic dock placement API (in Qt it is very difficult to 
programatically generate complex dock arrangements). Additionally, Qt's docks are 
designed to be used as small panels around the outer edge of a window. Pyqtgraph's 
docks were created with the notion that the entire window (or any portion of it) 
would consist of dockable components.
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets
app = pg.mkQApp('DockArea Example')
win = QtWidgets.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1000, 500)
win.setWindowTitle('pyqtgraph example: dockarea')
d1 = Dock('Dock1', size=(1, 1))
d2 = Dock('Dock2 - Console', size=(500, 300), closable=True)
d3 = Dock('Dock3', size=(500, 400))
d4 = Dock('Dock4 (tabbed) - Plot', size=(500, 200))
d5 = Dock('Dock5 - Image', size=(500, 200))
d6 = Dock('Dock6 (tabbed) - Plot', size=(500, 200))
area.addDock(d1, 'left')
area.addDock(d2, 'right')
area.addDock(d3, 'bottom', d1)
area.addDock(d4, 'right')
area.addDock(d5, 'left', d1)
area.addDock(d6, 'top', d4)
area.moveDock(d4, 'top', d2)
area.moveDock(d6, 'above', d4)
area.moveDock(d5, 'top', d2)
w1 = pg.LayoutWidget()
label = QtWidgets.QLabel(' -- DockArea Example -- \nThis window has 6 Dock widgets in it. Each dock can be dragged\nby its title bar to occupy a different space within the window \nbut note that one dock has its title bar hidden). Additionally,\nthe borders between docks may be dragged to resize. Docks that are dragged on top\nof one another are stacked in a tabbed layout. Double-click a dock title\nbar to place it in its own window.\n')
saveBtn = QtWidgets.QPushButton('Save dock state')
restoreBtn = QtWidgets.QPushButton('Restore dock state')
restoreBtn.setEnabled(False)
w1.addWidget(label, row=0, col=0)
w1.addWidget(saveBtn, row=1, col=0)
w1.addWidget(restoreBtn, row=2, col=0)
d1.addWidget(w1)
state = None

def save():
    if False:
        print('Hello World!')
    global state
    state = area.saveState()
    restoreBtn.setEnabled(True)

def load():
    if False:
        i = 10
        return i + 15
    global state
    area.restoreState(state)
saveBtn.clicked.connect(save)
restoreBtn.clicked.connect(load)
w2 = ConsoleWidget()
d2.addWidget(w2)
d3.hideTitleBar()
w3 = pg.PlotWidget(title='Plot inside dock with no title bar')
w3.plot(np.random.normal(size=100))
d3.addWidget(w3)
w4 = pg.PlotWidget(title='Dock 4 plot')
w4.plot(np.random.normal(size=100))
d4.addWidget(w4)
w5 = pg.ImageView()
w5.setImage(np.random.normal(size=(100, 100)))
d5.addWidget(w5)
w6 = pg.PlotWidget(title='Dock 6 plot')
w6.plot(np.random.normal(size=100))
d6.addWidget(w6)
win.show()
if __name__ == '__main__':
    pg.exec()