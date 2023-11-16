import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

def test_qmenu_leak_workaround():
    if False:
        while True:
            i = 10
    pg.mkQApp()
    topmenu = QtWidgets.QMenu()
    submenu = QtWidgets.QMenu()
    refcnt1 = sys.getrefcount(submenu)
    topmenu.addMenu(submenu)
    submenu.setParent(None)
    refcnt2 = sys.getrefcount(submenu)
    assert refcnt2 == refcnt1
    del topmenu
    assert pg.Qt.isQObjectAlive(submenu)