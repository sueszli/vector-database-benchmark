import faulthandler
import weakref
import pyqtgraph as pg
faulthandler.enable()
pg.mkQApp()

def test_getViewWidget():
    if False:
        print('Hello World!')
    view = pg.PlotWidget()
    vref = weakref.ref(view)
    item = pg.InfiniteLine()
    view.addItem(item)
    assert item.getViewWidget() is view
    del view
    assert vref() is None
    assert item.getViewWidget() is None

def test_getViewWidget_deleted():
    if False:
        print('Hello World!')
    view = pg.PlotWidget()
    item = pg.InfiniteLine()
    view.addItem(item)
    assert item.getViewWidget() is view
    obj = pg.QtWidgets.QWidget()
    view.setParent(obj)
    del obj
    assert not pg.Qt.isQObjectAlive(view)
    assert item.getViewWidget() is None