import pytest
import pyqtgraph as pg
pg.mkQApp()
import pyqtgraph.dockarea as da

def test_dock():
    if False:
        i = 10
        return i + 15
    name = 'évènts_zàhéér'
    dock = da.Dock(name=name)
    assert dock.name() == name
    assert type(dock.name()) == type(name)

def test_closable_dock():
    if False:
        while True:
            i = 10
    name = 'Test close dock'
    dock = da.Dock(name=name, closable=True)
    assert dock.label.closeButton is not None

def test_hide_title_dock():
    if False:
        return 10
    name = 'Test hide title dock'
    dock = da.Dock(name=name, hideTitle=True)
    assert dock.labelHidden == True

def test_close():
    if False:
        return 10
    name = 'Test close dock'
    dock = da.Dock(name=name, hideTitle=True)
    with pytest.warns(Warning):
        dock.close()