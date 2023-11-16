import pytest
Pmw = pytest.importorskip('Pmw')
from direct.showbase.ShowBase import ShowBase
from direct.tkpanels.Placer import Placer

def test_Placer(window, tk_toplevel):
    if False:
        print('Hello World!')
    base = ShowBase()
    base.start_direct()
    root = Pmw.initialise()
    widget = Placer()
    base.destroy()