import pytest
pytest.importorskip('OpenGL')
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLLinePlotItem
from common import ensure_parentItem

def test_parentItem():
    if False:
        for i in range(10):
            print('nop')
    parent = GLGraphicsItem()
    child = GLLinePlotItem(parentItem=parent)
    ensure_parentItem(parent, child)