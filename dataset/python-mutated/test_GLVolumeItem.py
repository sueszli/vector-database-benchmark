import pytest
pytest.importorskip('OpenGL')
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLVolumeItem
from common import ensure_parentItem

def test_parentItem():
    if False:
        i = 10
        return i + 15
    parent = GLGraphicsItem()
    child = GLVolumeItem(None, parentItem=parent)
    ensure_parentItem(parent, child)