import pytest
pytest.importorskip('OpenGL')
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.opengl import GLMeshItem
from common import ensure_parentItem

def test_parentItem():
    if False:
        for i in range(10):
            print('nop')
    parent = GLGraphicsItem()
    child = GLMeshItem(parentItem=parent)
    ensure_parentItem(parent, child)