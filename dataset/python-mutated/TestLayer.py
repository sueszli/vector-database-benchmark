from cura.Layer import Layer
from unittest.mock import MagicMock

def test_lineMeshVertexCount():
    if False:
        print('Hello World!')
    layer = Layer(1)
    layer_polygon = MagicMock()
    layer_polygon.lineMeshVertexCount = MagicMock(return_value=9001)
    layer.polygons.append(layer_polygon)
    assert layer.lineMeshVertexCount() == 9001

def test_lineMeshElementCount():
    if False:
        return 10
    layer = Layer(1)
    layer_polygon = MagicMock()
    layer_polygon.lineMeshElementCount = MagicMock(return_value=9001)
    layer.polygons.append(layer_polygon)
    assert layer.lineMeshElementCount() == 9001

def test_getAndSet():
    if False:
        while True:
            i = 10
    layer = Layer(0)
    layer.setThickness(12)
    assert layer.thickness == 12
    layer.setHeight(0.1)
    assert layer.height == 0.1

def test_elementCount():
    if False:
        while True:
            i = 10
    layer = Layer(1)
    layer_polygon = MagicMock()
    layer_polygon.lineMeshElementCount = MagicMock(return_value=9002)
    layer_polygon.lineMeshVertexCount = MagicMock(return_value=9001)
    layer_polygon.elementCount = 12
    layer.polygons.append(layer_polygon)
    assert layer.build(0, 0, [], [], [], [], [], [], []) == (9001, 9002)
    assert layer.elementCount == 12