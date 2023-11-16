import pytest
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

def test_Vector_init():
    if False:
        for i in range(10):
            print('nop')
    'Test construction of Vector objects from a variety of source types.'
    v = pg.Vector(0, 1)
    assert v.z() == 0
    v = pg.Vector(0.0, 1.0)
    assert v.z() == 0
    v = pg.Vector(0, 1, 2)
    assert v.x() == 0
    assert v.y() == 1
    assert v.z() == 2
    v = pg.Vector(0.0, 1.0, 2.0)
    assert v.x() == 0
    assert v.y() == 1
    assert v.z() == 2
    v = pg.Vector([0, 1])
    assert v.z() == 0
    v = pg.Vector([0, 1, 2])
    assert v.z() == 2
    v = pg.Vector(QtCore.QSizeF(1, 2))
    assert v.x() == 1
    assert v.z() == 0
    v = pg.Vector(QtCore.QPoint(0, 1))
    assert v.z() == 0
    v = pg.Vector(QtCore.QPointF(0, 1))
    assert v.z() == 0
    qv = QtGui.QVector3D(1, 2, 3)
    v = pg.Vector(qv)
    assert v == qv
    with pytest.raises(Exception):
        _ = pg.Vector(1, 2, 3, 4)

def test_Vector_interface():
    if False:
        print('Hello World!')
    'Test various aspects of the Vector API.'
    v = pg.Vector(-1, 2)
    assert len(v) == 3
    assert v[0] == -1
    assert v[2] == 0
    with pytest.raises(IndexError):
        _ = v[4]
    assert v[1] == 2
    v[1] = 5
    assert v[1] == 5
    v2 = pg.Vector(*v)
    assert v2 == v
    assert abs(v).x() == 1
    v1 = pg.Vector(1, 0)
    v2 = pg.Vector(1, 1)
    assert abs(v1.angle(v2) - 45) < 0.001