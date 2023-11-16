import pytest
from panda3d import core
egg = pytest.importorskip('panda3d.egg')

def read_egg_vertex(string):
    if False:
        i = 10
        return i + 15
    'Reads an EggVertex from a string.'
    data = '<VertexPool> pool { <Vertex> 1 { %s } }' % string
    stream = core.StringStream(data.encode('utf-8'))
    data = egg.EggData()
    assert data.read(stream)
    (pool,) = data.get_children()
    return pool.get_vertex(1)

def test_egg_vertex_uv_empty():
    if False:
        i = 10
        return i + 15
    vertex = read_egg_vertex('\n        0 0 0\n        <UV> {\n            0 0\n        }\n    ')
    obj = vertex.get_uv_obj('')
    assert not obj.has_tangent()
    assert not obj.has_tangent4()
    assert '<Tangent>' not in str(obj)

def test_egg_vertex_tangent():
    if False:
        print('Hello World!')
    vertex = read_egg_vertex('\n        0 0 0\n        <UV> {\n            0 0\n            <Tangent> { 2 3 4 }\n        }\n    ')
    obj = vertex.get_uv_obj('')
    assert obj.has_tangent()
    assert not obj.has_tangent4()
    assert obj.get_tangent() == (2, 3, 4)
    assert obj.get_tangent4() == (2, 3, 4, 1)
    assert '{ 2 3 4 }' in str(obj)

def test_egg_vertex_tangent4_pos():
    if False:
        return 10
    vertex = read_egg_vertex('\n        0 0 0\n        <UV> {\n            0 0\n            <Tangent> { 2 3 4 1 }\n        }\n    ')
    obj = vertex.get_uv_obj('')
    assert obj.has_tangent()
    assert obj.has_tangent4()
    assert obj.get_tangent() == (2, 3, 4)
    assert obj.get_tangent4() == (2, 3, 4, 1)
    assert '{ 2 3 4 1 }' in str(obj)

def test_egg_vertex_tangent4_neg():
    if False:
        print('Hello World!')
    vertex = read_egg_vertex('\n        0 0 0\n        <UV> {\n            0 0\n            <Tangent> { 2 3 4 -1 }\n        }\n    ')
    obj = vertex.get_uv_obj('')
    assert obj.has_tangent()
    assert obj.has_tangent4()
    assert obj.get_tangent() == (2, 3, 4)
    assert obj.get_tangent4() == (2, 3, 4, -1)
    assert '{ 2 3 4 -1 }' in str(obj)