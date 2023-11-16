from panda3d.core import Triangulator

def triangulate(vertices):
    if False:
        print('Hello World!')
    t = Triangulator()
    for (i, v) in enumerate(vertices):
        t.add_vertex(v)
        t.add_polygon_vertex(i)
    t.triangulate()
    result = set()
    for n in range(t.get_num_triangles()):
        v0 = vertices.index(vertices[t.get_triangle_v0(n)])
        v1 = vertices.index(vertices[t.get_triangle_v1(n)])
        v2 = vertices.index(vertices[t.get_triangle_v2(n)])
        if v1 < v0:
            (v0, v1, v2) = (v1, v2, v0)
        if v1 < v0:
            (v0, v1, v2) = (v1, v2, v0)
        result.add((v0, v1, v2))
    return result

def test_triangulator_degenerate():
    if False:
        return 10
    assert not triangulate([])
    assert not triangulate([(0, 0)])
    assert not triangulate([(0, 0), (0, 0)])
    assert not triangulate([(0, 0), (1, 0)])
    assert not triangulate([(0, 0), (0, 0), (0, 0)])
    assert not triangulate([(0, 0), (1, 0), (1, 0)])
    assert not triangulate([(1, 0), (1, 0), (1, 0)])
    assert not triangulate([(1, 0), (0, 0), (1, 0)])
    assert not triangulate([(0, 0), (0, 0), (0, 0), (0, 0)])

def test_triangulator_triangle():
    if False:
        print('Hello World!')
    assert triangulate([(0, 0), (1, 0), (1, 1)]) == {(0, 1, 2)}

def test_triangulator_tail():
    if False:
        i = 10
        return i + 15
    assert triangulate([(0, -1), (0, 1), (1, 0), (2, 0), (3, 1), (4, 0), (5, 0), (4, 0), (3, 1), (2, 0), (1, 0)]) == {(0, 2, 1)}

def test_triangulator_hourglass():
    if False:
        for i in range(10):
            print('nop')
    assert triangulate([(-1, 1), (-1, -1), (0, 0), (1, -1), (1, 1), (0, 0)]) == {(0, 1, 2), (2, 3, 4)}