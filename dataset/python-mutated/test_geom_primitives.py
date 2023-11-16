from panda3d import core

def test_geom_tristrips():
    if False:
        return 10
    prim = core.GeomTristrips(core.GeomEnums.UH_static)
    prim.add_vertex(0)
    prim.add_vertex(1)
    prim.add_vertex(2)
    prim.add_vertex(3)
    prim.close_primitive()
    prim.add_vertex(0)
    prim.add_vertex(1)
    prim.add_vertex(2)
    prim.close_primitive()
    verts = prim.get_vertex_list()
    assert tuple(verts) == (0, 1, 2, 3, 3, 0, 0, 1, 2)

def test_geom_triangles_adjacency():
    if False:
        while True:
            i = 10
    prim = core.GeomTriangles(core.GeomEnums.UH_static)
    prim.add_vertex(0)
    prim.add_vertex(1)
    prim.add_vertex(2)
    prim.close_primitive()
    prim.add_vertex(2)
    prim.add_vertex(1)
    prim.add_vertex(3)
    prim.close_primitive()
    adj = prim.make_adjacency()
    verts = adj.get_vertex_list()
    assert tuple(verts) == (0, 0, 1, 3, 2, 2, 2, 0, 1, 1, 3, 3)

def test_geom_lines_adjacency():
    if False:
        for i in range(10):
            print('nop')
    prim = core.GeomLines(core.GeomEnums.UH_static)
    prim.add_vertex(0)
    prim.add_vertex(1)
    prim.close_primitive()
    prim.add_vertex(1)
    prim.add_vertex(2)
    prim.close_primitive()
    prim.add_vertex(2)
    prim.add_vertex(3)
    prim.close_primitive()
    adj = prim.make_adjacency()
    verts = adj.get_vertex_list()
    assert tuple(verts) == (0, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 3)

def test_geom_linestrips_adjacency():
    if False:
        while True:
            i = 10
    prim = core.GeomLinestrips(core.GeomEnums.UH_static)
    prim.add_vertex(0)
    prim.add_vertex(1)
    prim.close_primitive()
    prim.add_vertex(1)
    prim.add_vertex(2)
    prim.add_vertex(3)
    prim.close_primitive()
    prim.add_vertex(3)
    prim.add_vertex(4)
    prim.add_vertex(5)
    prim.add_vertex(6)
    prim.close_primitive()
    adj = prim.make_adjacency()
    verts = adj.get_vertex_list()
    cut = adj.get_strip_cut_index()
    assert tuple(verts) == (0, 0, 1, 2, cut, 0, 1, 2, 3, 4, cut, 2, 3, 4, 5, 6, 6)
    prim = adj.decompose()
    assert isinstance(prim, core.GeomLinesAdjacency)
    verts = prim.get_vertex_list()
    assert tuple(verts) == (0, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 6)

def test_geom_linestrips_offset_indexed():
    if False:
        print('Hello World!')
    prim = core.GeomLinestrips(core.GeomEnums.UH_static)
    prim.add_vertex(0)
    prim.add_vertex(1)
    prim.close_primitive()
    prim.add_vertex(1)
    prim.add_vertex(2)
    prim.add_vertex(3)
    prim.close_primitive()
    prim.add_vertex(3)
    prim.add_vertex(4)
    prim.add_vertex(5)
    prim.add_vertex(6)
    prim.close_primitive()
    prim.offset_vertices(100)
    verts = prim.get_vertex_list()
    cut = prim.strip_cut_index
    assert tuple(verts) == (100, 101, cut, 101, 102, 103, cut, 103, 104, 105, 106)
    prim.offset_vertices(-100)
    verts = prim.get_vertex_list()
    cut = prim.strip_cut_index
    assert tuple(verts) == (0, 1, cut, 1, 2, 3, cut, 3, 4, 5, 6)
    prim.offset_vertices(100, 4, 9)
    verts = prim.get_vertex_list()
    cut = prim.strip_cut_index
    assert tuple(verts) == (0, 1, cut, 1, 102, 103, cut, 103, 104, 5, 6)
    prim.offset_vertices(100000)
    assert prim.index_type == core.GeomEnums.NT_uint32
    verts = prim.get_vertex_list()
    cut = prim.strip_cut_index
    assert tuple(verts) == (100000, 100001, cut, 100001, 100102, 100103, cut, 100103, 100104, 100005, 100006)