import pytest
import networkx as nx

def test_valid_degree_sequence1():
    if False:
        return 10
    n = 100
    p = 0.3
    for i in range(10):
        G = nx.erdos_renyi_graph(n, p)
        deg = (d for (n, d) in G.degree())
        assert nx.is_graphical(deg, method='eg')
        assert nx.is_graphical(deg, method='hh')

def test_valid_degree_sequence2():
    if False:
        return 10
    n = 100
    for i in range(10):
        G = nx.barabasi_albert_graph(n, 1)
        deg = (d for (n, d) in G.degree())
        assert nx.is_graphical(deg, method='eg')
        assert nx.is_graphical(deg, method='hh')

def test_string_input():
    if False:
        i = 10
        return i + 15
    pytest.raises(nx.NetworkXException, nx.is_graphical, [], 'foo')
    pytest.raises(nx.NetworkXException, nx.is_graphical, ['red'], 'hh')
    pytest.raises(nx.NetworkXException, nx.is_graphical, ['red'], 'eg')

def test_non_integer_input():
    if False:
        for i in range(10):
            print('nop')
    pytest.raises(nx.NetworkXException, nx.is_graphical, [72.5], 'eg')
    pytest.raises(nx.NetworkXException, nx.is_graphical, [72.5], 'hh')

def test_negative_input():
    if False:
        while True:
            i = 10
    assert not nx.is_graphical([-1], 'hh')
    assert not nx.is_graphical([-1], 'eg')

class TestAtlas:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        global atlas
        from networkx.generators import atlas
        cls.GAG = atlas.graph_atlas_g()

    def test_atlas(self):
        if False:
            for i in range(10):
                print('nop')
        for graph in self.GAG:
            deg = (d for (n, d) in graph.degree())
            assert nx.is_graphical(deg, method='eg')
            assert nx.is_graphical(deg, method='hh')

def test_small_graph_true():
    if False:
        i = 10
        return i + 15
    z = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    assert nx.is_graphical(z, method='hh')
    assert nx.is_graphical(z, method='eg')
    z = [10, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]
    assert nx.is_graphical(z, method='hh')
    assert nx.is_graphical(z, method='eg')
    z = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    assert nx.is_graphical(z, method='hh')
    assert nx.is_graphical(z, method='eg')

def test_small_graph_false():
    if False:
        i = 10
        return i + 15
    z = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    assert not nx.is_graphical(z, method='hh')
    assert not nx.is_graphical(z, method='eg')
    z = [6, 5, 4, 4, 2, 1, 1, 1]
    assert not nx.is_graphical(z, method='hh')
    assert not nx.is_graphical(z, method='eg')
    z = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    assert not nx.is_graphical(z, method='hh')
    assert not nx.is_graphical(z, method='eg')

def test_directed_degree_sequence():
    if False:
        i = 10
        return i + 15
    (n, r) = (100, 10)
    p = 1.0 / r
    for i in range(r):
        G = nx.erdos_renyi_graph(n, p * (i + 1), None, True)
        din = (d for (n, d) in G.in_degree())
        dout = (d for (n, d) in G.out_degree())
        assert nx.is_digraphical(din, dout)

def test_small_directed_sequences():
    if False:
        i = 10
        return i + 15
    dout = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    din = [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1]
    assert nx.is_digraphical(din, dout)
    dout = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    din = [103, 102, 102, 102, 102, 102, 102, 102, 102, 102]
    assert not nx.is_digraphical(din, dout)
    dout = [1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    din = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
    assert nx.is_digraphical(din, dout)
    din = [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1]
    assert not nx.is_digraphical(din, dout)
    din = [2, 2, 2, -2, 2, 2, 2, 2, 1, 1, 4]
    assert not nx.is_digraphical(din, dout)
    din = dout = [1, 1, 1.1, 1]
    assert not nx.is_digraphical(din, dout)
    din = dout = [1, 1, 'rer', 1]
    assert not nx.is_digraphical(din, dout)

def test_multi_sequence():
    if False:
        for i in range(10):
            print('nop')
    seq = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1]
    assert not nx.is_multigraphical(seq)
    seq = [6, 5, 4, 4, 2, 1, 1, 1]
    assert nx.is_multigraphical(seq)
    seq = [6, 5, 4, -4, 2, 1, 1, 1]
    assert not nx.is_multigraphical(seq)
    seq = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4]
    assert not nx.is_multigraphical(seq)
    seq = [1, 1, 1.1, 1]
    assert not nx.is_multigraphical(seq)
    seq = [1, 1, 'rer', 1]
    assert not nx.is_multigraphical(seq)

def test_pseudo_sequence():
    if False:
        print('Hello World!')
    seq = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1]
    assert nx.is_pseudographical(seq)
    seq = [1000, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    assert not nx.is_pseudographical(seq)
    seq = [1000, 3, 3, 3, 3, 2, 2, -2, 1, 1]
    assert not nx.is_pseudographical(seq)
    seq = [1, 1, 1.1, 1]
    assert not nx.is_pseudographical(seq)
    seq = [1, 1, 'rer', 1]
    assert not nx.is_pseudographical(seq)

def test_numpy_degree_sequence():
    if False:
        i = 10
        return i + 15
    np = pytest.importorskip('numpy')
    ds = np.array([1, 2, 2, 2, 1], dtype=np.int64)
    assert nx.is_graphical(ds, 'eg')
    assert nx.is_graphical(ds, 'hh')
    ds = np.array([1, 2, 2, 2, 1], dtype=np.float64)
    assert nx.is_graphical(ds, 'eg')
    assert nx.is_graphical(ds, 'hh')
    ds = np.array([1.1, 2, 2, 2, 1], dtype=np.float64)
    pytest.raises(nx.NetworkXException, nx.is_graphical, ds, 'eg')
    pytest.raises(nx.NetworkXException, nx.is_graphical, ds, 'hh')