import networkx as nx

def test_unionfind():
    if False:
        print('Hello World!')
    x = nx.utils.UnionFind()
    x.union(0, 'a')

def test_subtree_union():
    if False:
        return 10
    uf = nx.utils.UnionFind()
    uf.union(1, 2)
    uf.union(3, 4)
    uf.union(4, 5)
    uf.union(1, 5)
    assert list(uf.to_sets()) == [{1, 2, 3, 4, 5}]

def test_unionfind_weights():
    if False:
        return 10
    uf = nx.utils.UnionFind()
    uf.union(1, 4, 7)
    uf.union(2, 5, 8)
    uf.union(3, 6, 9)
    uf.union(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert uf.weights[uf[1]] == 9

def test_unbalanced_merge_weights():
    if False:
        print('Hello World!')
    uf = nx.utils.UnionFind()
    uf.union(1, 2, 3)
    uf.union(4, 5, 6, 7, 8, 9)
    assert uf.weights[uf[1]] == 3
    assert uf.weights[uf[4]] == 6
    largest_root = uf[4]
    uf.union(1, 4)
    assert uf[1] == largest_root
    assert uf.weights[largest_root] == 9

def test_empty_union():
    if False:
        for i in range(10):
            print('nop')
    uf = nx.utils.UnionFind((0, 1))
    uf.union()
    assert uf[0] == 0
    assert uf[1] == 1