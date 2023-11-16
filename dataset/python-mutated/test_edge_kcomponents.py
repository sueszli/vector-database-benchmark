import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise

def fset(list_of_sets):
    if False:
        i = 10
        return i + 15
    'allows == to be used for list of sets'
    return set(map(frozenset, list_of_sets))

def _assert_subgraph_edge_connectivity(G, ccs_subgraph, k):
    if False:
        i = 10
        return i + 15
    '\n    tests properties of k-edge-connected subgraphs\n\n    the actual edge connectivity should be no less than k unless the cc is a\n    single node.\n    '
    for cc in ccs_subgraph:
        C = G.subgraph(cc)
        if len(cc) > 1:
            connectivity = nx.edge_connectivity(C)
            assert connectivity >= k

def _memo_connectivity(G, u, v, memo):
    if False:
        for i in range(10):
            print('nop')
    edge = (u, v)
    if edge in memo:
        return memo[edge]
    if not G.is_directed():
        redge = (v, u)
        if redge in memo:
            return memo[redge]
    memo[edge] = nx.edge_connectivity(G, *edge)
    return memo[edge]

def _all_pairs_connectivity(G, cc, k, memo):
    if False:
        i = 10
        return i + 15
    for (u, v) in it.combinations(cc, 2):
        connectivity = _memo_connectivity(G, u, v, memo)
        if G.is_directed():
            connectivity = min(connectivity, _memo_connectivity(G, v, u, memo))
        assert connectivity >= k

def _assert_local_cc_edge_connectivity(G, ccs_local, k, memo):
    if False:
        while True:
            i = 10
    '\n    tests properties of k-edge-connected components\n\n    the local edge connectivity between each pair of nodes in the original\n    graph should be no less than k unless the cc is a single node.\n    '
    for cc in ccs_local:
        if len(cc) > 1:
            C = G.subgraph(cc)
            connectivity = nx.edge_connectivity(C)
            if connectivity < k:
                _all_pairs_connectivity(G, cc, k, memo)

def _check_edge_connectivity(G):
    if False:
        while True:
            i = 10
    '\n    Helper - generates all k-edge-components using the aux graph.  Checks the\n    both local and subgraph edge connectivity of each cc. Also checks that\n    alternate methods of computing the k-edge-ccs generate the same result.\n    '
    aux_graph = EdgeComponentAuxGraph.construct(G)
    memo = {}
    for k in it.count(1):
        ccs_local = fset(aux_graph.k_edge_components(k))
        ccs_subgraph = fset(aux_graph.k_edge_subgraphs(k))
        _assert_local_cc_edge_connectivity(G, ccs_local, k, memo)
        _assert_subgraph_edge_connectivity(G, ccs_subgraph, k)
        if k == 1 or (k == 2 and (not G.is_directed())):
            assert ccs_local == ccs_subgraph, 'Subgraphs and components should be the same when k == 1 or (k == 2 and not G.directed())'
        if G.is_directed():
            if k == 1:
                alt_sccs = fset(nx.strongly_connected_components(G))
                assert alt_sccs == ccs_local, 'k=1 failed alt'
                assert alt_sccs == ccs_subgraph, 'k=1 failed alt'
        elif k == 1:
            alt_ccs = fset(nx.connected_components(G))
            assert alt_ccs == ccs_local, 'k=1 failed alt'
            assert alt_ccs == ccs_subgraph, 'k=1 failed alt'
        elif k == 2:
            alt_bridge_ccs = fset(bridge_components(G))
            assert alt_bridge_ccs == ccs_local, 'k=2 failed alt'
            assert alt_bridge_ccs == ccs_subgraph, 'k=2 failed alt'
        alt_subgraph_ccs = fset([set(C.nodes()) for C in general_k_edge_subgraphs(G, k=k)])
        assert alt_subgraph_ccs == ccs_subgraph, 'alt subgraph method failed'
        if k > 2 and all((len(cc) == 1 for cc in ccs_local)):
            break

def test_zero_k_exception():
    if False:
        for i in range(10):
            print('nop')
    G = nx.Graph()
    pytest.raises(ValueError, nx.k_edge_components, G, k=0)
    pytest.raises(ValueError, nx.k_edge_subgraphs, G, k=0)
    aux_graph = EdgeComponentAuxGraph.construct(G)
    pytest.raises(ValueError, list, aux_graph.k_edge_components(k=0))
    pytest.raises(ValueError, list, aux_graph.k_edge_subgraphs(k=0))
    pytest.raises(ValueError, list, general_k_edge_subgraphs(G, k=0))

def test_empty_input():
    if False:
        print('Hello World!')
    G = nx.Graph()
    assert [] == list(nx.k_edge_components(G, k=5))
    assert [] == list(nx.k_edge_subgraphs(G, k=5))
    G = nx.DiGraph()
    assert [] == list(nx.k_edge_components(G, k=5))
    assert [] == list(nx.k_edge_subgraphs(G, k=5))

def test_not_implemented():
    if False:
        return 10
    G = nx.MultiGraph()
    pytest.raises(nx.NetworkXNotImplemented, EdgeComponentAuxGraph.construct, G)
    pytest.raises(nx.NetworkXNotImplemented, nx.k_edge_components, G, k=2)
    pytest.raises(nx.NetworkXNotImplemented, nx.k_edge_subgraphs, G, k=2)
    with pytest.raises(nx.NetworkXNotImplemented):
        next(bridge_components(G))
    with pytest.raises(nx.NetworkXNotImplemented):
        next(bridge_components(nx.DiGraph()))

def test_general_k_edge_subgraph_quick_return():
    if False:
        while True:
            i = 10
    G = nx.Graph()
    G.add_node(0)
    subgraphs = list(general_k_edge_subgraphs(G, k=1))
    assert len(subgraphs) == 1
    for subgraph in subgraphs:
        assert subgraph.number_of_nodes() == 1
    G.add_node(1)
    subgraphs = list(general_k_edge_subgraphs(G, k=1))
    assert len(subgraphs) == 2
    for subgraph in subgraphs:
        assert subgraph.number_of_nodes() == 1

def test_random_gnp():
    if False:
        for i in range(10):
            print('nop')
    seeds = [12, 13]
    for seed in seeds:
        G = nx.gnp_random_graph(20, 0.2, seed=seed)
        _check_edge_connectivity(G)

def test_configuration():
    if False:
        print('Hello World!')
    seeds = [14, 15]
    for seed in seeds:
        deg_seq = nx.random_powerlaw_tree_sequence(20, seed=seed, tries=5000)
        G = nx.Graph(nx.configuration_model(deg_seq, seed=seed))
        G.remove_edges_from(nx.selfloop_edges(G))
        _check_edge_connectivity(G)

def test_shell():
    if False:
        return 10
    seeds = [20]
    for seed in seeds:
        constructor = [(12, 70, 0.8), (15, 40, 0.6)]
        G = nx.random_shell_graph(constructor, seed=seed)
        _check_edge_connectivity(G)

def test_karate():
    if False:
        i = 10
        return i + 15
    G = nx.karate_club_graph()
    _check_edge_connectivity(G)

def test_tarjan_bridge():
    if False:
        print('Hello World!')
    ccs = [(1, 2, 4, 3, 1, 4), (5, 6, 7, 5), (8, 9, 10, 8), (17, 18, 16, 15, 17), (11, 12, 14, 13, 11, 14)]
    bridges = [(4, 8), (3, 5), (3, 17)]
    G = nx.Graph(it.chain(*(pairwise(path) for path in ccs + bridges)))
    _check_edge_connectivity(G)

def test_bridge_cc():
    if False:
        for i in range(10):
            print('nop')
    cc2 = [(1, 2, 4, 3, 1, 4), (8, 9, 10, 8), (11, 12, 13, 11)]
    bridges = [(4, 8), (3, 5), (20, 21), (22, 23, 24)]
    G = nx.Graph(it.chain(*(pairwise(path) for path in cc2 + bridges)))
    bridge_ccs = fset(bridge_components(G))
    target_ccs = fset([{1, 2, 3, 4}, {5}, {8, 9, 10}, {11, 12, 13}, {20}, {21}, {22}, {23}, {24}])
    assert bridge_ccs == target_ccs
    _check_edge_connectivity(G)

def test_undirected_aux_graph():
    if False:
        i = 10
        return i + 15
    (a, b, c, d, e, f, g, h, i) = 'abcdefghi'
    paths = [(a, d, b, f, c), (a, e, b), (a, e, b, c, g, b, a), (c, b), (f, g, f), (h, i)]
    G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    aux_graph = EdgeComponentAuxGraph.construct(G)
    components_1 = fset(aux_graph.k_edge_subgraphs(k=1))
    target_1 = fset([{a, b, c, d, e, f, g}, {h, i}])
    assert target_1 == components_1
    alt_1 = fset(nx.k_edge_subgraphs(G, k=1))
    assert alt_1 == components_1
    components_2 = fset(aux_graph.k_edge_subgraphs(k=2))
    target_2 = fset([{a, b, c, d, e, f, g}, {h}, {i}])
    assert target_2 == components_2
    alt_2 = fset(nx.k_edge_subgraphs(G, k=2))
    assert alt_2 == components_2
    components_3 = fset(aux_graph.k_edge_subgraphs(k=3))
    target_3 = fset([{a}, {b, c, f, g}, {d}, {e}, {h}, {i}])
    assert target_3 == components_3
    components_4 = fset(aux_graph.k_edge_subgraphs(k=4))
    target_4 = fset([{a}, {b}, {c}, {d}, {e}, {f}, {g}, {h}, {i}])
    assert target_4 == components_4
    _check_edge_connectivity(G)

def test_local_subgraph_difference():
    if False:
        i = 10
        return i + 15
    paths = [(11, 12, 13, 14, 11, 13, 14, 12), (21, 22, 23, 24, 21, 23, 24, 22), (11, 101, 21), (12, 102, 22), (13, 103, 23), (14, 104, 24)]
    G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    aux_graph = EdgeComponentAuxGraph.construct(G)
    subgraph_ccs = fset(aux_graph.k_edge_subgraphs(3))
    subgraph_target = fset([{101}, {102}, {103}, {104}, {21, 22, 23, 24}, {11, 12, 13, 14}])
    assert subgraph_ccs == subgraph_target
    local_ccs = fset(aux_graph.k_edge_components(3))
    local_target = fset([{101}, {102}, {103}, {104}, {11, 12, 13, 14, 21, 22, 23, 24}])
    assert local_ccs == local_target

def test_local_subgraph_difference_directed():
    if False:
        while True:
            i = 10
    dipaths = [(1, 2, 3, 4, 1), (1, 3, 1)]
    G = nx.DiGraph(it.chain(*[pairwise(path) for path in dipaths]))
    assert fset(nx.k_edge_components(G, k=1)) == fset(nx.k_edge_subgraphs(G, k=1))
    assert fset(nx.k_edge_components(G, k=2)) != fset(nx.k_edge_subgraphs(G, k=2))
    assert fset(nx.k_edge_components(G, k=3)) == fset(nx.k_edge_subgraphs(G, k=3))
    _check_edge_connectivity(G)

def test_triangles():
    if False:
        print('Hello World!')
    paths = [(11, 12, 13, 11), (21, 22, 23, 21), (11, 21)]
    G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    assert fset(nx.k_edge_components(G, k=1)) == fset(nx.k_edge_subgraphs(G, k=1))
    assert fset(nx.k_edge_components(G, k=2)) == fset(nx.k_edge_subgraphs(G, k=2))
    assert fset(nx.k_edge_components(G, k=3)) == fset(nx.k_edge_subgraphs(G, k=3))
    _check_edge_connectivity(G)

def test_four_clique():
    if False:
        return 10
    paths = [(11, 12, 13, 14, 11, 13, 14, 12), (21, 22, 23, 24, 21, 23, 24, 22), (100, 13), (12, 100, 22), (13, 200, 23), (14, 300, 24)]
    G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))
    local_ccs = fset(nx.k_edge_components(G, k=3))
    subgraphs = fset(nx.k_edge_subgraphs(G, k=3))
    assert local_ccs != subgraphs
    clique1 = frozenset(paths[0])
    clique2 = frozenset(paths[1])
    assert clique1.union(clique2).union({100}) in local_ccs
    assert clique1 in subgraphs
    assert clique2 in subgraphs
    assert G.degree(100) == 3
    _check_edge_connectivity(G)

def test_five_clique():
    if False:
        print('Hello World!')
    G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    paths = [(1, 100, 6), (2, 100, 7), (3, 200, 8), (4, 200, 100)]
    G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    assert min(dict(nx.degree(G)).values()) == 4
    assert fset(nx.k_edge_components(G, k=3)) == fset(nx.k_edge_subgraphs(G, k=3))
    assert fset(nx.k_edge_components(G, k=4)) != fset(nx.k_edge_subgraphs(G, k=4))
    assert fset(nx.k_edge_components(G, k=5)) != fset(nx.k_edge_subgraphs(G, k=5))
    assert fset(nx.k_edge_components(G, k=6)) == fset(nx.k_edge_subgraphs(G, k=6))
    _check_edge_connectivity(G)

def test_directed_aux_graph():
    if False:
        while True:
            i = 10
    (a, b, c, d, e, f, g, h, i) = 'abcdefghi'
    dipaths = [(a, d, b, f, c), (a, e, b), (a, e, b, c, g, b, a), (c, b), (f, g, f), (h, i)]
    G = nx.DiGraph(it.chain(*[pairwise(path) for path in dipaths]))
    aux_graph = EdgeComponentAuxGraph.construct(G)
    components_1 = fset(aux_graph.k_edge_subgraphs(k=1))
    target_1 = fset([{a, b, c, d, e, f, g}, {h}, {i}])
    assert target_1 == components_1
    alt_1 = fset(nx.strongly_connected_components(G))
    assert alt_1 == components_1
    components_2 = fset(aux_graph.k_edge_subgraphs(k=2))
    target_2 = fset([{i}, {e}, {d}, {b, c, f, g}, {h}, {a}])
    assert target_2 == components_2
    components_3 = fset(aux_graph.k_edge_subgraphs(k=3))
    target_3 = fset([{a}, {b}, {c}, {d}, {e}, {f}, {g}, {h}, {i}])
    assert target_3 == components_3

def test_random_gnp_directed():
    if False:
        for i in range(10):
            print('nop')
    seeds = [21]
    for seed in seeds:
        G = nx.gnp_random_graph(20, 0.2, directed=True, seed=seed)
        _check_edge_connectivity(G)

def test_configuration_directed():
    if False:
        i = 10
        return i + 15
    seeds = [67]
    for seed in seeds:
        deg_seq = nx.random_powerlaw_tree_sequence(20, seed=seed, tries=5000)
        G = nx.DiGraph(nx.configuration_model(deg_seq, seed=seed))
        G.remove_edges_from(nx.selfloop_edges(G))
        _check_edge_connectivity(G)

def test_shell_directed():
    if False:
        while True:
            i = 10
    seeds = [31]
    for seed in seeds:
        constructor = [(12, 70, 0.8), (15, 40, 0.6)]
        G = nx.random_shell_graph(constructor, seed=seed).to_directed()
        _check_edge_connectivity(G)

def test_karate_directed():
    if False:
        for i in range(10):
            print('nop')
    G = nx.karate_club_graph().to_directed()
    _check_edge_connectivity(G)