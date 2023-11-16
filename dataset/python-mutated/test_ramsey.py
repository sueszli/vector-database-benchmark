import networkx as nx
import networkx.algorithms.approximation as apxa

def test_ramsey():
    if False:
        while True:
            i = 10
    graph = nx.complete_graph(10)
    (c, i) = apxa.ramsey_R2(graph)
    cdens = nx.density(graph.subgraph(c))
    assert cdens == 1.0, 'clique not correctly found by ramsey!'
    idens = nx.density(graph.subgraph(i))
    assert idens == 0.0, 'i-set not correctly found by ramsey!'
    graph = nx.trivial_graph()
    (c, i) = apxa.ramsey_R2(graph)
    assert c == {0}, 'clique not correctly found by ramsey!'
    assert i == {0}, 'i-set not correctly found by ramsey!'
    graph = nx.barbell_graph(10, 5, nx.Graph())
    (c, i) = apxa.ramsey_R2(graph)
    cdens = nx.density(graph.subgraph(c))
    assert cdens == 1.0, 'clique not correctly found by ramsey!'
    idens = nx.density(graph.subgraph(i))
    assert idens == 0.0, 'i-set not correctly found by ramsey!'
    graph.add_edges_from([(n, n) for n in range(0, len(graph), 2)])
    (cc, ii) = apxa.ramsey_R2(graph)
    assert cc == c
    assert ii == i