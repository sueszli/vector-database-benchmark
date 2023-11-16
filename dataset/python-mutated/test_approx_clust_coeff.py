import networkx as nx
from networkx.algorithms.approximation import average_clustering

def test_petersen():
    if False:
        return 10
    G = nx.petersen_graph()
    assert average_clustering(G, trials=len(G) // 2) == nx.average_clustering(G)

def test_petersen_seed():
    if False:
        for i in range(10):
            print('nop')
    G = nx.petersen_graph()
    assert average_clustering(G, trials=len(G) // 2, seed=1) == nx.average_clustering(G)

def test_tetrahedral():
    if False:
        while True:
            i = 10
    G = nx.tetrahedral_graph()
    assert average_clustering(G, trials=len(G) // 2) == nx.average_clustering(G)

def test_dodecahedral():
    if False:
        for i in range(10):
            print('nop')
    G = nx.dodecahedral_graph()
    assert average_clustering(G, trials=len(G) // 2) == nx.average_clustering(G)

def test_empty():
    if False:
        for i in range(10):
            print('nop')
    G = nx.empty_graph(5)
    assert average_clustering(G, trials=len(G) // 2) == 0

def test_complete():
    if False:
        i = 10
        return i + 15
    G = nx.complete_graph(5)
    assert average_clustering(G, trials=len(G) // 2) == 1
    G = nx.complete_graph(7)
    assert average_clustering(G, trials=len(G) // 2) == 1