import pytest
pytest.importorskip('numpy')
pytest.importorskip('scipy')
from networkx import is_isomorphic
from networkx.exception import NetworkXError
from networkx.generators import karate_club_graph
from networkx.generators.spectral_graph_forge import spectral_graph_forge
from networkx.utils import nodes_equal

def test_spectral_graph_forge():
    if False:
        for i in range(10):
            print('nop')
    G = karate_club_graph()
    seed = 54321
    H = spectral_graph_forge(G, 0.1, transformation='identity', seed=seed)
    assert nodes_equal(G, H)
    I = spectral_graph_forge(G, 0.1, transformation='identity', seed=seed)
    assert nodes_equal(G, H)
    assert is_isomorphic(I, H)
    I = spectral_graph_forge(G, 0.1, transformation='modularity', seed=seed)
    assert nodes_equal(G, I)
    assert not is_isomorphic(I, H)
    H = spectral_graph_forge(G, 1, transformation='modularity', seed=seed)
    assert nodes_equal(G, H)
    assert is_isomorphic(G, H)
    H = spectral_graph_forge(G, -1, transformation='identity', seed=seed)
    assert nodes_equal(G, H)
    H = spectral_graph_forge(G, 10, transformation='identity', seed=seed)
    assert nodes_equal(G, H)
    assert is_isomorphic(G, H)
    pytest.raises(NetworkXError, spectral_graph_forge, G, 0.1, transformation='unknown', seed=seed)