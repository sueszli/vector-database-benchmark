import pytest
import networkx as nx
from networkx import Graph, NetworkXError
from networkx.algorithms.community import asyn_fluidc

def test_exceptions():
    if False:
        while True:
            i = 10
    test = Graph()
    test.add_node('a')
    pytest.raises(NetworkXError, asyn_fluidc, test, 'hi')
    pytest.raises(NetworkXError, asyn_fluidc, test, -1)
    pytest.raises(NetworkXError, asyn_fluidc, test, 3)
    test.add_node('b')
    pytest.raises(NetworkXError, asyn_fluidc, test, 1)

def test_single_node():
    if False:
        while True:
            i = 10
    test = Graph()
    test.add_node('a')
    ground_truth = {frozenset(['a'])}
    communities = asyn_fluidc(test, 1)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth

def test_two_nodes():
    if False:
        for i in range(10):
            print('nop')
    test = Graph()
    test.add_edge('a', 'b')
    ground_truth = {frozenset(['a']), frozenset(['b'])}
    communities = asyn_fluidc(test, 2)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth

def test_two_clique_communities():
    if False:
        print('Hello World!')
    test = Graph()
    test.add_edge('a', 'b')
    test.add_edge('a', 'c')
    test.add_edge('b', 'c')
    test.add_edge('c', 'd')
    test.add_edge('d', 'e')
    test.add_edge('d', 'f')
    test.add_edge('f', 'e')
    ground_truth = {frozenset(['a', 'c', 'b']), frozenset(['e', 'd', 'f'])}
    communities = asyn_fluidc(test, 2, seed=7)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth

def test_five_clique_ring():
    if False:
        while True:
            i = 10
    test = Graph()
    test.add_edge('1a', '1b')
    test.add_edge('1a', '1c')
    test.add_edge('1a', '1d')
    test.add_edge('1b', '1c')
    test.add_edge('1b', '1d')
    test.add_edge('1c', '1d')
    test.add_edge('2a', '2b')
    test.add_edge('2a', '2c')
    test.add_edge('2a', '2d')
    test.add_edge('2b', '2c')
    test.add_edge('2b', '2d')
    test.add_edge('2c', '2d')
    test.add_edge('3a', '3b')
    test.add_edge('3a', '3c')
    test.add_edge('3a', '3d')
    test.add_edge('3b', '3c')
    test.add_edge('3b', '3d')
    test.add_edge('3c', '3d')
    test.add_edge('4a', '4b')
    test.add_edge('4a', '4c')
    test.add_edge('4a', '4d')
    test.add_edge('4b', '4c')
    test.add_edge('4b', '4d')
    test.add_edge('4c', '4d')
    test.add_edge('5a', '5b')
    test.add_edge('5a', '5c')
    test.add_edge('5a', '5d')
    test.add_edge('5b', '5c')
    test.add_edge('5b', '5d')
    test.add_edge('5c', '5d')
    test.add_edge('1a', '2c')
    test.add_edge('2a', '3c')
    test.add_edge('3a', '4c')
    test.add_edge('4a', '5c')
    test.add_edge('5a', '1c')
    ground_truth = {frozenset(['1a', '1b', '1c', '1d']), frozenset(['2a', '2b', '2c', '2d']), frozenset(['3a', '3b', '3c', '3d']), frozenset(['4a', '4b', '4c', '4d']), frozenset(['5a', '5b', '5c', '5d'])}
    communities = asyn_fluidc(test, 5, seed=9)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth