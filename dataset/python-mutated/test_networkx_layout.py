from vispy import testing
from vispy.visuals.graphs.layouts import get_layout
from vispy.visuals.graphs.layouts.networkx_layout import NetworkxCoordinates
import numpy as np
try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

def test_networkx_layout_with_graph():
    if False:
        print('Hello World!')
    'Testing the various inputs to the networkx layout.'
    settings = dict(name='networkx_layout')
    if nx is None:
        return testing.SkipTest("'networkx' required")
    graph = nx.complete_graph(5)
    layout = np.random.rand(5, 2)
    settings['graph'] = graph
    settings['layout'] = layout
    testing.assert_true(isinstance(get_layout(**settings), NetworkxCoordinates))
    settings['layout'] = 'circular'
    testing.assert_true(isinstance(get_layout(**settings), NetworkxCoordinates))
    settings['layout'] = nx.circular_layout(graph)
    testing.assert_true(isinstance(get_layout(**settings), NetworkxCoordinates))

def test_networkx_layout_no_networkx():
    if False:
        while True:
            i = 10
    settings = dict(name='networkx_layout')
    testing.assert_raises(ValueError, get_layout, **settings)