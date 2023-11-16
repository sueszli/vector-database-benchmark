import networkx as nx
from networkx.utils.decorators import not_implemented_for, py_random_state
__all__ = ['randomized_partitioning', 'one_exchange']

@not_implemented_for('directed', 'multigraph')
@py_random_state(1)
@nx._dispatch(edge_attrs='weight')
def randomized_partitioning(G, seed=None, p=0.5, weight=None):
    if False:
        return 10
    'Compute a random partitioning of the graph nodes and its cut value.\n\n    A partitioning is calculated by observing each node\n    and deciding to add it to the partition with probability `p`,\n    returning a random cut and its corresponding value (the\n    sum of weights of edges connecting different partitions).\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    p : scalar\n        Probability for each node to be part of the first partition.\n        Should be in [0,1]\n\n    weight : object\n        Edge attribute key to use as weight. If not specified, edges\n        have weight one.\n\n    Returns\n    -------\n    cut_size : scalar\n        Value of the minimum cut.\n\n    partition : pair of node sets\n        A partitioning of the nodes that defines a minimum cut.\n    '
    cut = {node for node in G.nodes() if seed.random() < p}
    cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    partition = (cut, G.nodes - cut)
    return (cut_size, partition)

def _swap_node_partition(cut, node):
    if False:
        i = 10
        return i + 15
    return cut - {node} if node in cut else cut.union({node})

@not_implemented_for('directed', 'multigraph')
@py_random_state(2)
@nx._dispatch(edge_attrs='weight')
def one_exchange(G, initial_cut=None, seed=None, weight=None):
    if False:
        return 10
    'Compute a partitioning of the graphs nodes and the corresponding cut value.\n\n    Use a greedy one exchange strategy to find a locally maximal cut\n    and its value, it works by finding the best node (one that gives\n    the highest gain to the cut value) to add to the current cut\n    and repeats this process until no improvement can be made.\n\n    Parameters\n    ----------\n    G : networkx Graph\n        Graph to find a maximum cut for.\n\n    initial_cut : set\n        Cut to use as a starting point. If not supplied the algorithm\n        starts with an empty cut.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    weight : object\n        Edge attribute key to use as weight. If not specified, edges\n        have weight one.\n\n    Returns\n    -------\n    cut_value : scalar\n        Value of the maximum cut.\n\n    partition : pair of node sets\n        A partitioning of the nodes that defines a maximum cut.\n    '
    if initial_cut is None:
        initial_cut = set()
    cut = set(initial_cut)
    current_cut_size = nx.algorithms.cut_size(G, cut, weight=weight)
    while True:
        nodes = list(G.nodes())
        seed.shuffle(nodes)
        best_node_to_swap = max(nodes, key=lambda v: nx.algorithms.cut_size(G, _swap_node_partition(cut, v), weight=weight), default=None)
        potential_cut = _swap_node_partition(cut, best_node_to_swap)
        potential_cut_size = nx.algorithms.cut_size(G, potential_cut, weight=weight)
        if potential_cut_size > current_cut_size:
            cut = potential_cut
            current_cut_size = potential_cut_size
        else:
            break
    partition = (cut, G.nodes - cut)
    return (current_cut_size, partition)