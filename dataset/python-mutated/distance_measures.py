"""Distance measures approximated metrics."""
import networkx as nx
from networkx.utils.decorators import py_random_state
__all__ = ['diameter']

@py_random_state(1)
@nx._dispatch(name='approximate_diameter')
def diameter(G, seed=None):
    if False:
        return 10
    'Returns a lower bound on the diameter of the graph G.\n\n    The function computes a lower bound on the diameter (i.e., the maximum eccentricity)\n    of a directed or undirected graph G. The procedure used varies depending on the graph\n    being directed or not.\n\n    If G is an `undirected` graph, then the function uses the `2-sweep` algorithm [1]_.\n    The main idea is to pick the farthest node from a random node and return its eccentricity.\n\n    Otherwise, if G is a `directed` graph, the function uses the `2-dSweep` algorithm [2]_,\n    The procedure starts by selecting a random source node $s$ from which it performs a\n    forward and a backward BFS. Let $a_1$ and $a_2$ be the farthest nodes in the forward and\n    backward cases, respectively. Then, it computes the backward eccentricity of $a_1$ using\n    a backward BFS and the forward eccentricity of $a_2$ using a forward BFS.\n    Finally, it returns the best lower bound between the two.\n\n    In both cases, the time complexity is linear with respect to the size of G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    d : integer\n       Lower Bound on the Diameter of G\n\n    Raises\n    ------\n    NetworkXError\n        If the graph is empty or\n        If the graph is undirected and not connected or\n        If the graph is directed and not strongly connected.\n\n    See Also\n    --------\n    networkx.algorithms.distance_measures.diameter\n\n    References\n    ----------\n    .. [1] Magnien, Clémence, Matthieu Latapy, and Michel Habib.\n       *Fast computation of empirically tight bounds for the diameter of massive graphs.*\n       Journal of Experimental Algorithmics (JEA), 2009.\n       https://arxiv.org/pdf/0904.2728.pdf\n    .. [2] Crescenzi, Pierluigi, Roberto Grossi, Leonardo Lanzi, and Andrea Marino.\n       *On computing the diameter of real-world directed (weighted) graphs.*\n       International Symposium on Experimental Algorithms. Springer, Berlin, Heidelberg, 2012.\n       https://courses.cs.ut.ee/MTAT.03.238/2014_fall/uploads/Main/diameter.pdf\n    '
    if not G:
        raise nx.NetworkXError('Expected non-empty NetworkX graph!')
    if G.number_of_nodes() == 1:
        return 0
    if G.is_directed():
        return _two_sweep_directed(G, seed)
    return _two_sweep_undirected(G, seed)

def _two_sweep_undirected(G, seed):
    if False:
        return 10
    'Helper function for finding a lower bound on the diameter\n        for undirected Graphs.\n\n        The idea is to pick the farthest node from a random node\n        and return its eccentricity.\n\n        ``G`` is a NetworkX undirected graph.\n\n    .. note::\n\n        ``seed`` is a random.Random or numpy.random.RandomState instance\n    '
    source = seed.choice(list(G))
    distances = nx.shortest_path_length(G, source)
    if len(distances) != len(G):
        raise nx.NetworkXError('Graph not connected.')
    (*_, node) = distances
    return nx.eccentricity(G, node)

def _two_sweep_directed(G, seed):
    if False:
        print('Hello World!')
    'Helper function for finding a lower bound on the diameter\n        for directed Graphs.\n\n        It implements 2-dSweep, the directed version of the 2-sweep algorithm.\n        The algorithm follows the following steps.\n        1. Select a source node $s$ at random.\n        2. Perform a forward BFS from $s$ to select a node $a_1$ at the maximum\n        distance from the source, and compute $LB_1$, the backward eccentricity of $a_1$.\n        3. Perform a backward BFS from $s$ to select a node $a_2$ at the maximum\n        distance from the source, and compute $LB_2$, the forward eccentricity of $a_2$.\n        4. Return the maximum between $LB_1$ and $LB_2$.\n\n        ``G`` is a NetworkX directed graph.\n\n    .. note::\n\n        ``seed`` is a random.Random or numpy.random.RandomState instance\n    '
    G_reversed = G.reverse()
    source = seed.choice(list(G))
    forward_distances = nx.shortest_path_length(G, source)
    backward_distances = nx.shortest_path_length(G_reversed, source)
    n = len(G)
    if len(forward_distances) != n or len(backward_distances) != n:
        raise nx.NetworkXError('DiGraph not strongly connected.')
    (*_, a_1) = forward_distances
    (*_, a_2) = backward_distances
    return max(nx.eccentricity(G_reversed, a_1), nx.eccentricity(G, a_2))