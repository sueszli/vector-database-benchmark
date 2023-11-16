"""
Algorithms for asteroidal triples and asteroidal numbers in graphs.

An asteroidal triple in a graph G is a set of three non-adjacent vertices
u, v and w such that there exist a path between any two of them that avoids
closed neighborhood of the third. More formally, v_j, v_k belongs to the same
connected component of G - N[v_i], where N[v_i] denotes the closed neighborhood
of v_i. A graph which does not contain any asteroidal triples is called
an AT-free graph. The class of AT-free graphs is a graph class for which
many NP-complete problems are solvable in polynomial time. Amongst them,
independent set and coloring.
"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['is_at_free', 'find_asteroidal_triple']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def find_asteroidal_triple(G):
    if False:
        for i in range(10):
            print('nop')
    'Find an asteroidal triple in the given graph.\n\n    An asteroidal triple is a triple of non-adjacent vertices such that\n    there exists a path between any two of them which avoids the closed\n    neighborhood of the third. It checks all independent triples of vertices\n    and whether they are an asteroidal triple or not. This is done with the\n    help of a data structure called a component structure.\n    A component structure encodes information about which vertices belongs to\n    the same connected component when the closed neighborhood of a given vertex\n    is removed from the graph. The algorithm used to check is the trivial\n    one, outlined in [1]_, which has a runtime of\n    :math:`O(|V||\\overline{E} + |V||E|)`, where the second term is the\n    creation of the component structure.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        The graph to check whether is AT-free or not\n\n    Returns\n    -------\n    list or None\n        An asteroidal triple is returned as a list of nodes. If no asteroidal\n        triple exists, i.e. the graph is AT-free, then None is returned.\n        The returned value depends on the certificate parameter. The default\n        option is a bool which is True if the graph is AT-free, i.e. the\n        given graph contains no asteroidal triples, and False otherwise, i.e.\n        if the graph contains at least one asteroidal triple.\n\n    Notes\n    -----\n    The component structure and the algorithm is described in [1]_. The current\n    implementation implements the trivial algorithm for simple graphs.\n\n    References\n    ----------\n    .. [1] Ekkehard Köhler,\n       "Recognizing Graphs without asteroidal triples",\n       Journal of Discrete Algorithms 2, pages 439-452, 2004.\n       https://www.sciencedirect.com/science/article/pii/S157086670400019X\n    '
    V = set(G.nodes)
    if len(V) < 6:
        return None
    component_structure = create_component_structure(G)
    E_complement = set(nx.complement(G).edges)
    for e in E_complement:
        u = e[0]
        v = e[1]
        u_neighborhood = set(G[u]).union([u])
        v_neighborhood = set(G[v]).union([v])
        union_of_neighborhoods = u_neighborhood.union(v_neighborhood)
        for w in V - union_of_neighborhoods:
            if component_structure[u][v] == component_structure[u][w] and component_structure[v][u] == component_structure[v][w] and (component_structure[w][u] == component_structure[w][v]):
                return [u, v, w]
    return None

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def is_at_free(G):
    if False:
        while True:
            i = 10
    'Check if a graph is AT-free.\n\n    The method uses the `find_asteroidal_triple` method to recognize\n    an AT-free graph. If no asteroidal triple is found the graph is\n    AT-free and True is returned. If at least one asteroidal triple is\n    found the graph is not AT-free and False is returned.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        The graph to check whether is AT-free or not.\n\n    Returns\n    -------\n    bool\n        True if G is AT-free and False otherwise.\n\n    Examples\n    --------\n    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])\n    >>> nx.is_at_free(G)\n    True\n\n    >>> G = nx.cycle_graph(6)\n    >>> nx.is_at_free(G)\n    False\n    '
    return find_asteroidal_triple(G) is None

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def create_component_structure(G):
    if False:
        print('Hello World!')
    'Create component structure for G.\n\n    A *component structure* is an `nxn` array, denoted `c`, where `n` is\n    the number of vertices,  where each row and column corresponds to a vertex.\n\n    .. math::\n        c_{uv} = \\begin{cases} 0, if v \\in N[u] \\\\\n            k, if v \\in component k of G \\setminus N[u] \\end{cases}\n\n    Where `k` is an arbitrary label for each component. The structure is used\n    to simplify the detection of asteroidal triples.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        Undirected, simple graph.\n\n    Returns\n    -------\n    component_structure : dictionary\n        A dictionary of dictionaries, keyed by pairs of vertices.\n\n    '
    V = set(G.nodes)
    component_structure = {}
    for v in V:
        label = 0
        closed_neighborhood = set(G[v]).union({v})
        row_dict = {}
        for u in closed_neighborhood:
            row_dict[u] = 0
        G_reduced = G.subgraph(set(G.nodes) - closed_neighborhood)
        for cc in nx.connected_components(G_reduced):
            label += 1
            for u in cc:
                row_dict[u] = label
        component_structure[v] = row_dict
    return component_structure