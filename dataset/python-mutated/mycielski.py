"""Functions related to the Mycielski Operation and the Mycielskian family
of graphs.

"""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['mycielskian', 'mycielski_graph']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def mycielskian(G, iterations=1):
    if False:
        return 10
    "Returns the Mycielskian of a simple, undirected graph G\n\n    The Mycielskian of graph preserves a graph's triangle free\n    property while increasing the chromatic number by 1.\n\n    The Mycielski Operation on a graph, :math:`G=(V, E)`, constructs a new\n    graph with :math:`2|V| + 1` nodes and :math:`3|E| + |V|` edges.\n\n    The construction is as follows:\n\n    Let :math:`V = {0, ..., n-1}`. Construct another vertex set\n    :math:`U = {n, ..., 2n}` and a vertex, `w`.\n    Construct a new graph, `M`, with vertices :math:`U \\bigcup V \\bigcup w`.\n    For edges, :math:`(u, v) \\in E` add edges :math:`(u, v), (u, v + n)`, and\n    :math:`(u + n, v)` to M. Finally, for all vertices :math:`u \\in U`, add\n    edge :math:`(u, w)` to M.\n\n    The Mycielski Operation can be done multiple times by repeating the above\n    process iteratively.\n\n    More information can be found at https://en.wikipedia.org/wiki/Mycielskian\n\n    Parameters\n    ----------\n    G : graph\n        A simple, undirected NetworkX graph\n    iterations : int\n        The number of iterations of the Mycielski operation to\n        perform on G. Defaults to 1. Must be a non-negative integer.\n\n    Returns\n    -------\n    M : graph\n        The Mycielskian of G after the specified number of iterations.\n\n    Notes\n    -----\n    Graph, node, and edge data are not necessarily propagated to the new graph.\n\n    "
    M = nx.convert_node_labels_to_integers(G)
    for i in range(iterations):
        n = M.number_of_nodes()
        M.add_nodes_from(range(n, 2 * n))
        old_edges = list(M.edges())
        M.add_edges_from(((u, v + n) for (u, v) in old_edges))
        M.add_edges_from(((u + n, v) for (u, v) in old_edges))
        M.add_node(2 * n)
        M.add_edges_from(((u + n, 2 * n) for u in range(n)))
    return M

@nx._dispatch(graphs=None)
def mycielski_graph(n):
    if False:
        i = 10
        return i + 15
    'Generator for the n_th Mycielski Graph.\n\n    The Mycielski family of graphs is an infinite set of graphs.\n    :math:`M_1` is the singleton graph, :math:`M_2` is two vertices with an\n    edge, and, for :math:`i > 2`, :math:`M_i` is the Mycielskian of\n    :math:`M_{i-1}`.\n\n    More information can be found at\n    http://mathworld.wolfram.com/MycielskiGraph.html\n\n    Parameters\n    ----------\n    n : int\n        The desired Mycielski Graph.\n\n    Returns\n    -------\n    M : graph\n        The n_th Mycielski Graph\n\n    Notes\n    -----\n    The first graph in the Mycielski sequence is the singleton graph.\n    The Mycielskian of this graph is not the :math:`P_2` graph, but rather the\n    :math:`P_2` graph with an extra, isolated vertex. The second Mycielski\n    graph is the :math:`P_2` graph, so the first two are hard coded.\n    The remaining graphs are generated using the Mycielski operation.\n\n    '
    if n < 1:
        raise nx.NetworkXError('must satisfy n >= 1')
    if n == 1:
        return nx.empty_graph(1)
    else:
        return mycielskian(nx.path_graph(2), n - 2)