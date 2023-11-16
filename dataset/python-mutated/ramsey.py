"""
Ramsey numbers.
"""
import networkx as nx
from networkx.utils import not_implemented_for
from ...utils import arbitrary_element
__all__ = ['ramsey_R2']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def ramsey_R2(G):
    if False:
        return 10
    'Compute the largest clique and largest independent set in `G`.\n\n    This can be used to estimate bounds for the 2-color\n    Ramsey number `R(2;s,t)` for `G`.\n\n    This is a recursive implementation which could run into trouble\n    for large recursions. Note that self-loop edges are ignored.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Undirected graph\n\n    Returns\n    -------\n    max_pair : (set, set) tuple\n        Maximum clique, Maximum independent set.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or is a multigraph.\n    '
    if not G:
        return (set(), set())
    node = arbitrary_element(G)
    nbrs = (nbr for nbr in nx.all_neighbors(G, node) if nbr != node)
    nnbrs = nx.non_neighbors(G, node)
    (c_1, i_1) = ramsey_R2(G.subgraph(nbrs).copy())
    (c_2, i_2) = ramsey_R2(G.subgraph(nnbrs).copy())
    c_1.add(node)
    i_2.add(node)
    return (max(c_1, c_2, key=len), max(i_1, i_2, key=len))