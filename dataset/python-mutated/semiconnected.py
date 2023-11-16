"""Semiconnectedness."""
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
__all__ = ['is_semiconnected']

@not_implemented_for('undirected')
@nx._dispatch
def is_semiconnected(G):
    if False:
        print('Hello World!')
    'Returns True if the graph is semiconnected, False otherwise.\n\n    A graph is semiconnected if and only if for any pair of nodes, either one\n    is reachable from the other, or they are mutually reachable.\n\n    This function uses a theorem that states that a DAG is semiconnected\n    if for any topological sort, for node $v_n$ in that sort, there is an\n    edge $(v_i, v_{i+1})$. That allows us to check if a non-DAG `G` is\n    semiconnected by condensing the graph: i.e. constructing a new graph `H`\n    with nodes being the strongly connected components of `G`, and edges\n    (scc_1, scc_2) if there is a edge $(v_1, v_2)$ in `G` for some\n    $v_1 \\in scc_1$ and $v_2 \\in scc_2$. That results in a DAG, so we compute\n    the topological sort of `H` and check if for every $n$ there is an edge\n    $(scc_n, scc_{n+1})$.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        A directed graph.\n\n    Returns\n    -------\n    semiconnected : bool\n        True if the graph is semiconnected, False otherwise.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is undirected.\n\n    NetworkXPointlessConcept\n        If the graph is empty.\n\n    Examples\n    --------\n    >>> G = nx.path_graph(4, create_using=nx.DiGraph())\n    >>> print(nx.is_semiconnected(G))\n    True\n    >>> G = nx.DiGraph([(1, 2), (3, 2)])\n    >>> print(nx.is_semiconnected(G))\n    False\n\n    See Also\n    --------\n    is_strongly_connected\n    is_weakly_connected\n    is_connected\n    is_biconnected\n    '
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Connectivity is undefined for the null graph.')
    if not nx.is_weakly_connected(G):
        return False
    H = nx.condensation(G)
    return all((H.has_edge(u, v) for (u, v) in pairwise(nx.topological_sort(H))))