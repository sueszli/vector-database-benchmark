"""
Stoer-Wagner minimum cut algorithm.
"""
from itertools import islice
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for
__all__ = ['stoer_wagner']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def stoer_wagner(G, weight='weight', heap=BinaryHeap):
    if False:
        for i in range(10):
            print('nop')
    'Returns the weighted minimum edge cut using the Stoer-Wagner algorithm.\n\n    Determine the minimum edge cut of a connected graph using the\n    Stoer-Wagner algorithm. In weighted cases, all weights must be\n    nonnegative.\n\n    The running time of the algorithm depends on the type of heaps used:\n\n    ============== =============================================\n    Type of heap   Running time\n    ============== =============================================\n    Binary heap    $O(n (m + n) \\log n)$\n    Fibonacci heap $O(nm + n^2 \\log n)$\n    Pairing heap   $O(2^{2 \\sqrt{\\log \\log n}} nm + n^2 \\log n)$\n    ============== =============================================\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Edges of the graph are expected to have an attribute named by the\n        weight parameter below. If this attribute is not present, the edge is\n        considered to have unit weight.\n\n    weight : string\n        Name of the weight attribute of the edges. If the attribute is not\n        present, unit weight is assumed. Default value: \'weight\'.\n\n    heap : class\n        Type of heap to be used in the algorithm. It should be a subclass of\n        :class:`MinHeap` or implement a compatible interface.\n\n        If a stock heap implementation is to be used, :class:`BinaryHeap` is\n        recommended over :class:`PairingHeap` for Python implementations without\n        optimized attribute accesses (e.g., CPython) despite a slower\n        asymptotic running time. For Python implementations with optimized\n        attribute accesses (e.g., PyPy), :class:`PairingHeap` provides better\n        performance. Default value: :class:`BinaryHeap`.\n\n    Returns\n    -------\n    cut_value : integer or float\n        The sum of weights of edges in a minimum cut.\n\n    partition : pair of node lists\n        A partitioning of the nodes that defines a minimum cut.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the graph is directed or a multigraph.\n\n    NetworkXError\n        If the graph has less than two nodes, is not connected or has a\n        negative-weighted edge.\n\n    Examples\n    --------\n    >>> G = nx.Graph()\n    >>> G.add_edge("x", "a", weight=3)\n    >>> G.add_edge("x", "b", weight=1)\n    >>> G.add_edge("a", "c", weight=3)\n    >>> G.add_edge("b", "c", weight=5)\n    >>> G.add_edge("b", "d", weight=4)\n    >>> G.add_edge("d", "e", weight=2)\n    >>> G.add_edge("c", "y", weight=2)\n    >>> G.add_edge("e", "y", weight=3)\n    >>> cut_value, partition = nx.stoer_wagner(G)\n    >>> cut_value\n    4\n    '
    n = len(G)
    if n < 2:
        raise nx.NetworkXError('graph has less than two nodes.')
    if not nx.is_connected(G):
        raise nx.NetworkXError('graph is not connected.')
    G = nx.Graph(((u, v, {'weight': e.get(weight, 1)}) for (u, v, e) in G.edges(data=True) if u != v))
    for (u, v, e) in G.edges(data=True):
        if e['weight'] < 0:
            raise nx.NetworkXError('graph has a negative-weighted edge.')
    cut_value = float('inf')
    nodes = set(G)
    contractions = []
    for i in range(n - 1):
        u = arbitrary_element(G)
        A = {u}
        h = heap()
        for (v, e) in G[u].items():
            h.insert(v, -e['weight'])
        for j in range(n - i - 2):
            u = h.pop()[0]
            A.add(u)
            for (v, e) in G[u].items():
                if v not in A:
                    h.insert(v, h.get(v, 0) - e['weight'])
        (v, w) = h.min()
        w = -w
        if w < cut_value:
            cut_value = w
            best_phase = i
        contractions.append((u, v))
        for (w, e) in G[v].items():
            if w != u:
                if w not in G[u]:
                    G.add_edge(u, w, weight=e['weight'])
                else:
                    G[u][w]['weight'] += e['weight']
        G.remove_node(v)
    G = nx.Graph(islice(contractions, best_phase))
    v = contractions[best_phase][1]
    G.add_node(v)
    reachable = set(nx.single_source_shortest_path_length(G, v))
    partition = (list(reachable), list(nodes - reachable))
    return (cut_value, partition)