"""
Algorithms for finding k-edge-connected components and subgraphs.

A k-edge-connected component (k-edge-cc) is a maximal set of nodes in G, such
that all pairs of node have an edge-connectivity of at least k.

A k-edge-connected subgraph (k-edge-subgraph) is a maximal set of nodes in G,
such that the subgraph of G defined by the nodes has an edge-connectivity at
least k.
"""
import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
__all__ = ['k_edge_components', 'k_edge_subgraphs', 'bridge_components', 'EdgeComponentAuxGraph']

@not_implemented_for('multigraph')
@nx._dispatch
def k_edge_components(G, k):
    if False:
        i = 10
        return i + 15
    'Generates nodes in each maximal k-edge-connected component in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    k : Integer\n        Desired edge connectivity\n\n    Returns\n    -------\n    k_edge_components : a generator of k-edge-ccs. Each set of returned nodes\n       will have k-edge-connectivity in the graph G.\n\n    See Also\n    --------\n    :func:`local_edge_connectivity`\n    :func:`k_edge_subgraphs` : similar to this function, but the subgraph\n        defined by the nodes must also have k-edge-connectivity.\n    :func:`k_components` : similar to this function, but uses node-connectivity\n        instead of edge-connectivity\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is a multigraph.\n\n    ValueError:\n        If k is less than 1\n\n    Notes\n    -----\n    Attempts to use the most efficient implementation available based on k.\n    If k=1, this is simply connected components for directed graphs and\n    connected components for undirected graphs.\n    If k=2 on an efficient bridge connected component algorithm from _[1] is\n    run based on the chain decomposition.\n    Otherwise, the algorithm from _[2] is used.\n\n    Examples\n    --------\n    >>> import itertools as it\n    >>> from networkx.utils import pairwise\n    >>> paths = [\n    ...     (1, 2, 4, 3, 1, 4),\n    ...     (5, 6, 7, 8, 5, 7, 8, 6),\n    ... ]\n    >>> G = nx.Graph()\n    >>> G.add_nodes_from(it.chain(*paths))\n    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))\n    >>> # note this returns {1, 4} unlike k_edge_subgraphs\n    >>> sorted(map(sorted, nx.k_edge_components(G, k=3)))\n    [[1, 4], [2], [3], [5, 6, 7, 8]]\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Bridge_%28graph_theory%29\n    .. [2] Wang, Tianhao, et al. (2015) A simple algorithm for finding all\n        k-edge-connected components.\n        http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136264\n    '
    if k < 1:
        raise ValueError('k cannot be less than 1')
    if G.is_directed():
        if k == 1:
            return nx.strongly_connected_components(G)
        else:
            aux_graph = EdgeComponentAuxGraph.construct(G)
            return aux_graph.k_edge_components(k)
    elif k == 1:
        return nx.connected_components(G)
    elif k == 2:
        return bridge_components(G)
    else:
        aux_graph = EdgeComponentAuxGraph.construct(G)
        return aux_graph.k_edge_components(k)

@not_implemented_for('multigraph')
@nx._dispatch
def k_edge_subgraphs(G, k):
    if False:
        return 10
    'Generates nodes in each maximal k-edge-connected subgraph in G.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    k : Integer\n        Desired edge connectivity\n\n    Returns\n    -------\n    k_edge_subgraphs : a generator of k-edge-subgraphs\n        Each k-edge-subgraph is a maximal set of nodes that defines a subgraph\n        of G that is k-edge-connected.\n\n    See Also\n    --------\n    :func:`edge_connectivity`\n    :func:`k_edge_components` : similar to this function, but nodes only\n        need to have k-edge-connectivity within the graph G and the subgraphs\n        might not be k-edge-connected.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is a multigraph.\n\n    ValueError:\n        If k is less than 1\n\n    Notes\n    -----\n    Attempts to use the most efficient implementation available based on k.\n    If k=1, or k=2 and the graph is undirected, then this simply calls\n    `k_edge_components`.  Otherwise the algorithm from _[1] is used.\n\n    Examples\n    --------\n    >>> import itertools as it\n    >>> from networkx.utils import pairwise\n    >>> paths = [\n    ...     (1, 2, 4, 3, 1, 4),\n    ...     (5, 6, 7, 8, 5, 7, 8, 6),\n    ... ]\n    >>> G = nx.Graph()\n    >>> G.add_nodes_from(it.chain(*paths))\n    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))\n    >>> # note this does not return {1, 4} unlike k_edge_components\n    >>> sorted(map(sorted, nx.k_edge_subgraphs(G, k=3)))\n    [[1], [2], [3], [4], [5, 6, 7, 8]]\n\n    References\n    ----------\n    .. [1] Zhou, Liu, et al. (2012) Finding maximal k-edge-connected subgraphs\n        from a large graph.  ACM International Conference on Extending Database\n        Technology 2012 480-–491.\n        https://openproceedings.org/2012/conf/edbt/ZhouLYLCL12.pdf\n    '
    if k < 1:
        raise ValueError('k cannot be less than 1')
    if G.is_directed():
        if k <= 1:
            return k_edge_components(G, k)
        else:
            return _k_edge_subgraphs_nodes(G, k)
    elif k <= 2:
        return k_edge_components(G, k)
    else:
        return _k_edge_subgraphs_nodes(G, k)

def _k_edge_subgraphs_nodes(G, k):
    if False:
        i = 10
        return i + 15
    'Helper to get the nodes from the subgraphs.\n\n    This allows k_edge_subgraphs to return a generator.\n    '
    for C in general_k_edge_subgraphs(G, k):
        yield set(C.nodes())

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def bridge_components(G):
    if False:
        return 10
    'Finds all bridge-connected components G.\n\n    Parameters\n    ----------\n    G : NetworkX undirected graph\n\n    Returns\n    -------\n    bridge_components : a generator of 2-edge-connected components\n\n\n    See Also\n    --------\n    :func:`k_edge_subgraphs` : this function is a special case for an\n        undirected graph where k=2.\n    :func:`biconnected_components` : similar to this function, but is defined\n        using 2-node-connectivity instead of 2-edge-connectivity.\n\n    Raises\n    ------\n    NetworkXNotImplemented\n        If the input graph is directed or a multigraph.\n\n    Notes\n    -----\n    Bridge-connected components are also known as 2-edge-connected components.\n\n    Examples\n    --------\n    >>> # The barbell graph with parameter zero has a single bridge\n    >>> G = nx.barbell_graph(5, 0)\n    >>> from networkx.algorithms.connectivity.edge_kcomponents import bridge_components\n    >>> sorted(map(sorted, bridge_components(G)))\n    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]\n    '
    H = G.copy()
    H.remove_edges_from(nx.bridges(G))
    yield from nx.connected_components(H)

class EdgeComponentAuxGraph:
    """A simple algorithm to find all k-edge-connected components in a graph.

    Constructing the auxiliary graph (which may take some time) allows for the
    k-edge-ccs to be found in linear time for arbitrary k.

    Notes
    -----
    This implementation is based on [1]_. The idea is to construct an auxiliary
    graph from which the k-edge-ccs can be extracted in linear time. The
    auxiliary graph is constructed in $O(|V|\\cdot F)$ operations, where F is the
    complexity of max flow. Querying the components takes an additional $O(|V|)$
    operations. This algorithm can be slow for large graphs, but it handles an
    arbitrary k and works for both directed and undirected inputs.

    The undirected case for k=1 is exactly connected components.
    The undirected case for k=2 is exactly bridge connected components.
    The directed case for k=1 is exactly strongly connected components.

    References
    ----------
    .. [1] Wang, Tianhao, et al. (2015) A simple algorithm for finding all
        k-edge-connected components.
        http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136264

    Examples
    --------
    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> from networkx.algorithms.connectivity import EdgeComponentAuxGraph
    >>> # Build an interesting graph with multiple levels of k-edge-ccs
    >>> paths = [
    ...     (1, 2, 3, 4, 1, 3, 4, 2),  # a 3-edge-cc (a 4 clique)
    ...     (5, 6, 7, 5),  # a 2-edge-cc (a 3 clique)
    ...     (1, 5),  # combine first two ccs into a 1-edge-cc
    ...     (0,),  # add an additional disconnected 1-edge-cc
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> # Constructing the AuxGraph takes about O(n ** 4)
    >>> aux_graph = EdgeComponentAuxGraph.construct(G)
    >>> # Once constructed, querying takes O(n)
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=1)))
    [[0], [1, 2, 3, 4, 5, 6, 7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=2)))
    [[0], [1, 2, 3, 4], [5, 6, 7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=3)))
    [[0], [1, 2, 3, 4], [5], [6], [7]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=4)))
    [[0], [1], [2], [3], [4], [5], [6], [7]]

    The auxiliary graph is primarily used for k-edge-ccs but it
    can also speed up the queries of k-edge-subgraphs by refining the
    search space.

    >>> import itertools as it
    >>> from networkx.utils import pairwise
    >>> from networkx.algorithms.connectivity import EdgeComponentAuxGraph
    >>> paths = [
    ...     (1, 2, 4, 3, 1, 4),
    ... ]
    >>> G = nx.Graph()
    >>> G.add_nodes_from(it.chain(*paths))
    >>> G.add_edges_from(it.chain(*[pairwise(path) for path in paths]))
    >>> aux_graph = EdgeComponentAuxGraph.construct(G)
    >>> sorted(map(sorted, aux_graph.k_edge_subgraphs(k=3)))
    [[1], [2], [3], [4]]
    >>> sorted(map(sorted, aux_graph.k_edge_components(k=3)))
    [[1, 4], [2], [3]]
    """

    @classmethod
    def construct(EdgeComponentAuxGraph, G):
        if False:
            return 10
        'Builds an auxiliary graph encoding edge-connectivity between nodes.\n\n        Notes\n        -----\n        Given G=(V, E), initialize an empty auxiliary graph A.\n        Choose an arbitrary source node s.  Initialize a set N of available\n        nodes (that can be used as the sink). The algorithm picks an\n        arbitrary node t from N - {s}, and then computes the minimum st-cut\n        (S, T) with value w. If G is directed the minimum of the st-cut or\n        the ts-cut is used instead. Then, the edge (s, t) is added to the\n        auxiliary graph with weight w. The algorithm is called recursively\n        first using S as the available nodes and s as the source, and then\n        using T and t. Recursion stops when the source is the only available\n        node.\n\n        Parameters\n        ----------\n        G : NetworkX graph\n        '
        not_implemented_for('multigraph')(lambda G: G)(G)

        def _recursive_build(H, A, source, avail):
            if False:
                while True:
                    i = 10
            if {source} == avail:
                return
            sink = arbitrary_element(avail - {source})
            (value, (S, T)) = nx.minimum_cut(H, source, sink)
            if H.is_directed():
                (value_, (T_, S_)) = nx.minimum_cut(H, sink, source)
                if value_ < value:
                    (value, S, T) = (value_, S_, T_)
            A.add_edge(source, sink, weight=value)
            _recursive_build(H, A, source, avail.intersection(S))
            _recursive_build(H, A, sink, avail.intersection(T))
        H = G.__class__()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges(), capacity=1)
        A = nx.Graph()
        if H.number_of_nodes() > 0:
            source = arbitrary_element(H.nodes())
            avail = set(H.nodes())
            _recursive_build(H, A, source, avail)
        self = EdgeComponentAuxGraph()
        self.A = A
        self.H = H
        return self

    def k_edge_components(self, k):
        if False:
            for i in range(10):
                print('nop')
        'Queries the auxiliary graph for k-edge-connected components.\n\n        Parameters\n        ----------\n        k : Integer\n            Desired edge connectivity\n\n        Returns\n        -------\n        k_edge_components : a generator of k-edge-ccs\n\n        Notes\n        -----\n        Given the auxiliary graph, the k-edge-connected components can be\n        determined in linear time by removing all edges with weights less than\n        k from the auxiliary graph.  The resulting connected components are the\n        k-edge-ccs in the original graph.\n        '
        if k < 1:
            raise ValueError('k cannot be less than 1')
        A = self.A
        aux_weights = nx.get_edge_attributes(A, 'weight')
        R = nx.Graph()
        R.add_nodes_from(A.nodes())
        R.add_edges_from((e for (e, w) in aux_weights.items() if w >= k))
        yield from nx.connected_components(R)

    def k_edge_subgraphs(self, k):
        if False:
            for i in range(10):
                print('nop')
        'Queries the auxiliary graph for k-edge-connected subgraphs.\n\n        Parameters\n        ----------\n        k : Integer\n            Desired edge connectivity\n\n        Returns\n        -------\n        k_edge_subgraphs : a generator of k-edge-subgraphs\n\n        Notes\n        -----\n        Refines the k-edge-ccs into k-edge-subgraphs. The running time is more\n        than $O(|V|)$.\n\n        For single values of k it is faster to use `nx.k_edge_subgraphs`.\n        But for multiple values of k, it can be faster to build AuxGraph and\n        then use this method.\n        '
        if k < 1:
            raise ValueError('k cannot be less than 1')
        H = self.H
        A = self.A
        aux_weights = nx.get_edge_attributes(A, 'weight')
        R = nx.Graph()
        R.add_nodes_from(A.nodes())
        R.add_edges_from((e for (e, w) in aux_weights.items() if w >= k))
        for cc in nx.connected_components(R):
            if len(cc) < k:
                for node in cc:
                    yield {node}
            else:
                C = H.subgraph(cc)
                yield from k_edge_subgraphs(C, k)

def _low_degree_nodes(G, k, nbunch=None):
    if False:
        for i in range(10):
            print('nop')
    'Helper for finding nodes with degree less than k.'
    if G.is_directed():
        seen = set()
        for (node, degree) in G.out_degree(nbunch):
            if degree < k:
                seen.add(node)
                yield node
        for (node, degree) in G.in_degree(nbunch):
            if node not in seen and degree < k:
                seen.add(node)
                yield node
    else:
        for (node, degree) in G.degree(nbunch):
            if degree < k:
                yield node

def _high_degree_components(G, k):
    if False:
        i = 10
        return i + 15
    "Helper for filtering components that can't be k-edge-connected.\n\n    Removes and generates each node with degree less than k.  Then generates\n    remaining components where all nodes have degree at least k.\n    "
    H = G.copy()
    singletons = set(_low_degree_nodes(H, k))
    while singletons:
        nbunch = set(it.chain.from_iterable(map(H.neighbors, singletons)))
        nbunch.difference_update(singletons)
        H.remove_nodes_from(singletons)
        for node in singletons:
            yield {node}
        singletons = set(_low_degree_nodes(H, k, nbunch))
    if G.is_directed():
        yield from nx.strongly_connected_components(H)
    else:
        yield from nx.connected_components(H)

@nx._dispatch
def general_k_edge_subgraphs(G, k):
    if False:
        while True:
            i = 10
    'General algorithm to find all maximal k-edge-connected subgraphs in G.\n\n    Returns\n    -------\n    k_edge_subgraphs : a generator of nx.Graphs that are k-edge-subgraphs\n        Each k-edge-subgraph is a maximal set of nodes that defines a subgraph\n        of G that is k-edge-connected.\n\n    Notes\n    -----\n    Implementation of the basic algorithm from _[1].  The basic idea is to find\n    a global minimum cut of the graph. If the cut value is at least k, then the\n    graph is a k-edge-connected subgraph and can be added to the results.\n    Otherwise, the cut is used to split the graph in two and the procedure is\n    applied recursively. If the graph is just a single node, then it is also\n    added to the results. At the end, each result is either guaranteed to be\n    a single node or a subgraph of G that is k-edge-connected.\n\n    This implementation contains optimizations for reducing the number of calls\n    to max-flow, but there are other optimizations in _[1] that could be\n    implemented.\n\n    References\n    ----------\n    .. [1] Zhou, Liu, et al. (2012) Finding maximal k-edge-connected subgraphs\n        from a large graph.  ACM International Conference on Extending Database\n        Technology 2012 480-–491.\n        https://openproceedings.org/2012/conf/edbt/ZhouLYLCL12.pdf\n\n    Examples\n    --------\n    >>> from networkx.utils import pairwise\n    >>> paths = [\n    ...     (11, 12, 13, 14, 11, 13, 14, 12),  # a 4-clique\n    ...     (21, 22, 23, 24, 21, 23, 24, 22),  # another 4-clique\n    ...     # connect the cliques with high degree but low connectivity\n    ...     (50, 13),\n    ...     (12, 50, 22),\n    ...     (13, 102, 23),\n    ...     (14, 101, 24),\n    ... ]\n    >>> G = nx.Graph(it.chain(*[pairwise(path) for path in paths]))\n    >>> sorted(map(len, k_edge_subgraphs(G, k=3)))\n    [1, 1, 1, 4, 4]\n    '
    if k < 1:
        raise ValueError('k cannot be less than 1')
    find_ccs = partial(_high_degree_components, k=k)
    if G.number_of_nodes() < k:
        for node in G.nodes():
            yield G.subgraph([node]).copy()
        return
    R0 = {G.subgraph(cc).copy() for cc in find_ccs(G)}
    while R0:
        G1 = R0.pop()
        if G1.number_of_nodes() == 1:
            yield G1
        else:
            cut_edges = nx.minimum_edge_cut(G1)
            cut_value = len(cut_edges)
            if cut_value < k:
                G1.remove_edges_from(cut_edges)
                for cc in find_ccs(G1):
                    R0.add(G1.subgraph(cc).copy())
            else:
                yield G1