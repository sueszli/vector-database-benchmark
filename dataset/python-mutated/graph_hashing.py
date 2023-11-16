"""
Functions for hashing graphs to strings.
Isomorphic graphs should be assigned identical hashes.
For now, only Weisfeiler-Lehman hashing is implemented.
"""
from collections import Counter, defaultdict
from hashlib import blake2b
import networkx as nx
__all__ = ['weisfeiler_lehman_graph_hash', 'weisfeiler_lehman_subgraph_hashes']

def _hash_label(label, digest_size):
    if False:
        while True:
            i = 10
    return blake2b(label.encode('ascii'), digest_size=digest_size).hexdigest()

def _init_node_labels(G, edge_attr, node_attr):
    if False:
        return 10
    if node_attr:
        return {u: str(dd[node_attr]) for (u, dd) in G.nodes(data=True)}
    elif edge_attr:
        return {u: '' for u in G}
    else:
        return {u: str(deg) for (u, deg) in G.degree()}

def _neighborhood_aggregate(G, node, node_labels, edge_attr=None):
    if False:
        return 10
    "\n    Compute new labels for given node by aggregating\n    the labels of each node's neighbors.\n    "
    label_list = []
    for nbr in G.neighbors(node):
        prefix = '' if edge_attr is None else str(G[node][nbr][edge_attr])
        label_list.append(prefix + node_labels[nbr])
    return node_labels[node] + ''.join(sorted(label_list))

@nx._dispatch(edge_attrs={'edge_attr': None}, node_attrs='node_attr')
def weisfeiler_lehman_graph_hash(G, edge_attr=None, node_attr=None, iterations=3, digest_size=16):
    if False:
        return 10
    'Return Weisfeiler Lehman (WL) graph hash.\n\n    The function iteratively aggregates and hashes neighbourhoods of each node.\n    After each node\'s neighbors are hashed to obtain updated node labels,\n    a hashed histogram of resulting labels is returned as the final hash.\n\n    Hashes are identical for isomorphic graphs and strong guarantees that\n    non-isomorphic graphs will get different hashes. See [1]_ for details.\n\n    If no node or edge attributes are provided, the degree of each node\n    is used as its initial label.\n    Otherwise, node and/or edge labels are used to compute the hash.\n\n    Parameters\n    ----------\n    G: graph\n        The graph to be hashed.\n        Can have node and/or edge attributes. Can also have no attributes.\n    edge_attr: string, default=None\n        The key in edge attribute dictionary to be used for hashing.\n        If None, edge labels are ignored.\n    node_attr: string, default=None\n        The key in node attribute dictionary to be used for hashing.\n        If None, and no edge_attr given, use the degrees of the nodes as labels.\n    iterations: int, default=3\n        Number of neighbor aggregations to perform.\n        Should be larger for larger graphs.\n    digest_size: int, default=16\n        Size (in bits) of blake2b hash digest to use for hashing node labels.\n\n    Returns\n    -------\n    h : string\n        Hexadecimal string corresponding to hash of the input graph.\n\n    Examples\n    --------\n    Two graphs with edge attributes that are isomorphic, except for\n    differences in the edge labels.\n\n    >>> G1 = nx.Graph()\n    >>> G1.add_edges_from(\n    ...     [\n    ...         (1, 2, {"label": "A"}),\n    ...         (2, 3, {"label": "A"}),\n    ...         (3, 1, {"label": "A"}),\n    ...         (1, 4, {"label": "B"}),\n    ...     ]\n    ... )\n    >>> G2 = nx.Graph()\n    >>> G2.add_edges_from(\n    ...     [\n    ...         (5, 6, {"label": "B"}),\n    ...         (6, 7, {"label": "A"}),\n    ...         (7, 5, {"label": "A"}),\n    ...         (7, 8, {"label": "A"}),\n    ...     ]\n    ... )\n\n    Omitting the `edge_attr` option, results in identical hashes.\n\n    >>> nx.weisfeiler_lehman_graph_hash(G1)\n    \'7bc4dde9a09d0b94c5097b219891d81a\'\n    >>> nx.weisfeiler_lehman_graph_hash(G2)\n    \'7bc4dde9a09d0b94c5097b219891d81a\'\n\n    With edge labels, the graphs are no longer assigned\n    the same hash digest.\n\n    >>> nx.weisfeiler_lehman_graph_hash(G1, edge_attr="label")\n    \'c653d85538bcf041d88c011f4f905f10\'\n    >>> nx.weisfeiler_lehman_graph_hash(G2, edge_attr="label")\n    \'3dcd84af1ca855d0eff3c978d88e7ec7\'\n\n    Notes\n    -----\n    To return the WL hashes of each subgraph of a graph, use\n    `weisfeiler_lehman_subgraph_hashes`\n\n    Similarity between hashes does not imply similarity between graphs.\n\n    References\n    ----------\n    .. [1] Shervashidze, Nino, Pascal Schweitzer, Erik Jan Van Leeuwen,\n       Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler Lehman\n       Graph Kernels. Journal of Machine Learning Research. 2011.\n       http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf\n\n    See also\n    --------\n    weisfeiler_lehman_subgraph_hashes\n    '

    def weisfeiler_lehman_step(G, labels, edge_attr=None):
        if False:
            while True:
                i = 10
        '\n        Apply neighborhood aggregation to each node\n        in the graph.\n        Computes a dictionary with labels for each node.\n        '
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            new_labels[node] = _hash_label(label, digest_size)
        return new_labels
    node_labels = _init_node_labels(G, edge_attr, node_attr)
    subgraph_hash_counts = []
    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)
        counter = Counter(node_labels.values())
        subgraph_hash_counts.extend(sorted(counter.items(), key=lambda x: x[0]))
    return _hash_label(str(tuple(subgraph_hash_counts)), digest_size)

@nx._dispatch(edge_attrs={'edge_attr': None}, node_attrs='node_attr')
def weisfeiler_lehman_subgraph_hashes(G, edge_attr=None, node_attr=None, iterations=3, digest_size=16):
    if False:
        while True:
            i = 10
    "\n    Return a dictionary of subgraph hashes by node.\n\n    Dictionary keys are nodes in `G`, and values are a list of hashes.\n    Each hash corresponds to a subgraph rooted at a given node u in `G`.\n    Lists of subgraph hashes are sorted in increasing order of depth from\n    their root node, with the hash at index i corresponding to a subgraph\n    of nodes at most i edges distance from u. Thus, each list will contain\n    ``iterations + 1`` elements - a hash for a subgraph at each depth, and\n    additionally a hash of the initial node label (or equivalently a\n    subgraph of depth 0)\n\n    The function iteratively aggregates and hashes neighbourhoods of each node.\n    This is achieved for each step by replacing for each node its label from\n    the previous iteration with its hashed 1-hop neighborhood aggregate.\n    The new node label is then appended to a list of node labels for each\n    node.\n\n    To aggregate neighborhoods at each step for a node $n$, all labels of\n    nodes adjacent to $n$ are concatenated. If the `edge_attr` parameter is set,\n    labels for each neighboring node are prefixed with the value of this attribute\n    along the connecting edge from this neighbor to node $n$. The resulting string\n    is then hashed to compress this information into a fixed digest size.\n\n    Thus, at the $i$-th iteration, nodes within $i$ hops influence any given\n    hashed node label. We can therefore say that at depth $i$ for node $n$\n    we have a hash for a subgraph induced by the $2i$-hop neighborhood of $n$.\n\n    The output can be used to to create general Weisfeiler-Lehman graph kernels,\n    or generate features for graphs or nodes - for example to generate 'words' in\n    a graph as seen in the 'graph2vec' algorithm.\n    See [1]_ & [2]_ respectively for details.\n\n    Hashes are identical for isomorphic subgraphs and there exist strong\n    guarantees that non-isomorphic graphs will get different hashes.\n    See [1]_ for details.\n\n    If no node or edge attributes are provided, the degree of each node\n    is used as its initial label.\n    Otherwise, node and/or edge labels are used to compute the hash.\n\n    Parameters\n    ----------\n    G: graph\n        The graph to be hashed.\n        Can have node and/or edge attributes. Can also have no attributes.\n    edge_attr: string, default=None\n        The key in edge attribute dictionary to be used for hashing.\n        If None, edge labels are ignored.\n    node_attr: string, default=None\n        The key in node attribute dictionary to be used for hashing.\n        If None, and no edge_attr given, use the degrees of the nodes as labels.\n    iterations: int, default=3\n        Number of neighbor aggregations to perform.\n        Should be larger for larger graphs.\n    digest_size: int, default=16\n        Size (in bits) of blake2b hash digest to use for hashing node labels.\n        The default size is 16 bits\n\n    Returns\n    -------\n    node_subgraph_hashes : dict\n        A dictionary with each key given by a node in G, and each value given\n        by the subgraph hashes in order of depth from the key node.\n\n    Examples\n    --------\n    Finding similar nodes in different graphs:\n\n    >>> G1 = nx.Graph()\n    >>> G1.add_edges_from([\n    ...     (1, 2), (2, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)\n    ... ])\n    >>> G2 = nx.Graph()\n    >>> G2.add_edges_from([\n    ...     (1, 3), (2, 3), (1, 6), (1, 5), (4, 6)\n    ... ])\n    >>> g1_hashes = nx.weisfeiler_lehman_subgraph_hashes(G1, iterations=3, digest_size=8)\n    >>> g2_hashes = nx.weisfeiler_lehman_subgraph_hashes(G2, iterations=3, digest_size=8)\n\n    Even though G1 and G2 are not isomorphic (they have different numbers of edges),\n    the hash sequence of depth 3 for node 1 in G1 and node 5 in G2 are similar:\n\n    >>> g1_hashes[1]\n    ['a93b64973cfc8897', 'db1b43ae35a1878f', '57872a7d2059c1c0']\n    >>> g2_hashes[5]\n    ['a93b64973cfc8897', 'db1b43ae35a1878f', '1716d2a4012fa4bc']\n\n    The first 2 WL subgraph hashes match. From this we can conclude that it's very\n    likely the neighborhood of 4 hops around these nodes are isomorphic: each\n    iteration aggregates 1-hop neighbourhoods meaning hashes at depth $n$ are influenced\n    by every node within $2n$ hops.\n\n    However the neighborhood of 6 hops is no longer isomorphic since their 3rd hash does\n    not match.\n\n    These nodes may be candidates to be classified together since their local topology\n    is similar.\n\n    Notes\n    -----\n    To hash the full graph when subgraph hashes are not needed, use\n    `weisfeiler_lehman_graph_hash` for efficiency.\n\n    Similarity between hashes does not imply similarity between graphs.\n\n    References\n    ----------\n    .. [1] Shervashidze, Nino, Pascal Schweitzer, Erik Jan Van Leeuwen,\n       Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler Lehman\n       Graph Kernels. Journal of Machine Learning Research. 2011.\n       http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf\n    .. [2] Annamalai Narayanan, Mahinthan Chandramohan, Rajasekar Venkatesan,\n       Lihui Chen, Yang Liu and Shantanu Jaiswa. graph2vec: Learning\n       Distributed Representations of Graphs. arXiv. 2017\n       https://arxiv.org/pdf/1707.05005.pdf\n\n    See also\n    --------\n    weisfeiler_lehman_graph_hash\n    "

    def weisfeiler_lehman_step(G, labels, node_subgraph_hashes, edge_attr=None):
        if False:
            i = 10
            return i + 15
        '\n        Apply neighborhood aggregation to each node\n        in the graph.\n        Computes a dictionary with labels for each node.\n        Appends the new hashed label to the dictionary of subgraph hashes\n        originating from and indexed by each node in G\n        '
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            hashed_label = _hash_label(label, digest_size)
            new_labels[node] = hashed_label
            node_subgraph_hashes[node].append(hashed_label)
        return new_labels
    node_labels = _init_node_labels(G, edge_attr, node_attr)
    node_subgraph_hashes = defaultdict(list)
    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, node_subgraph_hashes, edge_attr)
    return dict(node_subgraph_hashes)